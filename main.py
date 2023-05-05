import os
import time 
from collections import deque
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.7'
import jax 
import numpy as np  

from mini_args import Arg
from env.habitat import construct_envs
from env import make_vec_envs 
from nts.wrappers import NHWCWrapper
from nts.utils import init_map_and_pose, get_local_map_boundaries, action2goal

import muax
from nts.model import NTS
from nts.nn import Decoder, PoseEstimator, _init_resnet_decoder_func, _init_resnet_pose_func
from muax.nn import ResNetRepresentation, ResNetPrediction, ResNetDynamic, _init_resnet_representation_func, _init_resnet_prediction_func, _init_resnet_dynamic_func
from nts.episode_tracer import NAVTransition, NAVPNStep
from nts.replay_buffer import NAVTrajectory, NAVTrajectoryReplayBuffer
from nts.loss import nts_loss_fn


def temperature_fn(max_training_steps, training_steps):
  r"""Determines the randomness for the action taken by the model"""
  if training_steps < 0.5 * max_training_steps:
      return 1.0
  elif training_steps < 0.75 * max_training_steps:
      return 0.5
  else:
      return 0.25
  

def main():
    args = Arg()
    
    envs = construct_envs(args)
    envs = NHWCWrapper(envs)

    action_width = args.action_width
    action_height = args.action_height
    num_actions = int(args.action_width * args.action_height)
    support_size = 10
    pred_channels = input_channels = 32
    output_channels = input_channels * 2
    full_support_size = int(support_size * 2 + 1)
    repr_fn = _init_resnet_representation_func(ResNetRepresentation, input_channels)
    pred_fn = _init_resnet_prediction_func(ResNetPrediction, num_actions, full_support_size, pred_channels)
    dy_fn = _init_resnet_dynamic_func(ResNetDynamic, num_actions, full_support_size, output_channels)

    dec_fn = _init_resnet_decoder_func(Decoder)
    pose_fn = _init_resnet_pose_func(PoseEstimator)

    discount = 0.997
    gradient_transform = muax.model.optimizer(init_value=1e-3, peak_value=2e-3, end_value=1e-3, warmup_steps=5000, transition_steps=5000)

    model = NTS(args,
                dec_fn=dec_fn, 
                pose_fn=pose_fn, 
                representation_fn=repr_fn, 
                prediction_fn=pred_fn, 
                dynamic_fn=dy_fn, 
                optimizer=gradient_transform,
                loss_fn=nts_loss_fn,
                discount=discount, 
                support_size=support_size)

    num_scenes = args.num_processes
    buffer = NAVTrajectoryReplayBuffer(args.buffer_capacity)
    tracers = [NAVPNStep(args.n_bootstrapping, discount) for _ in range(num_scenes)]

    obs, info = envs.reset()
    key = jax.random.PRNGKey(0)
    model.init(key, obs)

    if args.load_model != '0':
        print(f'loading model params from {args.load_model}')
        model.load(args.load_model)
    
    log_dir = os.path.join(args.dump_location, 'models', args.exp_name)
    dump_dir = os.path.join(args.dump_location, 'dump', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(dump_dir, exist_ok=True)
    num_scenes = args.num_processes
    # a train loop
    g_episode_rewards = deque()
    train_losses = deque()
    
    traj_lengths = args.max_episode_length // args.num_local_steps
    explored_area_log = np.zeros((num_scenes, args.num_episodes, traj_lengths))
    explored_ratio_log = np.zeros((num_scenes, args.num_episodes, traj_lengths))
    
    obs, infos = envs.reset()
    obs_last = np.array(obs)
    full_map, local_map, full_pose, local_pose, origins, lmb, planner_pose_inputs = init_map_and_pose(args)
    poses = np.asarray([info[i]['sensor_pose'] for i in range(args.num_processes)])

    map_pred, exp_pred, current_poses, pose_input = model.build_map(obs_last, obs, poses, local_map[:, :, :, 0], local_map[:, :, :, 1], local_pose)
    local_map[:, :, :, 0] = map_pred
    local_map[:, :, :, 1] = exp_pred 
    local_pose = current_poses

    # Compute planner inputs
    a, pi, v = model.act(key, obs, with_pi=True, with_value=True,
                        obs_from_batch=True, 
                        num_simulations=args.num_simulations,
                        temperature=1 if not args.eval else args.eval_temperature)

    global_goals = action2goal(a, local_map.shape[1], local_map.shape[2], action_width, action_height)
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals[e]
        p_input['map_pred'] = local_map[e, :, :, 0]
        p_input['exp_pred'] = local_map[e, :, :, 1]
        p_input['pose_pred'] = planner_pose_inputs[e]
    # Output stores local goals as well as the the ground-truth action
    output = envs.get_short_term_goal(planner_inputs)

    total_num_steps = -1
    training_step = 0
    best_g_reward = -float('inf')
    start = time.time()

    # test jax random
    for _ in range(args.split_key):
        key, subkey = jax.random.split(key, num=2)

    num_episodes = int(args.num_episodes)
    for ep_num in range(num_episodes):
        key, subkey = jax.random.split(key, num=2)
        Gs = np.zeros((num_scenes,))
        for e in range(num_scenes):
            tracers[e].reset()
        trajectories = [NAVTrajectory() for _ in range(num_scenes)]
        if not args.eval:
            temperature = temperature_fn(max_training_steps=args.max_training_steps, training_steps=training_step)
        else:
            temperature = args.eval_temperature
        _exp_r = np.zeros((num_scenes,))

        for step in range(args.max_episode_length):
            subkey, rng_key = jax.random.split(subkey)
            total_num_steps += 1
            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps

            obs_last = np.array(obs)
            planner_action = output[:, -1].astype(int)
            obs, r, done, info = envs.step(planner_action)

            # last step
            if step == args.max_episode_length - 1:
                full_map, local_map, full_pose, local_pose, origins, lmb, planner_pose_inputs = init_map_and_pose(args)
                obs_last = np.array(obs)
            
            poses = np.asarray([info[i]['sensor_pose'] for i in range(args.num_processes)])
            map_pred, exp_pred, current_poses, pose_input = model.build_map(obs_last, obs, poses, local_map[:, :, :, 0], local_map[:, :, :, 1], local_pose)
            local_map[:, :, :, 0] = map_pred
            local_map[:, :, :, 1] = exp_pred 
            local_pose = current_poses
            locs = np.array(local_pose)
            planner_pose_inputs[:, :3] = locs + origins
            local_map[:, :, :, 2] = 0
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100. / args.map_resolution), int(c * 100. / args.map_resolution)]
                local_map[e, loc_r - 2: loc_r + 3, loc_c - 2: loc_c + 3, 2:] = 1.
            
            if l_step == args.num_local_steps - 1:
                exp_ratio = np.asarray([info[e]['exp_ratio'] for e in range(num_scenes)])
                _exp_r += exp_ratio
                print(f'explore ratio: {_exp_r}, epoch step: {step}, ep: {ep_num}')
                for e in range(num_scenes):
                    full_map[e, lmb[e, 0]: lmb[e, 1], lmb[e, 2]: lmb[e, 3], :] = local_map[e]
                    full_pose[e] = local_pose[e] + np.asarray(origins[e], dtype=float)
                    locs = np.asarray(full_pose[e])
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]
                    full_w, full_h = full_map.shape[1], full_map.shape[2]
                    lmb[e] = get_local_map_boundaries((loc_r, loc_c), (local_map.shape[1], local_map.shape[2]), (full_w, full_h), args)
                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0, lmb[e][0] * args.map_resolution / 100.0, 0.]
                    local_map[e] = full_map[e, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3], :]
                    local_pose[e] = full_pose[e] - np.asarray(origins[e], dtype=float)
                locs = np.asarray(local_pose)
                a, pi, v = model.act(rng_key, obs, with_pi=True, with_value=True,
                                    obs_from_batch=True, 
                                    num_simulations=args.num_simulations,
                                    temperature=temperature)
                global_goals = action2goal(a, local_map.shape[1], local_map.shape[2], action_width, action_height)
                
                # insert transition 
                for e in range(num_scenes):
                    tracers[e].add(obs_last=obs_last[e], obs=obs[e], poses=np.asarray(info[e]['sensor_pose']), pose_input=pose_input[e],
                                   fp_proj=info[e]['fp_proj'], fp_explored=info[e]['fp_explored'], pose_err=np.asarray(info[e]['pose_err']), 
                                   a=a[e], r=info[e]['exp_reward'], done=done[e], v=v[e], pi=pi[e])
                    while tracers[e]:
                        trans = tracers[e].pop()
                        trajectories[e].add(trans)
                    
                    Gs[e] += info[e]['exp_reward']  

                if np.sum(Gs) != 0:
                    for G in Gs:
                        g_episode_rewards.append(G) if G != 0 else None 
                
                exp_ratio = np.asarray([info[e]['exp_ratio'] for e in range(num_scenes)])
                for e in range(num_scenes):
                    explored_area_log[e, ep_num, eval_g_step - 1] = explored_area_log[e, ep_num, eval_g_step - 2] + Gs[e] * 50.  # Convert reward to area in m2, same as nslam
                    explored_ratio_log[e, ep_num, eval_g_step - 1] = explored_ratio_log[e, ep_num, eval_g_step - 2] + exp_ratio[e]

            # planning
            
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['goal'] = global_goals[e]
                p_input['map_pred'] = local_map[e, :, :, 0]
                p_input['exp_pred'] = local_map[e, :, :, 1]
                p_input['pose_pred'] = planner_pose_inputs[e]
            # Output stores local goals as well as the the ground-truth action
            output = envs.get_short_term_goal(planner_inputs)
            
            # ------------------------------------------------------------------
            # Train 
            # print(g_step % args.num_global_steps, args.num_global_steps - 1, l_step, args.num_local_steps - 1)
            train_loss = 0
            if g_step % args.num_global_steps == args.num_global_steps - 1 and l_step == args.num_local_steps - 1 and len(buffer) >= args.num_trajectory and not args.eval:
                print(f'training... training step: {training_step}, epoch step: {step}, epoch: {ep_num}')
                for _ in range(args.num_update_per_episode):
                    transition_batch = buffer.sample(num_trajectory=args.num_trajectory,
                                            sample_per_trajectory=args.sample_per_trajectory,
                                            k_steps=args.k_steps)
                    loss_metric = model.update(transition_batch)
                    train_loss += loss_metric['loss']
                    training_step += 1
                train_loss /= args.num_update_per_episode
                train_losses.append(train_loss)
                print(f'train loss" {train_loss}')
        
        for e in range(num_scenes):
            trajectories[e].finalize()
            # print(len(trajectories[e]))
            buffer.add(trajectory=trajectories[e], w=trajectories[e].batched_transitions.w.mean())
        # print(len(buffer))
        # ------------------------------------------------------------------
        # Logging
        if total_num_steps % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(total_num_steps * num_scenes),
                "FPS {},".format(int(total_num_steps * num_scenes / (end - start)))
            ])

            log += "\n\tRewards:"
            if len(g_episode_rewards) > 0:
                log += " ".join([
                            " Global eps mean/med/min/max eps reward:",
                            "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                                np.mean(g_episode_rewards),
                                np.median(g_episode_rewards),
                                np.min(g_episode_rewards),
                                np.max(g_episode_rewards))
                        ])
            
            log += "\n\tLosses:"
            if len(train_losses) > 0:
                log += " ".join([" loss:", "{:.3f},".format(np.mean(train_losses))])
            
            print(log)
        
        # ------------------------------------------------------------------
        # Save best models
        if (total_num_steps * num_scenes) % args.save_interval < num_scenes:
            if len(g_episode_rewards) >= 100 and (np.mean(g_episode_rewards) >= best_g_reward) and not args.eval:
                        model.save(os.path.join(log_dir, "model_best"))
                        best_g_reward = np.mean(g_episode_rewards)
        # Save periodic models
        if (total_num_steps * num_scenes) % args.save_periodic < num_scenes:
                    step = total_num_steps * num_scenes
                    model.save(os.path.join(dump_dir, f'periodic_{step}_model'))

    
    logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
    for e in range(num_scenes):
        for i in range(explored_area_log[e].shape[0]):
            logfile.write(str(explored_area_log[e, i]) + "\n")
            logfile.flush()

    logfile.close()

    logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
    for e in range(num_scenes):
        for i in range(explored_ratio_log[e].shape[0]):
            logfile.write(str(explored_ratio_log[e, i]) + "\n")
            logfile.flush()

    logfile.close()

    log = "Final Exp Area: \n"
    for i in range(explored_area_log.shape[2]):
        log += "{:.5f}, ".format(
            np.mean(explored_area_log[:, :, i]))

    log += "\nFinal Exp Ratio: \n"
    for i in range(explored_ratio_log.shape[2]):
        log += "{:.5f}, ".format(
            np.mean(explored_ratio_log[:, :, i]))

    print(log)

if __name__ == '__main__':
    main()