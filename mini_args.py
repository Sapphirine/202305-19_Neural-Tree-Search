# For testing 
from dataclasses import dataclass

@dataclass
class Arg:
    task_config="tasks/pointnav_gibson.yaml"
    split="val"
    num_processes=1
    num_processes_on_first_gpu=1
    num_processes_per_gpu=0
    sim_gpu_id=1
    max_episode_length=1000
    max_training_steps=20000
    num_episodes=1
    env_frame_width=256
    env_frame_height=256
    hfov=90.0
    camera_height=1.25  # above required by construct_envs
    visualize=0
    print_images=1
    frame_height=128  
    frame_width=128  
    map_resolution=5
    map_size_cm=2400
    du_scale=2
    vision_range=64
    vis_type=2
    obstacle_boundary=5
    obs_threshold=1  # above required by VectorEnv
    randomize_env_every=1000  # required by reset()
    global_downscaling=2
    noisy_actions=1
    noisy_odometry=1
    noise_level=1.0
    num_local_steps=25
    num_global_steps=2
    short_goal_dist=1
    eval=1
    eval_temperature=0.2
    collision_threshold=0.2
    num_update_per_episode=50
    num_simulations=50
    num_trajectory=4
    sample_per_trajectory=16
    k_steps=5
    action_width=4
    action_height=4
    buffer_capacity=50
    n_bootstrapping=10
    log_interval=1
    save_trajectory_data='0'
    save_interval=1
    save_periodic=500000
    split_key=16
    dump_location='./tmp/'
    exp_name='nts4_eval_img'
    load_model='/home/fangbowen/Neural-Tree-Search/tmp/models/nts4/model_best.npy'
    
