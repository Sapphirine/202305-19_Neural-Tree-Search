import functools
from functools import partial
import numpy as np 
import jax 
from jax import numpy as jnp 

import haiku as hk 

from muax import MuZero 
from .utils import get_grid, bilinear_sampler


class NTS(MuZero):
    def __init__(self, cargs, dec_fn, pose_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec_func = hk.without_apply_rng(hk.transform(dec_fn))
        self.pose_func = hk.without_apply_rng(hk.transform(pose_fn))
        self.screen_h = cargs.frame_height
        self.screen_w = cargs.frame_width
        self.resolution = cargs.map_resolution
        self.map_size_cm = cargs.map_size_cm // cargs.global_downscaling
        self.num_processes = cargs.num_processes
        self.vision_range = cargs.vision_range
        self.n_channels = 3
        self.dropout = 0.5
    
    def init(self, rng_key, sample_input):
        """Inits `representation`, `prediction` and `dynamic` functions and optimizer
        
        Parameters
        ----------
        rng_key: jax.random.PRNGKey.
        sample_input: Array. The dimension is `[B, ...]` where B is the batch dimension.
        
        Returns
        ----------
        params: dict. {'representation': repr_params, 'prediction': pred_params, 'dynamic': dy_params}
        """
        repr_params = self.repr_func.init(rng_key, sample_input)
        s = self.repr_func.apply(repr_params, sample_input)
        dec_params = self.dec_func.init(rng_key, s)
        dec = self.dec_func.apply(dec_params, s)
        pose_params = self.pose_func.init(rng_key, np.concatenate([np.array(dec), np.array(dec)], axis=-1))
        pred_params = self.pred_func.init(rng_key, s)
        dy_params = self.dy_func.init(rng_key, s, jnp.zeros(s.shape[0]))
        self._params = {'representation': repr_params, 
                    'prediction': pred_params, 
                    'dynamic': dy_params,
                    'decoder': dec_params,
                    'pose': pose_params}
        self._opt_state = self._optimizer.init(self._params)
        return self._params

    def decoder(self, s):
        ds = self._dec_apply(self.params['decoder'], s)
        return ds 

    def pose_estimator(self, pose_input):
        pose_pred = self._pose_apply(self.params['pose'], pose_input) 
        return pose_pred 
    
    def slam(self, obs_last, obs, poses):
        # Get egocentric map prediction for the current obs
        # obs = np.moveaxis(obs, 1, -1)
        # obs_last = np.moveaxis(obs_last, 1, -1)
        bs, h, w, c = obs.shape
        encoded_state = self.representation(obs)
        decoded_state = self.decoder(encoded_state)
        # proj_pred = decoded_state[:, :, :, :1]
        # fp_exp_pred = decoded_state[:, :, :, 1:] 

        pred_last_st = self._get_st_prediction(obs_last, poses)

        pose_input = np.concatenate([np.array(jax.nn.sigmoid(decoded_state)), np.array(pred_last_st)], axis=-1)
        pose_pred = self.pose_estimator(pose_input)

        return decoded_state, pose_pred, pose_input

    def build_map(self, obs_last, obs, poses, maps, explored, current_poses):
        # Aggregate egocentric map prediction in the geocentric map
        # using the predicted pose
        decoded_state, pose_pred, pose_input = self.slam(obs_last, obs, poses)
        decoded_state = np.array(decoded_state)
        pose_pred = np.array(pose_pred)
        agent_view = np.zeros((self.num_processes,
                               self.map_size_cm // self.resolution,
                               self.map_size_cm // self.resolution,
                               2))
        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, y1:y2, x1:x2, :] = decoded_state 

        corrected_pose = poses + pose_pred
        current_poses = self._get_new_pose(current_poses, corrected_pose)
        map_pred, exp_pred = self._get_map_prediction(current_poses, agent_view, maps, explored)
        return map_pred, exp_pred, current_poses, pose_input

    def _get_st_prediction(self, obs_last, poses):
        bs, h, w, c = obs_last.shape
        # Get egocentric map prediction for the last obs
        obs_last = jax.lax.stop_gradient(obs_last)
        encoded_last = self.representation(obs_last)
        decoded_last = self.decoder(encoded_last)
        decoded_last = np.array(jax.nn.sigmoid(decoded_last))

        # ST of proj
        vr = self.vision_range
        grid_size = vr * 2
        st_poses = np.zeros((bs, 3))
        grid_map = np.zeros((bs, grid_size, grid_size, 2))
        st_poses[:, 0] = poses[:, 1] * 200. / self.resolution / grid_size
        st_poses[:, 1] = poses[:, 0] * 200. / self.resolution / grid_size
        st_poses[:, 2] = poses[:, 2] * 57.29577951308232
        rot_mat, trans_mat = get_grid(st_poses, grid_size, grid_size)
        grid_map[:, vr:, int(vr / 2):int(vr / 2 + vr), :] = decoded_last
        
        translated = bilinear_sampler(grid_map, trans_mat[:, 0, :, :], trans_mat[:, 1, :, :])
        rotated = bilinear_sampler(translated, rot_mat[:, 0, :, :], rot_mat[:, 1, :, :])
        rotated = rotated[:, vr:, int(vr / 2):int(vr / 2 + vr), :]

        pred_last_st = rotated

        return pred_last_st
    
    def _get_new_pose(self, pose, rel_pose_change):
        pose[:, 1] += (rel_pose_change[:, 0] * np.sin(pose[:, 2] / 57.29577951308232)
                        + rel_pose_change[:, 1] * np.cos(pose[:, 2] / 57.29577951308232))
        pose[:, 0] += (rel_pose_change[:, 0] * np.cos(pose[:, 2] / 57.29577951308232) 
                       - rel_pose_change[:, 1] * np.sin(pose[:, 2] / 57.29577951308232))
        pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

        pose[:, 2] = np.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
        pose[:, 2] = np.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

        return pose
    
    def _get_map_prediction(self, current_poses, agent_view, maps, explored):
        st_pose = np.array(current_poses)
        st_pose[:, :2] = -(st_pose[:, :2] * 100. / self.resolution - 
                           self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - st_pose[:, 2]
        rot_mat, trans_mat = get_grid(st_pose, self.map_size_cm // self.resolution, self.map_size_cm // self.resolution)

        rotated = bilinear_sampler(agent_view, rot_mat[:, 0, :, :], rot_mat[:, 1, :, :])
        translated = bilinear_sampler(rotated, trans_mat[:, 0, :, :], trans_mat[:, 1, :, :])
        maps2 = np.concatenate([np.expand_dims(maps, -1), translated[:, :, :, :1]], axis=-1)
        explored2 = np.concatenate([np.expand_dims(explored, -1), translated[:, :, :, 1:]], axis=-1)

        map_pred = np.array(self._channel_pool(maps2))
        map_pred = np.squeeze(map_pred, -1)
        exp_pred = np.array(self._channel_pool(explored2))
        exp_pred = np.squeeze(exp_pred, -1)
        
        return map_pred, exp_pred
    
    @partial(jax.jit, static_argnums=(0,), backend='cpu')
    def _channel_pool(self, x):
        n, w, h, c = x.shape
        x_ = jnp.reshape(x, (n, w * h, c)).copy()
        def max_pool_along_channels(x):
            return jnp.max(x, axis=-1, keepdims=True)
        pooled_ = jax.vmap(max_pool_along_channels)(x_)
        _, _, c = pooled_.shape
        o = jnp.reshape(pooled_, (n, w, h, c))
        return o
    
    @partial(jax.jit, static_argnums=(0,))
    def _dec_apply(self, dec_params, s):
        ds = self.dec_func.apply(dec_params, s)
        return ds

    @partial(jax.jit, static_argnums=(0,))
    def _pose_apply(self, pose_params, pose_input):
        pose_pred = self.pose_func.apply(pose_params, pose_input)
        return pose_pred
    
    # @partial(jax.jit, static_argnums=(0,))
    # def _root_inference(self, params, rng_key, obs, pose_input):
    #     r"""Given the observation, a (prior_logits, value, embedding) RootFnOutput is estimated. The
    #     prior_logits are from a policy network. The shapes are ([B, num_actions], [B], [B, ...]), respectively."""
    #     ec = self._enc_apply(params['encoder'], obs)
    #     ds = self._dec_apply(params['decoder'], ec)
    #     pose_pred = self._pose_apply(params['pose'], pose_input)
    #     pose_plane = jax.vmap(partial(pose2plane, shape=pose_pred.shape[1:]+pose_pred.shape[-1]))(pose_pred)
    #     s = jnp.concatenate([ds, pose_plane], axis=-1)
    #     s = self._repr_apply(params['representation'], obs)
    #     v, logits = self._pred_apply(params['prediction'], s)  
    #     v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    #     root = mctx.RootFnOutput(
    #         prior_logits=logits,
    #         value=v,
    #         embedding=s
    #     )
    #     return root 

    # def act(self, rng_key, obs, pose_input,
    #       with_pi: bool = False,
    #       with_value: bool = False,
    #       obs_from_batch: bool = False,
    #       num_simulations: int = 5,
    #       temperature: float = 1.,
    #       invalid_actions=None,
    #       max_depth: int = None, 
    #       loop_fn = jax.lax.fori_loop,
    #       qtransform=None, 
    #       dirichlet_fraction: float = 0.25, 
    #       dirichlet_alpha: float = 0.3, 
    #       pb_c_init: float = 1.25, 
    #       pb_c_base: float = 19652, 
    #       max_num_considered_actions: int = 16, 
    #       gumbel_scale: float = 1):
    #    
        
    #     
    #     if not obs_from_batch:
    #         obs = jnp.expand_dims(obs, axis=0)
    #     plan_output, root_value = self._plan(self.params, rng_key, obs, num_simulations, temperature,
    #                                         invalid_actions=invalid_actions,
    #                                         max_depth=max_depth, 
    #                                         loop_fn=loop_fn,
    #                                         qtransform=qtransform, 
    #                                         dirichlet_fraction=dirichlet_fraction, 
    #                                         dirichlet_alpha=dirichlet_alpha, 
    #                                         pb_c_init=pb_c_init, 
    #                                         pb_c_base=pb_c_base,
    #                                         max_num_considered_actions=max_num_considered_actions,
    #                                         gumbel_scale=gumbel_scale)
    #     root_value = root_value.item() if not obs_from_batch else root_value
    #     action = plan_output.action.item() if not obs_from_batch else np.asarray(plan_output.action)

    #     if with_pi and with_value: return action, plan_output.action_weights, root_value
    #     elif not with_pi and with_value: return action, root_value
    #     elif with_pi and not with_value: return action, plan_output.action_weights
    #     else: return action

    # def _plan(self, params, rng_key, obs, pose_input,
    #        num_simulations: int = 5,
    #        temperature: float = 1.,
    #       invalid_actions=None,
    #       max_depth: int = None, 
    #       loop_fn = jax.lax.fori_loop,
    #       qtransform=None, 
    #       dirichlet_fraction: float = 0.25, 
    #       dirichlet_alpha: float = 0.3, 
    #       pb_c_init: float = 1.25, 
    #       pb_c_base: float = 19652,
    #       max_num_considered_actions: int = 16,
    #       gumbel_scale: float = 1):
    #     root = self._root_inference(params, rng_key, obs)
    #     if self._policy_type == 'muzero':
    #         if qtransform is None:
    #             qtransform = qtransform_by_parent_and_siblings
    #         plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
    #                                 num_simulations=num_simulations,
    #                                 temperature=temperature,
    #                                 invalid_actions=invalid_actions,
    #                                 max_depth=max_depth, 
    #                                 loop_fn=loop_fn,
    #                                 qtransform=qtransform, 
    #                                 dirichlet_fraction=dirichlet_fraction, 
    #                                 dirichlet_alpha=dirichlet_alpha, 
    #                                 pb_c_init=pb_c_init, 
    #                                 pb_c_base=pb_c_base)
    #     elif self._policy_type == 'gumbel':
    #         if qtransform is None:
    #             qtransform = qtransform_completed_by_mix_value
    #         plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
    #                                 num_simulations=num_simulations,
    #                                 invalid_actions=invalid_actions, 
    #                                 max_depth=max_depth, 
    #                                 loop_fn=loop_fn, 
    #                                 qtransform=qtransform, 
    #                                 max_num_considered_actions=max_num_considered_actions, 
    #                                 gumbel_scale=gumbel_scale)
    #     return plan_output, root.value

    