import re 
import time
import logging 
import datetime 
import gymnasium as gym 
import numpy as np 
from typing import Mapping

from muax.wrappers import TrainMonitor

class NHWCWrapper(gym.Wrapper):
    """move axis to turn obs into NHWC format"""
    def __init__(self, env, channel_axis = 1):
        super().__init__(env)
        self.channel_axis = channel_axis
        self.observation_space = self._new_observation_space(env)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.moveaxis(observation, self.channel_axis, -1)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = np.moveaxis(observation, self.channel_axis, -1)
        return observation, info  
    
    def _new_observation_space(self, env):
        sample = env.observation_space.sample()
        new_sample = np.moveaxis(sample, self.channel_axis, -1)
        new_obs_space = type(env.observation_space)(shape=new_sample.shape, 
                                                    low=env.observation_space.low, 
                                                    high=env.observation_space.high, 
                                                    dtype=env.observation_space.dtype)
        return new_obs_space
    

class TensorBoardMonitor(TrainMonitor):
    r"""
    Environment wrapper for tracking statistics.
    This wrapper logs some diagnostics at the end of each episode and it also gives us some handy
    attributes (listed below).
    Parameters
    ----------
    from_batch: bool
        If the env returns batched information, the `from_batch` should be True. Default is False.
    env : gymnasium environment
        A gymnasium environment.
    tensorboard_dir : str, optional
        If provided, TrainMonitor will log all diagnostics to be viewed in tensorboard. To view
        these, point tensorboard to the same dir:
        .. code:: bash
            $ tensorboard --logdir {tensorboard_dir}
    tensorboard_write_all : bool, optional
        You may record your training metrics using the :attr:`record_metrics` method. Setting the
        ``tensorboard_write_all`` specifies whether to pass the metrics on to tensorboard
        immediately (``True``) or to wait and average them across the episode (``False``). The
        default setting (``False``) prevents tensorboard from being fluided by logs.
    log_all_metrics : bool, optional
        Whether to log all metrics. If ``log_all_metrics=False``, only a reduced set of metrics are
        logged.
    smoothing : positive int, optional
        The number of observations for smoothing the metrics. We use the following smooth update
        rule:
        .. math::
            n\ &\leftarrow\ \min(\text{smoothing}, n + 1) \\
            x_\text{avg}\ &\leftarrow\ x_\text{avg}
                + \frac{x_\text{obs} - x_\text{avg}}{n}
    \*\*logger_kwargs
        Keyword arguments to pass on to :func:`coax.utils.enable_logging`.
    Attributes
    ----------
    T : positive int
        Global step counter. This is not reset by ``env.reset()``, use ``env.reset_global()``
        instead.
    ep : positive int
        Global episode counter. This is not reset by ``env.reset()``, use ``env.reset_global()``
        instead.
    t : positive int
        Step counter within an episode.
    G : float
        The return, i.e. amount of reward accumulated from the start of the current episode.
    avg_G : float
        The average return G, averaged over the past 100 episodes.
    dt_ms : float
        The average wall time of a single step, in milliseconds.
    """
    _COUNTER_ATTRS = (
        'T', 'ep', 't', 'G', 'avg_G', '_n_avg_G', '_ep_starttime', '_ep_metrics', '_ep_actions',
        '_tensorboard_dir', '_period')

    def __init__(
            self, from_batch: bool = False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.from_batch = from_batch

    def step(self, a):
        self._ep_actions.append(a)
        s_next, r, done, truncated, info = self.env.step(a)
        if not info:
            info = {}
        info['monitor'] = {'T': self.T, 'ep': self.ep}
        self.t += 1
        self.T += 1
        self.G += r
        if done or truncated:
            if self._n_avg_G < self.smoothing:
                self._n_avg_G += 1.
            self.avg_G += (self.G - self.avg_G) / self._n_avg_G

        return s_next, r, done, truncated, info

    def record_metrics(self, metrics):
        r"""
        Record metrics during the training process.
        These are used to print more diagnostics.
        Parameters
        ----------
        metrics : dict
            A dict of metrics, of type ``{name <str>: value <float>}``.
        """
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a Mapping")

        # write metrics to tensoboard
        if self.tensorboard is not None and self.tensorboard_write_all:
            for name, metric in metrics.items():
                self.tensorboard.add_scalar(
                    str(name), float(metric), global_step=self.T)

        # compute episode averages
        for k, v in metrics.items():
            if k not in self._ep_metrics:
                self._ep_metrics[k] = v, 1.
            else:
                x, n = self._ep_metrics[k]
                self._ep_metrics[k] = x + v, n + 1

    def get_metrics(self):
        r"""
        Return the current state of the metrics.
        Returns
        -------
        metrics : dict
            A dict of metrics, of type ``{name <str>: value <float>}``.
        """
        return {k: float(x) / n for k, (x, n) in self._ep_metrics.items()}

    def period(self, name, T_period=None, ep_period=None):
        if T_period is not None:
            T_period = int(T_period)
            assert T_period > 0
            if name not in self._period['T']:
                self._period['T'][name] = 1
            if self.T >= self._period['T'][name] * T_period:
                self._period['T'][name] += 1
                return True or self.period(name, None, ep_period)
            return self.period(name, None, ep_period)
        if ep_period is not None:
            ep_period = int(ep_period)
            assert ep_period > 0
            if name not in self._period['ep']:
                self._period['ep'][name] = 1
            if self.ep >= self._period['ep'][name] * ep_period:
                self._period['ep'][name] += 1
                return True
        return False

    
