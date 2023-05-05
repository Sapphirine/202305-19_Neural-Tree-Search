from abc import ABC, abstractmethod
from typing import Any, Tuple
import dataclasses
from dataclasses import dataclass
import jax 
import numpy as np

from muax.utils import sliceable_deque

@dataclass
class NAVTransition:
    obs_last: Any = 0.
    obs: Any = 0.
    poses: Any = 0.
    pose_input: Any = 0.
    fp_proj: Any = 0.
    fp_explored: Any = 0.
    pose_err: Any = 0.
    a: int = 0
    r: float = 0.
    done: bool = False
    Rn: float = 0. 
    v: float = 0. 
    pi: Any = 0.
    w: float = 1.

    def __iter__(self):
      for field in dataclasses.fields(self):
        yield getattr(self, field.name)

    def __getitem__(self, index):
      return NAVTransition(*(_attr[index] for _attr in self))

def flatten_transition_func(transition: NAVTransition) -> Tuple:
  return iter(transition), None 

def unflatten_transition_func(treedef, leaves) -> NAVTransition:
  return NAVTransition(*leaves)

jax.tree_util.register_pytree_node(
    NAVTransition,
    flatten_func=flatten_transition_func,
    unflatten_func=unflatten_transition_func
)  


class BaseTracer(ABC):

    @abstractmethod
    def reset(self):
        r"""
        Reset the cache to the initial state.
        """
        pass

    @abstractmethod
    def add(self, obs_last, obs, poses, pose_input, fp_proj, fp_explored, pose_err, a, r, done, v=0.0, pi=0.0, w=1.0):
        r"""
        Add a transition to the experience cache.
        Parameters
        ----------
        obs_last: last observation 
        obs : state observation
            A single state observation.
        poses: Current sensor poses.
        pose_input: Input for pose estimator
        fp_proj: ground truth map
        fp_explored: ground truth explored 
        pose_err: ground truth pose
        a : action
            A single action.
        r : float
            A single observed reward.
        done : bool
            Whether the episode has finished.
        v : search tree root node value.
        pi : float, optional
            The action weights.
        w : float, optional
            Sample weight associated with the given state-action pair.
        """
        pass

    @abstractmethod
    def pop(self):
        r"""
        Pop a single transition from the cache.
        Returns
        -------
        transition : An instance of Transition
            
        """
        pass


class NAVNStep(BaseTracer):
    r"""
    A short-term cache for :math:`n`-step bootstrapping.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    """

    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        r"""
        Reset the cache to the initial state.
        """
        self._deque_s = sliceable_deque([])
        self._deque_r = sliceable_deque([])
        self._done = False
        self._gammas = np.power(self.gamma, np.arange(self.n))
        self._gamman = np.power(self.gamma, self.n)

    def add(self, obs_last, obs, poses, pose_input, fp_proj, fp_explored, pose_err, a, r, done, v=0.0, pi=0.0, w=1.0):
        # if self._done and len(self):
        #     raise EpisodeDoneError(
        #         "please flush cache (or repeatedly call popleft) before appending new transitions")

        self._deque_s.append((obs_last, obs, poses, pose_input, fp_proj, fp_explored, pose_err, a, v, pi, w))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_s)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        r"""
        Pops a single transition from the cache. Computes n-step bootstrapping value.
        Returns
        -------
        transition : An instance of Transition
            
        """
        # if not self:
        #     raise InsufficientCacheError(
        #         "cache needs to receive more transitions before it can be popped from")

        # pop state-action (propensities) pair
        obs_last, obs, poses, pose_input, fp_proj, fp_explored, pose_err, a, v, pi, w = self._deque_s.popleft()

        # n-step partial return
        rs = np.array(self._deque_r[:self.n])
        Rn = np.sum(self._gammas[:len(rs)] * rs).item()
        r = self._deque_r.popleft()

        # keep in mind that we've already popped 
        if len(self) >= self.n:
            _, _, _, _, _, _, _, _, v_next, _, _ = self._deque_s[self.n - 1]
            done = False
            gamman = self._gamman
        else:
            # no more bootstrapping
            v_next = 0
            done = True
            gamman = self._gammas[len(rs) - 1]
        
        Rn += v_next * gamman

        return NAVTransition(obs_last=obs_last, obs=obs, poses=poses, pose_input=pose_input, fp_proj=fp_proj, fp_explored=fp_explored, pose_err=pose_err, a=a, r=r, done=done, Rn=Rn, v=v, pi=pi, w=w)


class NAVPNStep(NAVNStep):
    r"""
    A short-term cache for :math:`n`-step bootstrapping with priority.
    The weight `w` is calcualted as: `w=abs(v - Rn) ** alpha`, 
    where `v` is the value predicted from the model, 
    `Rn` is the n-step bootstrapping value calculated from the rewards.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    alpha: float between 0 and 1
        The PER alpha.
    """
    def __init__(self, n, gamma, alpha: float = 0.5):
      self.alpha = float(alpha)
      super().__init__(n, gamma)
    
    def pop(self):
        # if not self:
        #     raise InsufficientCacheError(
        #         "cache needs to receive more transitions before it can be popped from")

        # pop state-action (propensities) pair
        obs_last, obs, poses, pose_input, fp_proj, fp_explored, pose_err, a, v, pi, w = self._deque_s.popleft()

        # n-step partial return
        rs = np.array(self._deque_r[:self.n])
        Rn = np.sum(self._gammas[:len(rs)] * rs).item()
        r = self._deque_r.popleft()

        # keep in mind that we've already popped 
        if len(self) >= self.n:
            _, _, _, _, _, _, _, _, v_next, _, _ = self._deque_s[self.n - 1]
            done = False
            gamman = self._gamman
        else:
            # no more bootstrapping
            v_next = 0
            done = True
            gamman = self._gammas[len(rs) - 1]
        
        Rn += v_next * gamman

        w = abs(v - Rn) ** self.alpha

        return NAVTransition(obs_last=obs_last, obs=obs, poses=poses, pose_input=pose_input, fp_proj=fp_proj, fp_explored=fp_explored, pose_err=pose_err, a=a, r=r, done=done, Rn=Rn, v=v, pi=pi, w=w)