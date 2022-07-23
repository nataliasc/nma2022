import gym
from collections import deque
from scipy import ndimage as ndi
import cv2
import numpy as np

class SaliencyMap(gym.ObservationWrapper):
    def __init__(self, env, new_step_api=False)
        super().__init__()

        self.frames = deque(maxlen=5)
        self.num_stack = 5
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        # mock saliency map
        self.s_map = np.zeros((84, 84))
        cv2.circle(self.s_map, (30, 42), radius=4, color=1, thickness=1)
        self.s_map = ndi.gaussian_filter(s_map, sigma=6)


    def observation(self, obs):
        # s_map = generate_saliency_map(obs)
        # fake "saliency map" for testing
        s_map = self.saliency_map
        self.frames.append(s_map)
        assert len(self.frames) == 5, (len(self.frames), self.num_stack)
        return observation

    def reset(self, **kwargs):
        """Reset the environment with kwargs.
        Args:
            **kwargs: The kwargs for the environment reset
        Returns:
            Observations combined with a saliency map
        """
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        self.frames.append(saliency_map)

        if kwargs.get("return_info", False):
            return self.observation(None), info
        else:
            return self.observation(None)
