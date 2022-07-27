import gym
from gym import spaces
from gym.utils.step_api_compatibility import step_api_compatibility

import numpy as np

from collections import deque

class CustomFrameStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations.
    Note:
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
          - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,
    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4 * 96, 96, 3)
        >>> obs = env.reset()
        >>> obs.shape
        (4 * 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        new_step_api: bool = False,
        width: int = 84,
        height: int = 84
    ):
        """Observation wrapper that stacks the observations in a rolling manner.
        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            new_step_api (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env, new_step_api)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)
        self.width = width
        self.height = height

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(int(self.height * num_stack), self.width, 1), dtype=env.observation_space.dtype
        # )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.num_stack, self.height, self.width), dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.
        Args:
            observation: Ignored
        Returns:
            a stack of frames
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.array(self.frames)[:, :, :, -1]

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.
        Args:
            action: The action to step through the environment with
        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = step_api_compatibility(
            self.env.step(action), True
        )
        self.frames.append(observation)
        return step_api_compatibility(
            (self.observation(None), reward, terminated, truncated, info),
            self.new_step_api,
        )

    def reset(self, **kwargs):
        """Reset the environment with kwargs.
        Args:
            **kwargs: The kwargs for the environment reset
        Returns:
            The stacked observations
        """
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        [self.frames.append(obs) for _ in range(self.num_stack)]

        if kwargs.get("return_info", False):
            return self.observation(None), info
        else:
            return self.observation(None)
