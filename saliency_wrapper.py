import gym
import torch
from gym import spaces
from scipy import ndimage as ndi
import cv2
import numpy as np


class SaliencyMap(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, net, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )

        # mock saliency map
        self.s_map = net # net for saliency pred
        cv2.circle(self.s_map, (30, 42), radius=4, color=1, thickness=1)
        self.s_map = ndi.gaussian_filter(self.s_map, sigma=6)


    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation combined with a saliency map
        generated by the saliency prediction NN

        :param frame: environment frame
        :return: the observation
        """
        # call to generate saliency map here
        saliency_map = self.s_map
        frame = frame[:, :, 0]
        product = np.multiply(frame, saliency_map)
        frame = np.mean( np.array([ frame, product]), axis=0 )

        return frame[:, :, None]

class SaliencyMap4F(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, net, device, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.num_stack = 4
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.num_stack, self.height, self.width), dtype=env.observation_space.dtype
        )

        # load net
        self.s_map = net
        self.net_device = device
        # cv2.circle(self.s_map, (30, 42), radius=4, color=1, thickness=1)
        # self.s_map = ndi.gaussian_filter(self.s_map, sigma=9)


    def observation(self, frames: np.ndarray) -> np.ndarray:
        """
        returns the current observation combined with a saliency map
        generated by the saliency prediction NN

        :param frame: environment frame
        :return: the observation
        """
        # call to generate saliency map here
        frames = torch.tensor(frames).to(self.net_device)
        frames = torch.unsqueeze(frames, dim=0).float()  # add batch dim
        saliency_map = torch.squeeze(self.s_map(frames)).numpy()
        frame_stack = []
        for i in range(frames.shape[0]):
            product = np.multiply(frames[i], saliency_map)
            frame = np.mean( np.array([ frames[i], product]), axis=0 )
            frame_stack.append(frame)

        return np.array(frame_stack)[:, :, :]
