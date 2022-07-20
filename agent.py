import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from model import DQN
from replaybuffer import ReplayBuffer
from utils_saliency import set_device, set_seed
import random
import torch
import numpy as np

class Agent():
    def __init__(self,
                 env,
                 gamma=0.99,
                 tau=1e-3,
                 epsilon=0.2,
                 buffer_size=1000,
                 learning_rate=1e-3,
                 batch_size=64):
        self.env = env
        self.Q_target = DQN(env, learning_rate)
        self.Q = DQN(env, learning_rate)
        self.Q_target.load_state_dict(self.Q.state_dict()) # set the weights of the target network to those of the policy network
        self.action_space = env.action_space.n
        self.buffer = ReplayBuffer(env, buffer_size, batch_size=batch_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def step(self, action):
        pass

    def train(self, num_episodes):
        pass
