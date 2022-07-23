"""
this is the main script for training and testing DQN
"""
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, RecordVideo
import matplotlib.pyplot as plt
from utils_saliency import set_device, set_seed
from agent import Agent
import random
import torch
import torch.optim as optim
import numpy as np
import wandb
import time


# set device and random seed
DEVICE = set_device()
print(DEVICE + ' is available')

SEED = 2022
set_seed(seed=SEED)

#############################
# hyperparameters
#############################

wandb.init(project="test-project", entity="nma2022", monitor_gym=True)
config = wandb.config
config.num_episodes = 50_000
config.buffer_size = 10_000
config.learning_rate = 1e-5
config.gamma = 0.99
config.tau = 0.05
config.epsilon = 0.9
config.min_epsilon = 0.01
config.epsilon_decay = 0.99995
config.batch_size = 64

env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariPreprocessing(env, frame_skip=4)
env = FrameStack(env, 4)
env = RecordVideo(env, './video', episode_trigger=lambda x: x%100==0) # change how often a video is saved!

agi = Agent(env,
            gamma=config.gamma,
            tau=config.tau,
            epsilon=config.epsilon,
            min_epsilon=config.min_epsilon,
            epsilon_decay=config.epsilon_decay,
            buffer_size=config.buffer_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            device=DEVICE
            )
agi.train(config.num_episodes)
# W&B: watch the model
wandb.watch(agi.Q)
# NOTE: For __very__ serious training, save the model weights
# torch.save(agi.Q.state_dict(), 'model_weights_Q.pth')
# torch.save(agi.Q_target.state_dict(), 'model_weights_Q_target.pth')
