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
        avg_q = 0
        epsilon = 0.2
        q_values = torch.Tensor(np.empty(self.env.action_space.n))

        #every iteration:
        # put a sample in the ReplayBuffer
        # once the buffer has a minimum number of experiences:
        #           start sampling from the buffer
        #   every x epochs:
        #       set the weights of the target model to the weights of the policy model

        for episode in range(num_episodes):

            #Fill ReplayBuffer with enough samples
            while len(self.buffer) < self.batch_size:
                #reset the environment
                state = self.env.reset()
                done = 0
                action = self.env.action_space.sample()
                #take an action
                next_state, reward, done, truncated, info = self.env.step(action)
                #add the tuple to the ReplayBuffer
                sample = (state, action, reward, next_state, done)
                self.buffer.store(sample)

            #reset the environment
            state = self.env.reset()
            done = 0

            while not done:
                if random.random() < epsilon:  # epsilon-random policy
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_values)

                avg_q = 0.9 * avg_q + 0.1 * q_values.mean().item()
                done = 1
                print(avg_q)

class EpsilonScheduler():

    def __init__(self, schedule):
        self.steps = 0
        self.schedule = sorted(schedule, key=lambda x: x[0])

        if len(self.schedule) < 1 or self.schedule[0][0] != 0:
            raise ValueError("schedule must have length > 0 and must begin with an initial setting")


    def step(self, n):
        self.steps += n


    def step_count(self):
        return self.steps


    def epsilon(self):
        for i, (next_step, next_epsilon) in enumerate(self.schedule):
            if next_step > self.steps:
                prior = self.schedule[i - 1]
                progress = (self.steps - prior[0]) / (next_step - prior[0])
                return progress * next_epsilon + (1 - progress) * prior[1]
        return self.schedule[-1][1]