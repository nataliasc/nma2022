import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from model import DQN
from replaybuffer import ReplayBuffer
from utils_saliency import set_device, set_seed
import random
import torch
import torch.optim as optim
import numpy as np


class Agent():
    def __init__(self,
                 env,
                 gamma=0.99,
                 tau=1e-3,
                 epsilon=0.2,
                 buffer_size=1000,
                 learning_rate=1e-6,
                 batch_size=64):
        self.env = env
        self.Q_target = DQN(env, learning_rate)
        self.Q = DQN(env, learning_rate)
        self.Q_target.load_state_dict(
            self.Q.state_dict())  # set the weights of the target network to those of the policy network
        self.action_space = env.action_space.n
        self.buffer = ReplayBuffer(env, buffer_size, batch_size=batch_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.optimizer = optim.Adam(self.Q.parameters(), self.learning_rate, eps=1.5e-4)
        self.loss = torch.nn.SmoothL1Loss()

    def step(self, action):
        pass

    def train(self, num_episodes):
        state = self.env.reset()

        for episode in range(num_episodes):

            done = False
            self.epsilon = 0.2 # here we can decay epsilon
            while not done:

                # take an action
                q_values = self.Q(state)

                if random.random() < epsilon:  # epsilon-random policy
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_values)

                next_state, reward, done, truncated, info = self.env.step(action)
                # add the tuple to the ReplayBuffer
                sample = (state, action, reward, next_state, done)
                self.buffer.store(sample)
                state = next_state

                if self.buffer.full:
                    continue

                state, action, reward, next_state, done = self.buffer.sample()
                Q_target = self.Q_target(next_state)
                Q_max = torch.max(Q_target)
                y = reward + (1 - done) * self.gamma * Q_max
                x = self.Q(state)[range(BATCH_SIZE), action.squeeze()]
                loss = self.loss(x, y.squeeze())

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
                # polyak averaging

                for target_param, param in zip(Q_target.parameters(), Q.parameters()):
                    target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

                # need to add a way of keeping track of rewards

if __name__ == '__main__':
        import gym
        from gym.wrappers import AtariPreprocessing
        env = gym.make("ALE/Breakout-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4, new_step_api=True)
        agent = Agent(env)
        agent.train(5)
