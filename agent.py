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
                 min_epsilon=0.01,
                 epsilon_decay=0.99,
                 buffer_size=100000,
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
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.optimizer = optim.Adam(self.Q.parameters(), self.learning_rate, eps=1.5e-4)
        self.loss = torch.nn.SmoothL1Loss()

    def step(self, action):
        pass

    def train(self, num_episodes):

        avg_reward = 0
        for episode in range(num_episodes):

            state = self.env.reset()
            total_reward = 0
            done = False
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            while not done:

                # take an action
                q_values = self.Q(torch.Tensor(state).unsqueeze(0))

                if random.random() < self.epsilon:  # epsilon-random policy
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_values)

                next_state, reward, done, info = self.env.step(action)
                # add the tuple to the ReplayBuffer
                sample = (state, action, reward, next_state, done)
                self.buffer.store(sample)
                state = next_state
                total_reward += reward

                if not self.buffer.full():
                    continue

                states, actions, rewards, next_states, t = self.buffer.sample()
                actions = actions.long()
                Q_target = self.Q_target(next_states)
                # Q_target = self.Q_target(torch.empty(64, 4, 84, 84))
                Q_max = torch.max(Q_target)
                y = rewards + (1 - t) * self.gamma * Q_max
                x = self.Q(states)[range(self.batch_size), actions.squeeze()]
                loss = self.loss(x, y.squeeze())

                # backprop
                self.optimizer.zero_grad()
                loss.backward()

                # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
                # polyak averaging

                for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                    target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))
                    
                self.optimizer.step()

            avg_reward = 0.9 * avg_reward + 0.1 * total_reward
            print(avg_reward)

if __name__ == '__main__':
        import gym
        from gym.wrappers import AtariPreprocessing, FrameStack
        env = gym.make("ALE/Breakout-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4)
        env = FrameStack(env, 4)
        agent = Agent(env, buffer_size=100)
        agent.train(5)
