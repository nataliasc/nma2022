import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):
    def __init__(self, env, learning_rate=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = env.observation_space.shape[:1]
        self.output_shape = env.action_space.n


        self.conv1 = nn.Conv2d(in_channels=210*160, # for grayscale images?
                               out_channels=32, kernel_size=8,
                               padding=2,
                               stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4,
                               padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, 3,
                               padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(64, 512)
        self.lin2 = nn.Linear(512, self.output_shape)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.lin1(x), 0.01)
        return self.lin2(x)

if __name__ == '__main__':
        import gym
        env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
        env.reset()
        action = random.randrange(env.action_space.n)
        obs, reward, done, info = env.step(action)
        model = DQN(env)
        print(model(action))
