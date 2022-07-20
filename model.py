import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, env, learning_rate=1e-3):
        super().__init__()
        self.output_shape = env.action_space.n

        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32, kernel_size=8,
                               stride=4)

        self.conv2 = nn.Conv2d(32, 64, 4,
                               stride=2)

        self.conv3 = nn.Conv2d(64, 64, 3,
                               stride=1)

        self.fc1 = nn.Linear(in_features= 7 * 7 * 64, out_features=1024)

        self.fc2 = nn.Linear(1024, self.output_shape)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
        import gym
        from gym.wrappers import AtariPreprocessing
        env = gym.make("ALE/Breakout-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4)
        # env.reset()
        # action = env.observation_space.sample()
        # obs, reward, done, info = env.step(action)
        model = DQN(env)
        model(torch.empty((1, 4, 84, 84), dtype=torch.float32))
        model(torch.empty((64, 4, 84, 84), dtype=torch.float32)) # test
