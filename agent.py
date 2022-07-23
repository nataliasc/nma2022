import gym
from gym.wrappers import AtariPreprocessing, FrameStack, RecordVideo

from model import DQN
from replaybuffer import ReplayBuffer
from utils_saliency import set_device, set_seed
import random
import torch
import torch.optim as optim
import numpy as np
import wandb
import time

# Add hyperparameters to weights&biases
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

# Find the device that the code will be running on
DEVICE = set_device()


class Agent():
    def __init__(self,
                 env,
                 gamma=0.99,
                 tau=0.05,
                 epsilon=1.,
                 min_epsilon=0.01,
                 epsilon_decay=0.99995,
                 buffer_size=10_000,
                 learning_rate=1e-5,
                 batch_size=64,
                 device="cpu"):

        self.device = device
        self.env = env
        self.Q_target = DQN(env, learning_rate).to(device)
        self.Q = DQN(env, learning_rate).to(device)
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

    def train(self, num_episodes):

        # avg_reward = 0
        losses = []

        # episode = epoch
        # the agent die multiple times within an episode

        for episode in range(num_episodes):

            total_actions = 0
            state = self.env.reset()
            episode_reward = 0
            done = False
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            wandb.log({"epsilon": self.epsilon})

            # while the agent doesn't die
            while not done:

                # take an action
                # convert state to a single np.array (fix warning on conversion)
                state = np.array(state)
                q_values = self.Q(torch.Tensor(state).unsqueeze(0).to(self.device))

                if random.random() < self.epsilon:  # epsilon-random policy
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_values)

                # store how many lives the agent has left
                lives = self.env.ale.lives()

                next_state, reward, done, info = self.env.step(action)
                wandb.log({"frames": info['frame_number']})
                total_actions += 1

                # add the tuple to the ReplayBuffer
                sample = (state, action, reward, next_state, done or (self.env.ale.lives() != lives))
                self.buffer.store(sample)

                state = next_state
                episode_reward += reward

                # don't execute the rest if the buffer is not full
                if not self.buffer.full():
                    continue

                # The rest will only be executed if the buffer is full
                # print("Sampling from the buffer")

                # sample from the buffer
                states, actions, rewards, next_states, t = self.buffer.sample()
                actions = actions.long()

                # Decrease the buffer size, so new samples can be added
                self.buffer.size -= int(self.batch_size)

                with torch.no_grad():
                    Q_target = self.Q_target(next_states.to(self.device))
                    Q_max = torch.max(Q_target)
                    y = rewards.to(self.device) + (1 - t.to(self.device)) * self.gamma * Q_max

                # x = Q value predicted by the policy network
                x = self.Q(states.to(self.device))[range(self.batch_size), actions.squeeze()]
                loss = self.loss(x, y.squeeze())

                # log the loss to w&b
                wandb.log({"loss": loss, "episode": episode})

                # backprop
                self.optimizer.zero_grad()
                loss.backward()

                # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
                # polyak averaging

                for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                    target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

                # for param in self.Q.parameters(): # gradient clipping
                #     param.grad.data.clamp_(-1, 1)

                self.optimizer.step()
                # print(f"Episode {episode}: loss {loss.item()}")
                losses.append(loss.cpu().item())

            print(f"Episode {episode}: total actions {total_actions}; episode reward {episode_reward}")

            # at the end of the episide, log the total reward
            wandb.log({"episode_reward": episode_reward, "episode": episode, "total_episode_actions": total_actions})

            if episode % 100 == 0:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                torch.save(self.Q.state_dict(), f"model_weights_Q_e_{episode}_{timestr}.pth")
                torch.save(self.Q_target.state_dict(), f"model_weights_Q_target_episode{episode}_{timestr}.pth")

    def test(self):

        cumulative_reward = 0
        state = self.env.reset()
        while not done:

            # take an action
            q_values = self.Q(torch.Tensor(state).unsqueeze(0).to(self.device))

            if random.random() < 0.01:  # epsilon-random policy
                action = self.env.action_space.sample()
            else:
                action = torch.argmax(q_values)

            next_state, reward, done, info = self.env.step(action)
            cumulative_reward += reward


if __name__ == '__main__':
    import gym
    from gym.wrappers import AtariPreprocessing, FrameStack
    import matplotlib.pyplot as plt
    DEVICE = set_device()
    SEED = 2022
    set_seed(seed=SEED)

    # useful: https://brosa.ca/blog/ale-release-v0.7/#openai-gym
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4)
    env = FrameStack(env, 4)
    env = RecordVideo(env, './video', episode_trigger=lambda x: x%100==0)
    agent = Agent(env, buffer_size=100)

    # W&B: watch the model
    wandb.watch(agent.Q)
    # wandb.watch(agent.Q_target)
    agent.train(5)

    # NOTE: For __very__ serious training, save the model weights
    # torch.save(agent.Q.state_dict(), 'model_weights_Q.pth')
    # torch.save(agent.Q_target.state_dict(), 'model_weights_Q_target.pth')
