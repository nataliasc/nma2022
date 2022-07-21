import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from model import DQN
from replaybuffer import ReplayBuffer
from utils_saliency import set_device, set_seed
import random
import torch
import torch.optim as optim
import numpy as np

#WEIGHTS AND BIASES. NOTE: need to pip install wandb, then log in
import wandb
wandb.init(project="test-project", entity="nma2022")

#log hyperparameters
wandb.config = {
  "num_episodes": 500,
  "learning_rate": 0.00001,
  "batch_size": 64, 
  "gamma": 0.99,
  "tau": 0.001,
  "epsilon": 0.9,
  "min_epsilon": 0.01,
  "epsilon_decay": 0.99,
  "buffer_size": 100000,
  "watch()": "agent.Q_target"
}

class Agent():
    def __init__(self,
                 env,
                 gamma=0.99,
                 tau=1e-3,
                 epsilon=0.9,
                 min_epsilon=0.01,
                 epsilon_decay=0.99,
                 buffer_size=100000,
                 learning_rate=1e-5,
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
        
        #avg_reward = 0
        losses = []

        #episode = epoch
        #the agent die multiple times within an episode

        for episode in range(num_episodes):
            
            total_actions = 0
            state = self.env.reset()
            episode_reward = 0
            done = False
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            #while the agent doesn't die
            while not done:
                
                # take an action
                q_values = self.Q(torch.Tensor(state).unsqueeze(0))

                if random.random() < self.epsilon:  # epsilon-random policy
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_values)

                #store how many lives the agent has left
                lives = self.env.ale.lives()
                
                next_state, reward, done, info = self.env.step(action)

                total_actions += 1
                
                # add the tuple to the ReplayBuffer
                sample = (state, action, reward, next_state, done or (self.env.ale.lives() != lives))
                self.buffer.store(sample)

                state = next_state
                episode_reward += reward
                
                #don't execute the rest if the buffer is not full
                if not self.buffer.full():
                    continue
                
                #The rest will only be executed if the buffer is full
                #print("Sampling from the buffer")
                
                #sample from the buffer
                states, actions, rewards, next_states, t = self.buffer.sample()
                actions = actions.long()

                #Decrease the buffer size, so new samples can be added
                self.buffer.size -= int(self.batch_size)

                with torch.no_grad():
                    Q_target = self.Q_target(next_states)
                    Q_max = torch.max(Q_target)
                    y = rewards + (1 - t) * self.gamma * Q_max
                
                #x = Q value predicted by the policy network
                x = self.Q(states)[range(self.batch_size), actions.squeeze()]
                loss = self.loss(x, y.squeeze())

                #log the loss to w&b
                wandb.log({"loss": loss, "episode": episode})

                # backprop
                self.optimizer.zero_grad()
                loss.backward()

                # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
                # polyak averaging

                for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                    target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

                self.optimizer.step()
                #print(f"Episode {episode}: loss {loss.item()}")
                losses.append(loss.item())
            
            print(f"Episode {episode}: total actions {total_actions} episode reward {episode_reward}")

            #at the end of the episide, log the total reward
            wandb.log({"episode_reward": episode_reward, "episode": episode, "total_episode_actions": total_actions})

        #plt.plot(losses)
        #plt.show()

if __name__ == '__main__':
        import gym
        from gym.wrappers import AtariPreprocessing, FrameStack
        import matplotlib.pyplot as plt
        env = gym.make("ALE/Breakout-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4)
        env = FrameStack(env, 4)
        agent = Agent(env)
        #W&B: watch the model
        wandb.watch(agent.Q)
        #wandb.watch(agent.Q_target)
        agent.train(50000)
        
        #save the model weights
        torch.save(agent.Q.state_dict(), 'model_weights_Q.pth')
        torch.save(agent.Q_target.state_dict(), 'model_weights_Q_target.pth')