"""
this is the main script for training and testing DQN
"""
import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from model import DQN
from replaybuffer import ReplayBuffer
from utils_saliency import set_device, set_seed
import random
import torch
import numpy as np


# set device and random seed
DEVICE = set_device()
print(DEVICE + 'is available')

SEED = 2022
set_seed(seed=SEED)

#############################
# hyperparameters
#############################
NUM_TEST = 10  # number of tests on agent
MEM_SIZE = int(1e6)
EPISODES = int(1) # total training episodes
BATCH_SIZE = 64



#############################
# initialise environment
#############################
env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, new_step_api=True)
env = FrameStack(env, 4, new_step_api=True)
state = env.reset()
action = random.randrange(env.action_space.n)
next_state, reward, done, truncated, info = env.step(action)
model = DQN(env).to(DEVICE)
q = model(torch.Tensor(state).unsqueeze(0)) # input shape is now (1, 84, 84)

print('q values output by model')
print(q)

# %%
#############################
# initialise memory buffer
#############################

buffer = ReplayBuffer(env, 1000) # size of 1000 for testing

#############################
# testing function
#############################
def test(save=False):
    """
    tests how well the agent plays a game

    :param save: whether testing episodes are saved and rendered into gif files
    :return: averaged total reward across testing rounds
    """
    print("[TESTING]")
    total_reward = 0
    unclipped_reward = 0

    for i in range(NUM_TEST):
        if i == 0 and save:
            frames = []

        env.reset(eval=True)  # performs random actions to start
        state, _, done, _ = env.step(env.action_space.sample())
        frame = 0

        while not done:  # done controls whether the agent has died in a game
            if i == 0 and save:
                frames.append(state[0, 0])

            # env.render()
            q_values = q_func(state.to(DEVICE))  # TODO q_func is the agent forward pass
            # epsilon-greedy policy for action selection
            if np.random.random() > 0.01:  # small epsilon-greedy, sometimes 0.05
                action = torch.argmax(q_values, dim=1).item()
            else:
                action = env.action_space.sample()

            lives = env.ale.lives()
            next_state, reward, done, info = env.step(action)  # TODO needs to be modified based on how our env is constructed
            if env.ale.lives() != lives:  # lost life
                pass
                # plt.imshow(next_state[0,0])
                # plt.savefig(f"frame-{frame}.png")
                # print("LOST LIFE")

            # unclipped_reward += info['unclipped_reward'] we do not seem to have access to this
            total_reward += reward
            state = next_state
            frame += 1
            # print(f"[TESTING {frame}] Action: {action}, Q-Values: {np.array(q_values.cpu().detach())}, Reward: {reward}, Total Reward: {total_reward}, Terminal: {done}")
            # plt.imshow(state[0,0])
            # plt.savefig("frame-{}.png".format(frame))

        if i == 0 and save:
            frames.append(state[0, 0])
            save_gif(frames, "{}.gif".format(os.path.join(video_dir, str(scheduler.step_count()))))  # TODO safe_fig created in utils to store frames during

    total_reward /= NUM_TEST
    # unclipped_reward /= NUM_TEST # see note above
    # TODO maybe add plotting function or log to wandb
    print(f"[TESTING] Total Reward: {total_reward}")

    return total_reward

#############################
# training loop: MOVED TO agent.py
#############################
avg_q = 0
epsilon = 0.2
q_values = torch.Tensor(np.empty(env.action_space.n))

#every iteration:
# put a sample in the ReplayBuffer
# once the buffer has a minimum number of experiences:
#           start sampling from the buffer
#   every x epochs:
#       set the weights of the target model to the weights of the policy model

for episode in range(EPISODES):

    #Fill ReplayBuffer with enough samples
    while len(buffer) < BATCH_SIZE:
        #reset the environment
        state = env.reset()
        done = 0
        action = env.action_space.sample()
        #take an action
        next_state, reward, done, truncated, info = env.step(action)
        #add the tuple to the ReplayBuffer
        sample = (state, action, reward, next_state, done)
        buffer.store(sample)

    #reset the environment
    state = env.reset()
    done = 0

    while not done:
        if random.random() < epsilon:  # epsilon-random policy
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_values)

        avg_q = 0.9 * avg_q + 0.1 * q_values.mean().item()
        done = 1
        print(avg_q)
