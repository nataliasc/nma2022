"""
this is the main script for training and testing DQN
"""
import gym
from gym.wrappers import AtariPreprocessing

from model import DQN
from utils_saliency import *

# set device and random seed
DEVICE = set_device()

SEED = 2022
set_seed(seed=SEED)

#############################
# hyperparameters
#############################
NUM_TEST = 10  # number of tests on agent



#############################
# initialise environment
#############################
env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, new_step_api=True)
env.reset()
action = random.randrange(env.action_space.n)
obs, reward, done, terminal, info = env.step(action) # why both done and terminal?
model = DQN(env)
model(torch.Tensor(obs).unsqueeze(0)) # input shape is now (1, 84, 84)


#############################
# initialise memory buffer
#############################




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

            unclipped_reward += info['unclipped_reward']
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
    unclipped_reward /= NUM_TEST
    # TODO maybe add plotting function or log to wandb
    print(f"[TESTING] Total Reward: {total_reward}, Unclipped Reward: {unclipped_reward}")

    return total_reward



#############################
# training loop
#############################