import gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# useful: https://brosa.ca/blog/ale-release-v0.7/#openai-gym
env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariWrapper(env, frame_skip=4)
model = DQN("CnnPolicy",
            env,
            verbose=1,
            buffer_size=100_000,
            learning_rate=1e-4,
            batch_size=32,
            learning_starts=100000,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.01)
model.learn(total_timesteps=1e7, log_interval=4)
model.save("dqn_breakout")
