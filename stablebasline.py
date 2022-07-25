
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariPreprocessing(env, frame_skip=4)
env = FrameStack(env, 4)
env = RecordVideo(env, './video', episode_trigger=lambda x: x % 1000 == 0)
model = DQN("CnnPolicy", env, verbose=1, buffer_size=250_000)
model.set_logger(new_logger)
model.learn(total_timesteps=50_000, log_interval=4)
model.save("dqn_breakout")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
#     env.render()
    if done:
        break
