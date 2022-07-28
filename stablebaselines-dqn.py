from typing import Any, Dict

import gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from custom_framestack import CustomFrameStack

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariWrapper(env, frame_skip=4)
env = CustomFrameStack(env, 4)
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
            exploration_final_eps=0.01,
            tensorboard_log="./tb-logs")

video_recorder = VideoRecorderCallback(env, render_freq=100_000)
model.learn(total_timesteps=3e6, log_interval=4,
            tb_log_name="run_videos",
            callback=video_recorder)
model.save("dqn_breakout_4fr_baseline")

# run <tensorboard --logdir ./tb-logs> in the terminal
# need to install <pip install moviepy>
# to upload logs and videos online: <tensorboard dev upload --logdir ./tb-logs>
