import numpy as np
import gym
import torch

class ReplayBuffer():
    def __init__(self, env, size, batch_size=64):
        self.obs_space = env.observation_space.shape
        self.max_size, self.batch_size = size, batch_size
        self._idx, self.size = 0, 0

        self.states = np.empty((size, *self.obs_space), dtype=np.float64)
        self.actions  = np.empty((size), dtype=np.int32)
        self.rewards = np.empty((size), dtype=np.float64)
        self.next_states = np.empty((size, *self.obs_space), dtype=np.float64)
        self.done = np.empty((size))

    def store(self, sample):
        state, action, reward, next_state, done = sample

        self.states[self._idx] = state
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.next_states[self._idx] = next_state
        self.done[self._idx] = done
        self._idx += 1
        self._idx = self._idx % self.max_size
        self.size += 1
        self.size = min(self.size, self.max_size)

    def full(self):
        return self.max_size == self.size

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size,
                                replace=False)
        state = self.states[idxs,...]
        action = self.actions[idxs,...]
        reward = self.rewards[idxs,...]
        next_state = self.next_states[idxs,...]
        done = self.done[idxs,...]

        return torch.Tensor(state), torch.Tensor(action), torch.Tensor(reward), torch.Tensor(next_state), torch.Tensor(done)

    def __len__(self):
        return self.size

if __name__ == '__main__':
        from gym.wrappers import AtariPreprocessing
        env = gym.make("ALE/Breakout-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4, new_step_api=True)
        buffer = ReplayBuffer(env, 100)
        print(len(buffer))

        state = env.reset()
        done = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            buffer.store((state, action, reward, next_state, done))
            state = next_state
        s, a, r, s_prime, d = buffer.sample()
        print(s.shape) # first dim is batch_size
