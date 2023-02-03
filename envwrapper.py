import gymnasium as gym
import numpy as np


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, seed=None, n_tracks=1000, n_frames=4, img_size=(84, 84)):
        super().__init__(env)
        self.n_tracks = n_tracks
        self.n_frames = n_frames
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(n_frames,) + img_size, dtype=np.float32
        )

        # Uses an RNG with optional fixed random seeds for reproducibility
        self.rng = np.random.default_rng(seed=seed)

    def _preprocess(self, state):
        # Crop the image
        state = state[:-12, 6:-6]

        # Convert the image to grayscale
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        return state

    def reset(self, seed=None):
        if seed == None:
            seed = self.rng.integers(self.n_tracks).item()

        state, info = self.env.reset(seed=seed)
        state = self._preprocess(state)
        return np.expand_dims(state, axis=0).repeat(self.n_frames, axis=0), info

    def step(self, action):
        next_state = np.empty(self.observation_space.shape)
        total_reward = 0
        terminated = False
        truncated = False
        info = []
        for i in range(self.n_frames):
            if not terminated and not truncated:
                next_state_i, reward_i, terminated, truncated, info_i = self.env.step(
                    action
                )
                next_state_i = self._preprocess(next_state_i)
                total_reward += reward_i
                info.append(info_i)
            next_state[i] = next_state_i

        return next_state, total_reward, terminated, truncated, info
