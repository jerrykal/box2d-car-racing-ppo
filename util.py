import os
import random

import gymnasium as gym
import imageio
import numpy as np
import torch

from envwrapper import EnvWrapper


def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)


def save_gif(frames, fname):
    if not os.path.exists("./gif"):
        os.makedirs("./gif")
    imageio.mimwrite(f"./gif/{fname}", frames)


def play(model, fname):
    env = EnvWrapper(
        gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array")
    )

    frames = []
    state, _ = env.reset()
    while True:
        frames.append(env.render())

        with torch.no_grad():
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32
            )
            action, _, _ = model(state_tensor)
            action = action.cpu().numpy()[0]

        next_state, _, terminated, truncated, _ = env.step(action)

        state = next_state
        if terminated or truncated:
            break

    save_gif(frames, fname)
