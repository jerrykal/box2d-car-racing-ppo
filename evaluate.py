import os

import gymnasium as gym
import numpy as np
import torch

from envwrapper import EnvWrapper
from model import ActorCritic
from util import save_gif


def main():
    if not os.path.exists("./model"):
        print("ERROR: No model saved")
        exit(1)

    # Randomly generate 50 different seeds for env.reset to make sure that all models
    # are going through the same set of tracks
    rng = np.random.default_rng(seed=315)
    track_seeds = rng.choice(2**32 - 1, size=50, replace=False)

    env = EnvWrapper(
        gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array")
    )
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0])

    highest_score = -1000
    best_avg_score = -1000
    best_fname = ""

    # Put all the checkpoint models through 50 test drives to evaluate performance
    for fname in sorted(os.listdir("./model")):
        if not fname.endswith(".pt") or "best" in fname:
            continue

        print(f"Evaluating {fname} ... ", end="", flush=True)

        checkpoint = torch.load(f"./model/{fname}")
        model.load_state_dict(checkpoint["model"])

        avg_score = 0

        for seed in track_seeds:
            frames = []
            score = 0

            state, _ = env.reset(seed=seed.item())
            while True:
                frames.append(env.render())

                state_tensor = torch.tensor(
                    np.expand_dims(state, axis=0), dtype=torch.float32
                )
                with torch.no_grad():
                    action, _, _ = model(state_tensor, determinstic=True)
                    action = action.detach().cpu().numpy()[0]
                next_state, reward, terminated, truncated, _ = env.step(action)

                score += reward
                state = next_state
                if terminated or truncated:
                    break

            # Save the best play for demo
            if score > highest_score:
                highest_score = score
                save_gif(frames, "best_play.gif")

            avg_score += score

        print(f"Average Score = {avg_score / 50:.5f}")
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_fname = fname
            torch.save(checkpoint, "./model/model_best.pt")

    print(f"The best model is {best_fname}, with a average score of {best_avg_score / 50:5f}")


if __name__ == "__main__":
    main()
