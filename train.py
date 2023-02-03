import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from agent import PPO
from envwrapper import EnvWrapper
from model import ActorCritic
from util import fix_random_seeds, play


def calculate_discounted_returns(b_rewards, discount_factor):
    discounted_returns = np.empty_like(b_rewards)
    discounted_returns[-1] = b_rewards[-1]
    for i in reversed(range(b_rewards.shape[0] - 1)):
        discounted_returns[i] = (
            discounted_returns[i + 1] * discount_factor + b_rewards[i]
        )

    return discounted_returns


def train(env, model, agent, device, n_episodes=3000, discount_factor=0.99):
    if not os.path.exists("./model"):
        os.makedirs("./model")
    if not os.path.exists("./plot"):
        os.makedirs("./plot")

    writer = SummaryWriter()

    max_steps = 1000 // env.n_frames

    for episode in range(n_episodes):
        # Collecting data
        b_states = np.zeros((max_steps, *env.observation_space.shape), dtype=np.float32)
        b_actions = np.zeros((max_steps, env.action_space.shape[0]), dtype=np.float32)
        b_action_probs = np.zeros((max_steps,), dtype=np.float32)
        b_state_values = np.zeros((max_steps,), dtype=np.float32)
        b_rewards = np.zeros((max_steps,), dtype=np.float32)

        step_cnt = max_steps

        state, _ = env.reset()
        for step in range(max_steps):
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32, device=device
            )
            action, action_prob, state_value = model.forward(state_tensor)

            action = action.detach().cpu().numpy()[0]
            action_prob = action_prob.detach().cpu().numpy()
            state_value = state_value.detach().cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)

            b_states[step] = state
            b_actions[step] = action
            b_action_probs[step] = action_prob
            b_state_values[step] = state_value
            b_rewards[step] = reward

            state = next_state

            if terminated or truncated:
                step_cnt = step + 1
                break

        # Convert training datas to tensor
        b_states = torch.from_numpy(b_states[:step_cnt]).to(device)
        b_actions = torch.from_numpy(b_actions[:step_cnt]).to(device)
        b_action_probs = torch.from_numpy(b_action_probs[:step_cnt]).to(device)
        b_state_values = torch.from_numpy(b_state_values[:step_cnt]).to(device)

        # Calculate and normalize discounted returns
        b_returns = calculate_discounted_returns(b_rewards[:step_cnt], discount_factor)
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-6)
        b_returns = torch.from_numpy(b_returns).to(device)

        b_advantages = b_returns - b_state_values

        # Update parameters
        loss, actor_loss, critic_loss, entropy = agent.learn(
            b_states, b_actions, b_action_probs, b_returns, b_advantages
        )

        total_reward = b_rewards[:step_cnt].sum()
        print(
            f"[Episode {episode + 1:4d}/{n_episodes}] Loss = {loss:.5f}, ",
            f"Actor Loss = {actor_loss:.5f}, Critic Loss = {critic_loss:.5f} ",
            f"Entropy = {entropy:.5f}",
            f"Total Reward = {total_reward:.5f}",
        )

        # Saving plots
        writer.add_scalar("Loss/Episode", loss, episode + 1)
        writer.add_scalar("Actor Loss/Episode", actor_loss, episode + 1)
        writer.add_scalar("Critic Loss/Episode", critic_loss, episode + 1)
        writer.add_scalar("Entropy/Episode", entropy, episode + 1)
        writer.add_scalar("Total Reward/Episode", total_reward, episode + 1)
        writer.flush()

        if (episode + 1) % 100 == 0:
            print("Saving checkpoint... ", end="", flush=True)
            torch.save(
                {
                    "it": episode + 1,
                    "model": model.state_dict(),
                },
                f"./model/checkpoint_{episode + 1:04d}.pt",
            )
            play(model, f"train_{episode + 1:04d}.gif")
            print("Done!")

    writer.close()


def main():
    # Fix random seeds
    seed = 315
    fix_random_seeds(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = EnvWrapper(gym.make("CarRacing-v2", domain_randomize=False), seed=seed)
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0]).to(
        device
    )
    agent = PPO(model)

    train(env, model, agent, device)
    env.close()


if __name__ == "__main__":
    main()
