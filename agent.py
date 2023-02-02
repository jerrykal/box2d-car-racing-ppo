import numpy as np
import torch
import torch.nn.functional as F


class PPO:
    def __init__(
        self,
        model,
        lr=1e-4,
        n_epoches=10,
        batch_size=250,
        clip_val=0.2,
        critic_weight=0.5,
        ent_weight=0.01,
    ):
        self.model = model
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.critic_weight = critic_weight
        self.entropy_weight = ent_weight

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def learn(
        self,
        b_states,
        b_actions,
        b_action_probs,
        b_returns,
        b_advantages,
    ):
        step_cnt = b_states.size()[0]
        n_batches = step_cnt // self.batch_size
        if n_batches == 0:
            n_batches = 1
            batch_size = step_cnt
        else:
            batch_size = self.batch_size

        rand_idx = np.arange(step_cnt)

        for _ in range(self.n_epoches):
            np.random.shuffle(rand_idx)

            for i in range(n_batches):
                sample_idx = rand_idx[i * batch_size : (i + 1) * batch_size]

                sample_states = b_states[sample_idx]
                sample_actions = b_actions[sample_idx]
                sample_action_probs = b_action_probs[sample_idx]
                sample_returns = b_returns[sample_idx]
                sample_advantages = b_advantages[sample_idx]

                new_action_probs, entropies, new_state_values = self.model.evaluate(
                    sample_states, sample_actions
                )
                entropy = entropies.mean()

                # Calculating actor's loss
                ratio = (new_action_probs - sample_action_probs).exp()
                actor_loss = -torch.min(
                    sample_advantages * ratio,
                    sample_advantages
                    * ratio.clamp(1.0 - self.clip_val, 1.0 + self.clip_val),
                ).mean()

                # Calculating critic's loss
                critic_loss = F.mse_loss(sample_returns, new_state_values)

                # Update parameters
                self.optimizer.zero_grad()
                loss = (
                    actor_loss
                    + self.critic_weight * critic_loss
                    - self.entropy_weight * entropy
                )
                loss.backward()
                self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
