import numpy as np
import torch
import torch.nn as nn


class DiagonalGaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.b_logstd = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = torch.zeros_like(mean) + self.b_logstd

        return torch.distributions.Normal(mean, logstd.exp())


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            conv_out = np.prod(self.conv(torch.zeros(1, *input_shape)).size()).item()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            DiagonalGaussian(64, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, determinstic=False):
        state = self.conv(state)
        state = self.fc(state)

        dist = self.actor_head(state)
        if determinstic:
            action = dist.mean
        else:
            action = dist.sample()
        action_prob = dist.log_prob(action).sum(-1)

        state_value = self.critic_head(state)

        return action, action_prob, state_value[:, 0]

    def evaluate(self, state, action):
        state = self.conv(state)
        state = self.fc(state)

        dist = self.actor_head(state)
        action_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        state_value = self.critic_head(state)

        return action_prob, entropy, state_value[:, 0]
