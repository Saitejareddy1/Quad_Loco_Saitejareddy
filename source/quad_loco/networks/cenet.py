"""First-pass Context-Aided Estimator Network (CENet).

Estimates body velocity and produces a learned context vector
from a history of proprioceptive observations.
"""

import torch
from torch import nn


class CENet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        history_length: int,
        velocity_dim: int = 3,
        context_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.observation_dim = observation_dim
        self.history_length = history_length
        self.velocity_dim = velocity_dim
        self.context_dim = context_dim

        input_dim = observation_dim * history_length

        # Shared encoder for the observation history.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Estimate body velocity.
        self.velocity_head = nn.Linear(hidden_dim, velocity_dim)

        # Produce a distribution for the latent context.
        self.context_mean = nn.Linear(hidden_dim, context_dim)
        self.context_log_variance = nn.Linear(hidden_dim, context_dim)

    def forward(self, observation_history: torch.Tensor):
        """Return estimated velocity and sampled context.

        Args:
            observation_history: Tensor shaped
                (batch, history_length, observation_dim).

        Returns:
            velocity_estimate: (batch, velocity_dim)
            context: (batch, context_dim)
            context_mean: (batch, context_dim)
            context_log_variance: (batch, context_dim)
        """
        expected_shape = (self.history_length, self.observation_dim)
        if observation_history.shape[1:] != expected_shape:
            raise ValueError(
                "Expected observation history with trailing shape "
                f"{expected_shape}, got {tuple(observation_history.shape[1:])}."
            )

        flattened_history = observation_history.flatten(start_dim=1)
        features = self.encoder(flattened_history)

        velocity_estimate = self.velocity_head(features)
        context_mean = self.context_mean(features)
        context_log_variance = self.context_log_variance(features)

        # Reparameterization trick used for sampling a VAE latent.
        noise = torch.randn_like(context_mean)
        context = context_mean + noise * torch.exp(0.5 * context_log_variance)

        return (
            velocity_estimate,
            context,
            context_mean,
            context_log_variance,
        )