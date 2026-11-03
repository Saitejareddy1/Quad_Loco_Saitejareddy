"""CENet: Context-Aided Estimator Network for Quad Loco.

This is a PyTorch implementation scaffold based on the CENet design
described in the DreamWaQ paper. It has not yet been tested or connected
to Isaac Lab or RSL-RL.

CENet takes a history of proprioceptive observations and uses one shared
encoder to produce:
1. An estimate of the robot's body velocity.
2. A latent context distribution used to reconstruct the next observation.

The shared encoder is important: velocity estimation and next-observation
reconstruction train the same representation together.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CENet(nn.Module):
    """Shared-encoder velocity estimator and beta-VAE observation predictor."""

    def __init__(
        self,
        observation_dim: int,
        history_length: int,
        velocity_dim: int,
        context_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        """Create the CENet modules.

        Args:
            observation_dim: Number of values in one observation.
            history_length: Number of observations in the input history.
            velocity_dim: Number of body-velocity values to estimate.
            context_dim: Size of the learned latent context vector.
            hidden_dims: Encoder layer sizes. These are configurable and
                must be checked against the final paper architecture.
        """
        super().__init__()

        # Reject invalid dimensions so configuration mistakes are clearer.
        dimensions = {
            "observation_dim": observation_dim,
            "history_length": history_length,
            "velocity_dim": velocity_dim,
            "context_dim": context_dim,
        }
        for name, size in dimensions.items():
            if size <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if not hidden_dims or any(size <= 0 for size in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes.")

        self.observation_dim = observation_dim
        self.history_length = history_length
        self.velocity_dim = velocity_dim
        self.context_dim = context_dim

        # Flatten the observation history before passing it through the
        # shared encoder. Input shape is (batch, history, observation_dim).
        encoder_input_dim = history_length * observation_dim
        encoder_layers: list[nn.Module] = []
        previous_dim = encoder_input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ELU(),
                ]
            )
            previous_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        encoder_output_dim = previous_dim

        # Head 1: estimate body velocity from the shared encoded features.
        self.velocity_head = nn.Linear(
            encoder_output_dim,
            velocity_dim,
        )

        # VAE encoder heads: predict the mean and log-variance of the
        # latent context distribution q(z | observation history).
        self.context_mean_head = nn.Linear(
            encoder_output_dim,
            context_dim,
        )
        self.context_log_variance_head = nn.Linear(
            encoder_output_dim,
            context_dim,
        )

        # VAE decoder: use the sampled context to predict the next
        # observation. This is the forward-dynamics reconstruction task.
        decoder_layers: list[nn.Module] = []
        decoder_dims = tuple(reversed(hidden_dims))
        previous_dim = context_dim

        for hidden_dim in decoder_dims:
            decoder_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ELU(),
                ]
            )
            previous_dim = hidden_dim

        decoder_layers.append(nn.Linear(previous_dim, observation_dim))
        self.next_observation_decoder = nn.Sequential(*decoder_layers)

    def encode(
        self,
        observation_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation history and return velocity/context statistics.

        Args:
            observation_history: Tensor shaped
                (batch, history_length, observation_dim).

        Returns:
            velocity_estimate: Estimated body velocity.
            context_mean: Mean of the latent context distribution.
            context_log_variance: Log variance of that distribution.
        """
        expected_shape = (self.history_length, self.observation_dim)

        if observation_history.ndim != 3:
            raise ValueError(
                "observation_history must have shape "
                "(batch, history_length, observation_dim)."
            )

        if tuple(observation_history.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected trailing shape {expected_shape}, "
                f"got {tuple(observation_history.shape[1:])}."
            )

        # Both the velocity head and VAE heads use this shared representation.
        flattened_history = observation_history.flatten(start_dim=1)
        features = self.encoder(flattened_history)

        velocity_estimate = self.velocity_head(features)
        context_mean = self.context_mean_head(features)
        context_log_variance = self.context_log_variance_head(features)

        return (
            velocity_estimate,
            context_mean,
            context_log_variance,
        )

    @staticmethod
    def reparameterize(
        context_mean: torch.Tensor,
        context_log_variance: torch.Tensor,
        sample: bool,
    ) -> torch.Tensor:
        """Sample latent context while allowing gradients to pass through.

        During training, sample=True uses the VAE reparameterization trick.
        During evaluation, sample=False uses the mean for deterministic output.
        """
        if not sample:
            return context_mean

        # z = mean + standard_deviation * random_noise
        standard_deviation = torch.exp(0.5 * context_log_variance)
        noise = torch.randn_like(standard_deviation)
        return context_mean + standard_deviation * noise

    def forward(
        self,
        observation_history: torch.Tensor,
        sample_latent: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Estimate velocity, create context, and predict next observation.

        Args:
            observation_history: Tensor shaped
                (batch, history_length, observation_dim).
            sample_latent: Whether to sample the latent context. Use sampling
                during training and usually disable it for deterministic tests.

        Returns:
            A dictionary containing velocity estimate, context, latent
            distribution parameters, and reconstructed next observation.
        """
        (
            velocity_estimate,
            context_mean,
            context_log_variance,
        ) = self.encode(observation_history)

        context = self.reparameterize(
            context_mean=context_mean,
            context_log_variance=context_log_variance,
            sample=sample_latent,
        )

        reconstructed_next_observation = self.next_observation_decoder(context)

        return {
            "velocity_estimate": velocity_estimate,
            "context": context,
            "context_mean": context_mean,
            "context_log_variance": context_log_variance,
            "reconstructed_next_observation": reconstructed_next_observation,
        }


def cenet_loss(
    outputs: dict[str, torch.Tensor],
    target_body_velocity: torch.Tensor,
    target_next_observation: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Calculate CENet's combined velocity-estimation and beta-VAE losses.

    The paper combines:
    - MSE between estimated and simulator-provided body velocity.
    - Next-observation reconstruction MSE.
    - KL divergence between the learned latent distribution and a
      standard-normal prior.

    The paper's beta value should be verified from the full configuration
    before using this function for a faithful reproduction.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    # Explicit body-state estimation loss.
    velocity_loss = F.mse_loss(
        outputs["velocity_estimate"],
        target_body_velocity,
    )

    # VAE reconstruction loss for the next observation.
    reconstruction_loss = F.mse_loss(
        outputs["reconstructed_next_observation"],
        target_next_observation,
    )

    # KL divergence D_KL(q(z | history) || N(0, I)).
    # The reduction averages over the batch and sums over latent dimensions.
    context_mean = outputs["context_mean"]
    context_log_variance = outputs["context_log_variance"]
    kl_loss = -0.5 * torch.sum(
        1.0
        + context_log_variance
        - context_mean.pow(2)
        - context_log_variance.exp(),
        dim=-1,
    ).mean()

    # The CENet hybrid objective: velocity estimation plus beta-VAE loss.
    vae_loss = reconstruction_loss + beta * kl_loss
    total_loss = velocity_loss + vae_loss

    return {
        "total_loss": total_loss,
        "velocity_loss": velocity_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "vae_loss": vae_loss,
    }