"""Actor and critic networks for the Quad Loco project.

This is a standalone PyTorch implementation scaffold. It is not yet
connected to Isaac Lab or RSL-RL; their expected interfaces must be
checked against the university's installed versions.

DreamWaQ structure represented here:
- Actor input: policy observation, CENet velocity estimate, and context.
- Actor output: parameters of a 12-dimensional action distribution.
- Critic input: privileged training state.
- Critic output: one state-value estimate, V(s).

The paper shows hidden-layer sizes of 512, 256, and 128 for both
the policy and value networks, with 12 policy actions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    """Build a multilayer perceptron with ELU hidden activations.

    Args:
        input_dim: Number of input features.
        hidden_dims: Number of units in each hidden layer.
        output_dim: Number of output features.

    Returns:
        An MLP ending in a linear output layer.
    """
    layers: list[nn.Module] = []
    previous_dim = input_dim

    # Add a linear layer and ELU activation for each hidden layer.
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.ELU())
        previous_dim = hidden_dim

    # The output layer is linear; its meaning depends on the network.
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor chooses actions; critic estimates state value.

    Dimensions are supplied when constructing the model because the
    selected Isaac Lab robot and task determine the observation sizes.
    """

    def __init__(
        self,
        policy_observation_dim: int,
        velocity_dim: int,
        context_dim: int,
        privileged_state_dim: int,
        action_dim: int = 12,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        initial_action_std: float = 1.0,
    ) -> None:
        super().__init__()

        # Validate dimensions early so configuration mistakes are clear.
        dimensions = {
            "policy_observation_dim": policy_observation_dim,
            "velocity_dim": velocity_dim,
            "context_dim": context_dim,
            "privileged_state_dim": privileged_state_dim,
            "action_dim": action_dim,
        }
        for name, size in dimensions.items():
            if size <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if initial_action_std <= 0.0:
            raise ValueError("initial_action_std must be greater than zero.")

        # The actor combines the policy observation with CENet outputs.
        actor_input_dim = (
            policy_observation_dim + velocity_dim + context_dim
        )

        # Actor: maps available policy information to the mean action.
        # The 12 outputs correspond to the quadruped's 12 joint actions.
        self.actor = make_mlp(
            input_dim=actor_input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
        )

        # Critic: uses privileged training state to estimate V(s).
        # It does not directly choose actions.
        self.critic = make_mlp(
            input_dim=privileged_state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
        )

        # PPO commonly uses a Gaussian policy. This learnable parameter
        # controls action exploration; it is not an Isaac Lab actuator gain.
        self.log_action_std = nn.Parameter(
            torch.full((action_dim,), float(initial_action_std)).log()
        )

    def get_action_distribution(
        self,
        policy_observation: torch.Tensor,
        velocity_estimate: torch.Tensor,
        context: torch.Tensor,
    ) -> Normal:
        """Create the actor's Gaussian action distribution.

        CENet supplies velocity_estimate and context. The policy
        observation is the actor's proprioceptive input.
        """
        actor_input = torch.cat(
            (
                policy_observation,
                velocity_estimate,
                context,
            ),
            dim=-1,
        )

        # The network predicts the Gaussian mean; standard deviation is
        # represented by a learnable parameter shared across the batch.
        action_mean = self.actor(actor_input)
        action_std = self.log_action_std.exp().expand_as(action_mean)

        return Normal(action_mean, action_std)

    def act(
        self,
        policy_observation: torch.Tensor,
        velocity_estimate: torch.Tensor,
        context: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select an action from the actor's policy.

        During training, sample from the distribution for exploration.
        During evaluation, deterministic=True returns the mean action.
        """
        distribution = self.get_action_distribution(
            policy_observation=policy_observation,
            velocity_estimate=velocity_estimate,
            context=context,
        )

        if deterministic:
            return distribution.mean

        return distribution.sample()

    def evaluate_value(
        self,
        privileged_state: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate V(s) from privileged information during training."""
        return self.critic(privileged_state).squeeze(-1)

    def evaluate_actions(
        self,
        policy_observation: torch.Tensor,
        velocity_estimate: torch.Tensor,
        context: torch.Tensor,
        privileged_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return PPO quantities for previously selected actions.

        Returns:
            log_probability: Log probability of each action.
            entropy: Policy entropy, used to measure exploration.
            value: Critic's estimate V(s) for each sample.
        """
        distribution = self.get_action_distribution(
            policy_observation=policy_observation,
            velocity_estimate=velocity_estimate,
            context=context,
        )

        # Sum over the 12 action dimensions to get one value per sample.
        log_probability = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.evaluate_value(privileged_state)

        return log_probability, entropy, value