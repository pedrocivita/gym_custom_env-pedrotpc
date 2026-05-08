import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for v3.7 — minimalist (neighbors 5x5 + agent vec).

    Inputs (Dict obs):
      - "agent": (10,) — pose (2), coverage (1), 4 directional unvisited
                  ratios (4), nearest-unvisited compass (dx, dy, dist) (3).
      - "neighbors": (5, 5) — local sensor view (free=0 / wall=1 / visited=2).

    Both inputs are size-invariant (same shape across grid sizes), so
    weights transfer between 5x5 / 10x10 / 20x20 cleanly. Param count
    is identical across sizes, enabling curriculum learning without
    buffer mismatches.

    The `agent_dim` is read dynamically from `observation_space["agent"]`,
    so this extractor works for any agent vector size — v3.6 (7) or
    v3.7 (10). The new (dx, dy, dist) compass to the nearest unvisited
    cell that the agent has not seen as an obstacle gives a strong
    endgame signal: the directional ratios degrade to ~1/N when only one
    cell remains, which is too weak to commit to a direction. The
    compass instead points at the actual remaining cell with a unit-scale
    vector, fixing the late-game stall pattern observed in v3.6.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        nbr_shape = observation_space["neighbors"].shape  # (5, 5)
        agent_dim = observation_space["agent"].shape[0]

        self.neighbor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        nbr_out_dim = 32 * nbr_shape[0] * nbr_shape[1]  # 800

        self.agent_mlp = nn.Sequential(
            nn.Linear(agent_dim, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(nbr_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        nbr = observations["neighbors"].unsqueeze(1).float()
        agent = observations["agent"].float()

        nbr_features = self.neighbor_cnn(nbr)
        agent_features = self.agent_mlp(agent)

        combined = torch.cat([nbr_features, agent_features], dim=1)
        return self.combine(combined)
