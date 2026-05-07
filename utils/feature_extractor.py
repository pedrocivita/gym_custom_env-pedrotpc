import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for v3.2 — two 5x5 spatial windows + agent vector.

    Inputs (Dict obs):
      - "agent": (7,) — pose, coverage, 4 directional ratios.
      - "neighbors": (5, 5) — local sensor view (free / wall / visited).
      - "visited_neighbors": (5, 5) — binary memory of visited cells inside
        the same 5x5 window.

    Both spatial inputs are *fixed* 5x5 regardless of grid size, so the
    extractor parameters are identical for 5x5 / 10x10 / 20x20. This is
    what enables transfer learning: weights from a 5x5 model load
    directly into a 10x10 or 20x20 model.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        nbr_shape = observation_space["neighbors"].shape  # (5, 5)
        vis_shape = observation_space["visited_neighbors"].shape  # (5, 5)
        agent_dim = observation_space["agent"].shape[0]

        assert nbr_shape == vis_shape, "5x5 windows must match in shape"

        self.neighbor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.visited_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        spatial_out = 32 * nbr_shape[0] * nbr_shape[1]  # 32 * 5 * 5 = 800

        self.agent_mlp = nn.Sequential(
            nn.Linear(agent_dim, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(2 * spatial_out + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        nbr = observations["neighbors"].unsqueeze(1).float()
        vis = observations["visited_neighbors"].unsqueeze(1).float()
        agent = observations["agent"].float()

        nbr_features = self.neighbor_cnn(nbr)
        vis_features = self.visited_cnn(vis)
        agent_features = self.agent_mlp(agent)

        combined = torch.cat([nbr_features, vis_features, agent_features], dim=1)
        return self.combine(combined)
