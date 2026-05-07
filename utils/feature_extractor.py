import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for v3.5 — three spatial paths + agent vector.

    Inputs (Dict obs):
      - "agent": (7,) — pose, coverage, 4 directional ratios.
      - "neighbors": (5, 5) — local sensor view (free / wall / visited).
      - "visited_neighbors": (5, 5) — binary memory in the same 5x5 window.
      - "visited_map": (H, W) — global binary memory of where the agent has
        already stepped. Variable spatial shape (size depends on grid),
        which is why v3.5 uses per-size training instead of curriculum.

    Each spatial input goes through its own small CNN, then everything is
    concatenated with the agent MLP and projected to features_dim. The
    visited_map CNN uses two stride-2 convs to keep the flatten size
    manageable on 20x20.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        nbr_shape = observation_space["neighbors"].shape          # (5, 5)
        vis_nbr_shape = observation_space["visited_neighbors"].shape  # (5, 5)
        vmap_shape = observation_space["visited_map"].shape       # (H, W)
        agent_dim = observation_space["agent"].shape[0]

        assert nbr_shape == vis_nbr_shape == (5, 5), "5x5 windows expected"

        # 5x5 sensor view CNN — small input, no stride.
        self.neighbor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        nbr_out_dim = 32 * 5 * 5  # 800

        # 5x5 visited-window CNN — symmetric to neighbors.
        self.visited_nbr_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        vis_nbr_out_dim = 32 * 5 * 5  # 800

        # Global visited_map CNN — two stride-2 convs to halve dim twice.
        self.visited_map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        def _ds(x: int) -> int:
            return (x + 2 - 3) // 2 + 1

        h, w = vmap_shape
        out_h = _ds(_ds(h))
        out_w = _ds(_ds(w))
        vmap_out_dim = 32 * out_h * out_w

        self.agent_mlp = nn.Sequential(
            nn.Linear(agent_dim, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(nbr_out_dim + vis_nbr_out_dim + vmap_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        nbr = observations["neighbors"].unsqueeze(1).float()
        vis_nbr = observations["visited_neighbors"].unsqueeze(1).float()
        vmap = observations["visited_map"].unsqueeze(1).float()
        agent = observations["agent"].float()

        nbr_features = self.neighbor_cnn(nbr)
        vis_nbr_features = self.visited_nbr_cnn(vis_nbr)
        vmap_features = self.visited_map_cnn(vmap)
        agent_features = self.agent_mlp(agent)

        combined = torch.cat(
            [nbr_features, vis_nbr_features, vmap_features, agent_features],
            dim=1,
        )
        return self.combine(combined)
