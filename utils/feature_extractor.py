import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for the v3.1 partial-observability CPP observation.

    Inputs (Dict obs):
      - "agent": (7,) — pose, coverage, 4 directional unvisited ratios.
      - "neighbors": (5, 5) — local sensor view (single channel of int codes).
      - "visited_map": (H, W) — agent-built memory of visited cells.

    The neighbor and visited_map paths each go through a small CNN; their
    flattened features are concatenated with the agent vector and projected
    to `features_dim`. Stride-2 in the second visited-map conv keeps params
    manageable for 10x10 and 20x20 grids.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        nbr_shape = observation_space["neighbors"].shape  # (5, 5)
        vmap_shape = observation_space["visited_map"].shape  # (H, W)
        agent_dim = observation_space["agent"].shape[0]

        # 5x5 sensor view CNN — small spatial input, so no stride.
        self.neighbor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        nbr_out_dim = 32 * nbr_shape[0] * nbr_shape[1]  # 32 * 25 = 800

        # Visited-map CNN — two stride-2 convs cut spatial dim by ~4x so the
        # combiner stays light on 20x20.
        self.visited_cnn = nn.Sequential(
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
            nn.Linear(nbr_out_dim + vmap_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # neighbors: (B, 5, 5) -> (B, 1, 5, 5)
        nbr = observations["neighbors"].unsqueeze(1).float()
        # visited_map: (B, H, W) -> (B, 1, H, W)
        vmap = observations["visited_map"].unsqueeze(1).float()
        agent = observations["agent"].float()

        nbr_features = self.neighbor_cnn(nbr)
        vmap_features = self.visited_cnn(vmap)
        agent_features = self.agent_mlp(agent)

        combined = torch.cat([nbr_features, vmap_features, agent_features], dim=1)
        return self.combine(combined)
