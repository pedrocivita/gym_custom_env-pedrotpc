import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for v3.6 — minimalist (just neighbors 5x5 + agent vec).

    Inputs (Dict obs):
      - "agent": (7,) — pose, coverage, 4 directional unvisited ratios.
      - "neighbors": (5, 5) — local sensor view (free=0 / wall=1 / visited=2).

    Both inputs are size-invariant (always the same shape regardless of
    grid size), so weights transfer between 5x5 / 10x10 / 20x20 cleanly.
    Param count is identical across sizes (~110k), enabling curriculum
    learning without buffer mismatches.

    v3.6 reverts the over-specification of v3.5 (which added visited_map
    global as a 3rd spatial input and caused critic-policy collapse on
    10x10). The `neighbors` 5x5 already encodes visited cells (value 2),
    so a separate visited_neighbors channel was redundant. The directional
    ratios in the agent vector still give global "where to go" signal
    derived purely from the agent's own visited set.
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
