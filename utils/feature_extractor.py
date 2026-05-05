import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for the v3 CPP observation.

    Inputs (Dict obs):
      - "agent": (7,) float vector — pose + coverage + 4 directional ratios.
      - "global_map": (3, H, W) float tensor — obstacle / visited / agent channels.

    Outputs a `features_dim` vector that is fed into the actor & critic heads.
    The CNN's output dimension is size-specific because each grid size
    (5x5, 10x10, 20x20) is trained as its own model.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        map_shape = observation_space["global_map"].shape  # (C, H, W)
        agent_dim = observation_space["agent"].shape[0]

        in_channels = map_shape[0]
        h, w = map_shape[1], map_shape[2]

        # Stride-2 in the second conv halves the spatial dim of the feature map,
        # cutting the downstream linear layer by ~4x for larger grids.
        # Output spatial size after conv2 with stride=2, kernel=3, padding=1 is
        # floor((h + 2 - 3) / 2) + 1.
        self.map_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        out_h = (h + 2 - 3) // 2 + 1
        out_w = (w + 2 - 3) // 2 + 1
        cnn_out_dim = 64 * out_h * out_w

        self.agent_mlp = nn.Sequential(
            nn.Linear(agent_dim, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        gmap = observations["global_map"].float()
        # Already (B, C, H, W) — no need to unsqueeze.
        agent = observations["agent"].float()

        map_features = self.map_cnn(gmap)
        agent_features = self.agent_mlp(agent)

        combined = torch.cat([map_features, agent_features], dim=1)
        return self.combine(combined)
