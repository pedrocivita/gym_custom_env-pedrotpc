import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        neighbors_shape = observation_space["neighbors"].shape
        agent_dim = observation_space["agent"].shape[0]

        self.neighbor_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out_dim = 32 * neighbors_shape[0] * neighbors_shape[1]

        self.agent_mlp = nn.Sequential(
            nn.Linear(agent_dim, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        neighbors = observations["neighbors"].unsqueeze(1).float()
        agent = observations["agent"].float()

        neighbor_features = self.neighbor_cnn(neighbors)
        agent_features = self.agent_mlp(agent)

        combined = torch.cat([neighbor_features, agent_features], dim=1)
        return self.combine(combined)
