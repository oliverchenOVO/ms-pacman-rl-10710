"""
custom_policy_v2.py — CustomCNN with configurable input channels
================================================================
支援 4 通道（純灰階）或 5 通道（灰階 + ghost state）。
"""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = th.relu(self.conv1(x))
        x = self.conv2(x)
        return th.relu(x + residual)


class CustomCNNv2(BaseFeaturesExtractor):
    """
    增強版 CNN — 支援可變輸入通道數。
    
    輸入: (C, 84, 84)  C=4 (純灰階) or 5 (灰階+ghost_state)
    輸出: features_dim 維特徵向量
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
