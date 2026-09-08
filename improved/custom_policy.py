import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ResidualBlock(nn.Module):
    """殘差塊：幫助梯度流動，讓更深層網路可訓練"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = th.relu(self.conv1(x))
        x = self.conv2(x)
        return th.relu(x + residual)


class CustomCNN(BaseFeaturesExtractor):
    """
    增強版 CNN 特徵提取器 — 針對 Ms. Pac-Man 高分優化

    改進點 vs 原版:
    - 更多 channel（32→64→128 vs 原版 32→64→64）
    - 加入殘差塊 (Residual Block) 讓更深網路可訓練
    - 更大的特徵維度 (512→1024)
    - 更好的感受野覆蓋

    輸入: (4, 84, 84) — 4 幀堆疊灰階
    輸出: 1024 維特徵向量
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 4

        # === 特徵提取主幹網路 ===
        self.cnn = nn.Sequential(
            # Layer 1: 降採樣 84→20
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # Layer 2: 降採樣 20→9
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Layer 3: 保持 9×9
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # 殘差塊: 7×7 特徵圖上做更深層處理
            ResidualBlock(128),
            ResidualBlock(128),

            nn.Flatten()
        )

        # 計算 CNN 輸出維度
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # === 全連接層 ===
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 輕微 dropout 防止過擬合
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
