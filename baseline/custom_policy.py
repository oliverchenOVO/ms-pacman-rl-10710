import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CustomCNN(BaseFeaturesExtractor):
    """
    自定義CNN特徵提取器 - 學生可以修改網絡架構
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # 獲取輸入維度
        n_input_channels = observation_space.shape[0]
        
        # 定義CNN網絡結構
        self.cnn = nn.Sequential(
            # 第一層卷積
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # 第二層卷積
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # 第三層卷積
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # 展平層
            nn.Flatten()
        )
        
        # 計算CNN輸出特徵維度
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        # 全連接層
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        前向傳播函數
        Args:
            observations: 輸入的觀察值
        Returns:
            提取的特徵
        """
        return self.linear(self.cnn(observations))
