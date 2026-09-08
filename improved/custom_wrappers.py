import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    """
    自定義獎勵包裝器 - 針對 Ms. Pac-Man 高分優化

    Ms. Pac-Man 原始獎勵結構:
    - 吃普通豆子: 10 分
    - 吃能量豆 (power pellet): 50 分
    - 吃鬼 (連續): 200 → 400 → 800 → 1600 分
    - 吃水果: 100~5000 分 (隨關卡增加)

    獎勵設計目標:
    1. 大幅鼓勵吃鬼 — 這是衝高分的核心
    2. 鼓勵吃能量豆 — 是吃鬼的前提
    3. 適度存活獎勵 + 積極移動
    4. 保持獎勵數值在合理範圍，避免訓練不穩定
    """

    def __init__(self, env):
        super().__init__(env)
        self._steps_without_reward = 0
        self._total_ghosts_eaten = 0
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps_without_reward = 0
        self._total_ghosts_eaten = 0
        self._prev_lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # === Strategy 1: Score Delta 獎勵 (v2.0) ===
        # 直接用 ALE 原始分數差值，保持真實分數比例
        # 吃第4鬼(+1600)=32.0 vs 吃豆(+10)=0.2，比例 160:1
        # agent 會被 PPO advantage 自然推向吃鬼策略
        shaped_reward = reward / 50.0

        # 追蹤有無進展
        if reward > 0:
            self._steps_without_reward = 0
            if reward >= 200:
                self._total_ghosts_eaten += 1
        else:
            self._steps_without_reward += 1

        # === 死亡懲罰 ===
        current_lives = self.env.unwrapped.ale.lives()
        if self._prev_lives is not None and current_lives < self._prev_lives:
            shaped_reward -= 1.0  # 死亡懲罰
        self._prev_lives = current_lives

        # === 輔助獎勵/懲罰 ===

        # 長時間沒有進展的懲罰（鼓勵積極移動和探索）
        if self._steps_without_reward > 60:
            shaped_reward -= 0.02
        elif self._steps_without_reward > 30:
            shaped_reward -= 0.005

        # 存活獎勵（鼓勵活得更久，但數值很小避免 agent 只顧存活）
        if not (terminated or truncated):
            shaped_reward += 0.002

        # === 遊戲結束的額外獎勵 ===
        # 高分結束時給予額外激勵（只在 terminated 且非 truncated 時）
        if terminated and not truncated:
            # 根據吃鬼數量給予終局獎勵
            shaped_reward += self._total_ghosts_eaten * 0.1

        return obs, shaped_reward, terminated, truncated, info


class CustomObservationWrapper(gym.ObservationWrapper):
    """
    自定義觀察包裝器 - 可在此處做額外的觀察值處理

    標準 Atari 預處理（灰度轉換、縮放至 84x84、幀堆疊）
    已在 game_environment.py 中自動處理。

    可選增強：
    - 歸一化像素值到 [0, 1]
    - 加入歷史幀差分（幀差可提供運動資訊）
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # 在這裡修改觀察值的處理方式
        # 標準預處理已自動應用，此處可做額外處理
        return obs  # 默認不做修改
