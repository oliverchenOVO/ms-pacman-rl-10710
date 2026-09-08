"""
ram_wrappers.py — RAM-based Observation + PBRS Reward Shaping
==============================================================
提供兩個 wrapper，專為 Ms. Pac-Man 設計：

1. GhostStateObsWrapper: 從 ALE RAM 讀取鬼狀態，加入額外像素通道
   - RAM[20-23]: 鬼 0-3 狀態 (0=正常, 1=藍色/scared, 2=眼睛)
   - 任何鬼 scared → 通道值=255（亮點），否則=0

2. PBRSRewardWrapper: Potential-Based Reward Shaping (PBRS)
   - 基於 Ng et al. 1999 的理論保證不改變最優策略
   - Φ(s) 包含：鬼 scared 狀態、生存獎勵、死亡懲罰
   - 吃能量豆 → 鬼變藍 → 追鬼的因果鏈得到即時信號
"""

import gymnasium as gym
import numpy as np


class GhostStateObsWrapper(gym.ObservationWrapper):
    """
    將 ALE RAM[20-23] 的鬼狀態轉換為額外灰階通道。
    
    Input:  (H, W, 1) 灰階圖像 (WarpFrame 後)
    Output: (H, W, 2) 灰階 + ghost_state 通道
    
    ghost_state 通道：
      255 = 有鬼處於 scared 狀態（藍色，可吃）
      0   = 沒有 scared 鬼
    """
    
    def __init__(self, env):
        super().__init__(env)
        shp = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] + 1),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # Read ghost states from ALE RAM
        ale = self.unwrapped.ale
        ram = ale.getRAM()
        
        # RAM[20-23]: ghost 0-3 states (0=normal, 1=scared/blue, 2=eyes)
        any_scared = any(ram[20+i] == 1 for i in range(4))
        
        # Create ghost state channel
        ghost_channel = np.full((obs.shape[0], obs.shape[1], 1), 
                                255 if any_scared else 0, dtype=np.uint8)
        
        return np.concatenate([obs, ghost_channel], axis=2)


class PBRSRewardWrapper(gym.Wrapper):
    """
    Potential-Based Reward Shaping (PBRS)
    
    定理 (Ng et al. 1999): F(s, a, s') = γ·Φ(s') - Φ(s) 不改變最優策略。
    
    Φ(s) 設計：
      Φ_scared(s) = 5.0 if any ghost is scared else 0.0
      → 鬼變藍時 Φ 突然增大 → agent 得到 γ·5.0 - 0 = ~5.0 的信號
      → 鼓勵「吃能量豆」這個動作
      
      Φ_survive(s) = steps_alive * 0.001
      → 溫和的生存獎勵（PBRS 保證不濫用）
    
    搭配 stepped ghost reward（來自舊版 CustomRewardWrapper）：
      吃第1鬼=0.6, 第2鬼=1.2, 第3鬼=2.0, 第4鬼=3.0
      → 數值穩定，不會像 score delta 那樣崩潰
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._prev_any_scared = False
        self._total_ghosts_eaten = 0
        self._prev_lives = None
        self._steps_alive = 0
        self._prev_phi = 0.0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_any_scared = False
        self._total_ghosts_eaten = 0
        self._prev_lives = self.env.unwrapped.ale.lives()
        self._steps_alive = 0
        self._prev_phi = self._compute_phi()
        return obs, info
    
    def _compute_phi(self):
        """計算當前狀態的 potential Φ(s)"""
        ale = self.env.unwrapped.ale
        ram = ale.getRAM()
        any_scared = any(ram[20+i] == 1 for i in range(4))
        
        phi = 0.0
        # Φ_scared: 任何鬼 scared → +5.0
        if any_scared:
            phi += 5.0
        # Φ_survive: 線性生存 bonus
        phi += self._steps_alive * 0.001
        
        return phi
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # === 1. Stepped Ghost Reward (穩定版) ===
        shaped_reward = 0.0
        
        if reward > 0:
            if reward >= 1600:
                shaped_reward = 3.0       # 第 4 隻鬼
            elif reward >= 800:
                shaped_reward = 2.0       # 第 3 隻鬼
            elif reward >= 400:
                shaped_reward = 1.2       # 第 2 隻鬼
            elif reward >= 200:
                shaped_reward = 0.6       # 第 1 隻鬼
            elif reward >= 50:
                shaped_reward = 0.5       # 吃能量豆
            else:
                shaped_reward = 0.1       # 吃普通豆子
            
            if reward >= 200:
                self._total_ghosts_eaten += 1
        
        # === 2. PBRS (Potential-Based) ===
        self._steps_alive += 1
        current_phi = self._compute_phi()
        # F = γ·Φ(s') - Φ(s)   (γ≈0.99)
        pbrs_bonus = 0.99 * current_phi - self._prev_phi
        shaped_reward += pbrs_bonus
        self._prev_phi = current_phi
        
        # === 3. 死亡懲罰 ===
        current_lives = self.env.unwrapped.ale.lives()
        if self._prev_lives is not None and current_lives < self._prev_lives:
            shaped_reward -= 1.0
        self._prev_lives = current_lives
        
        # === 4. 存活獎勵（PBRS 已處理，此處小額補充）===
        if not (terminated or truncated):
            shaped_reward += 0.001
        
        # === 5. 終局獎勵 ===
        if terminated and not truncated:
            shaped_reward += self._total_ghosts_eaten * 0.1
        
        return obs, shaped_reward, terminated, truncated, info


class SteppedRewardWrapper(gym.Wrapper):
    """
    純 stepped ghost reward，無 PBRS。
    
    與 GhostChannel 搭配使用 — GhostChannel 提供鬼狀態資訊，
    值函數透過 bootstrap 自然回傳 credit，不需要 reward shaping。
    
    Reward 結構:
      吃第1鬼=0.6, 第2鬼=1.2, 第3鬼=2.0, 第4鬼=3.0
      吃能量豆=0.5, 吃普通豆=0.1
      死亡=-1.0, 存活=+0.001/step
    """

    def __init__(self, env):
        super().__init__(env)
        self._total_ghosts_eaten = 0
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._total_ghosts_eaten = 0
        self._prev_lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        # === Stepped Ghost Reward ===
        if reward > 0:
            if reward >= 1600:
                shaped_reward = 3.0       # 第 4 隻鬼
            elif reward >= 800:
                shaped_reward = 2.0       # 第 3 隻鬼
            elif reward >= 400:
                shaped_reward = 1.2       # 第 2 隻鬼
            elif reward >= 200:
                shaped_reward = 0.6       # 第 1 隻鬼
            elif reward >= 50:
                shaped_reward = 0.5       # 吃能量豆
            else:
                shaped_reward = 0.1       # 吃普通豆子

            if reward >= 200:
                self._total_ghosts_eaten += 1

        # === 死亡懲罰 ===
        current_lives = self.env.unwrapped.ale.lives()
        if self._prev_lives is not None and current_lives < self._prev_lives:
            shaped_reward -= 1.0
        self._prev_lives = current_lives

        # === 存活獎勵 ===
        if not (terminated or truncated):
            shaped_reward += 0.001

        return obs, shaped_reward, terminated, truncated, info
