import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    """
    自定義獎勵包裝器 - 可以修改這個類來自定義獎勵函數
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 在這裡修改獎勵計算方式
        # 示例：根據不同條件調整獎勵
        custom_reward = reward
        
        # 示例1：如果得到正獎勵，將其加倍
        if reward > 0:
            custom_reward = reward * 2
            
        # 示例2：在遊戲結束時給予額外獎勵或懲罰
        if terminated:
            if reward > 0:
                custom_reward += 10  # 遊戲結束時有分數，給予額外獎勵
            else:
                custom_reward -= 5   # 遊戲結束時沒分數，給予懲罰
                
        return obs, custom_reward, terminated, truncated, info

class CustomObservationWrapper(gym.ObservationWrapper):
    """
    自定義觀察包裝器 - 可以修改這個類來處理遊戲狀態
    """
    def __init__(self, env):
        super().__init__(env)
        
    def observation(self, obs):
        # 在這裡修改觀察值的處理方式
        # 示例：將圖像轉為灰度圖
        # gray_obs = np.mean(obs, axis=2)
        # return gray_obs
        
        return obs  # 默認不做修改
