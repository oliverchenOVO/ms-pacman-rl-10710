from stable_baselines3 import PPO
from game_environment import GameEnvironment

def main():
    """
    示範如何使用遊戲環境，包括自定義獎勵和策略網絡
    """
    # 玩家信息
    player_name = 'starpig'
    student_id = 2
    
    # 創建遊戲環境
    # 注意：自定義獎勵和觀察只在訓練過程中使用
    # 在最終遊玩和提交分數時會使用原始環境，以確保公平性
    env = GameEnvironment(
        env_id="ALE/MsPacman-v5",
        use_custom_reward=True,    # 只在訓練時使用自定義獎勵函數
        use_custom_obs=False       # 不使用自定義觀察處理
    )
    
    # 定義模型參數 - 學生可以自由調整這些參數
    model_params = {
        'verbose': 1,
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,        # 折扣因子 - 決定未來獎勵的重要性
        'gae_lambda': 0.95,   # GAE參數 - 用於平衡偏差和方差
        'clip_range': 0.2,    # PPO裁剪範圍 - 限制策略更新的幅度
        'ent_coef': 0.01      # 熵係數 - 鼓勵探索
    }
    
    # 示範三種不同的訓練方式：
    
    # 1. 使用默認設置訓練新模型
    env.train(
        model_class=PPO,
        model_params=model_params,
        total_timesteps=800,
        force_train=True      # 強制重新訓練
    )
    
    # 2. 使用自定義策略網絡訓練新模型
    env.train(
        model_class=PPO,
        model_params=model_params,
        total_timesteps=800,
        force_train=True,
        use_custom_policy=True    # 使用自定義策略網絡
    )
    
    # 3. 繼續訓練已有的模型
    env.train(
        model_class=PPO,
        model_params=model_params,
        total_timesteps=400,      # 額外訓練400步
        continue_from="models/MsPacman-v5.zip"  # 指定要繼續訓練的模型路徑
    )
    
    # 使用原始環境玩遊戲並提交結果
    # 注意：這裡不會使用自定義獎勵或觀察，以確保評分公平性
    env.play(player_name, student_id)

if __name__ == '__main__':
    main()

"""
學生可以通過以下方式自定義強化學習模型：

1. 修改獎勵函數（僅影響訓練過程）：
   - 打開 custom_wrappers.py
   - 在 CustomRewardWrapper 類中修改 step 方法
   - 可以根據遊戲狀態設計新的獎勵計算方式
   - 注意：自定義獎勵只在訓練時使用，提交分數時使用原始環境

2. 修改觀察處理（僅影響訓練過程）：
   - 打開 custom_wrappers.py
   - 在 CustomObservationWrapper 類中修改 observation 方法
   - 可以對遊戲畫面進行預處理，如轉灰度、裁剪等
   - 注意：自定義觀察只在訓練時使用，提交分數時使用原始環境

3. 修改策略網絡：
   - 打開 custom_policy.py
   - 在 CustomCNN 類中修改網絡架構
   - 可以調整卷積層、全連接層的參數和結構

4. 調整訓練參數：
   - 在 example_usage.py 中修改 model_params 字典
   - 可以調整學習率、批次大小、訓練步數等參數

重要說明：
1. 自定義獎勵和觀察處理只在訓練過程中使用
2. 最終遊玩和提交分數時會使用原始環境，以確保所有學生的評分標準一致
3. 這種設計允許學生通過自定義訓練過程來優化模型，同時保持評分的公平性

建議：
1. 先使用默認設置運行遊戲，了解基本效果
2. 逐步調整單個組件（獎勵/觀察/網絡/參數）
3. 觀察修改後的訓練效果，進行進一步優化
4. 可以嘗試組合不同的修改來獲得最佳訓練效果
5. 記住最終評分是在原始環境中進行，所以要確保模型在原始環境中表現良好
"""
