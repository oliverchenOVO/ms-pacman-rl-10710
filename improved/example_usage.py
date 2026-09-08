from stable_baselines3 import PPO
from game_environment import GameEnvironment
import os

def main():
    """
    使用優化的設定訓練 Ms. Pac-Man AI
    目標：達到 10,000 分以上

    關鍵優化:
    1. 標準 Atari 預處理（灰度、84x84、幀堆疊）- 大幅提升學習效率
    2. 8 個並行環境加速訓練
    3. 針對 Atari 優化的 PPO 超參數
    4. 自定義獎勵函數改善訓練信號
    5. 自動保存最佳模型 + 定期檢查點
    """
    # ========== 玩家信息 - 請修改為你的名字和學號 ==========
    player_name = 'starpig'
    student_id = 2

    # ========== 創建遊戲環境 ==========
    env = GameEnvironment(
        env_id="ALE/MsPacman-v5",
        use_custom_reward=True,    # 使用自定義獎勵函數（僅訓練時生效）
        use_custom_obs=False       # 不使用自定義觀察處理（標準預處理已自動應用）
    )

    # ========== PPO 超參數（針對 Atari 遊戲優化）==========
    # 來源: Stable-Baselines3 RL Zoo 的 Atari 最佳配置
    model_params = {
        'verbose': 1,
        'learning_rate': 2.5e-4,   # 學習率
        'n_steps': 128,            # 每個環境每次收集的步數
        'batch_size': 256,         # 小批次大小
        'n_epochs': 4,             # 每次更新的 epoch 數
        'gamma': 0.99,             # 折扣因子 - 決定未來獎勵的重要性
        'gae_lambda': 0.95,        # GAE 參數 - 平衡偏差和方差
        'clip_range': 0.1,         # PPO 裁剪範圍 - 限制策略更新幅度
        'ent_coef': 0.01,          # 熵係數 - 鼓勵探索
        'vf_coef': 0.5,            # 價值函數係數
        'max_grad_norm': 0.5,      # 梯度裁剪 - 防止梯度爆炸
    }

    # ========== 訓練模型 ==========
    # 使用 8 個並行環境加速訓練
    # 10,000,000 步約需 2-4 小時（SubprocVecEnv，視 CPU 而定）
    # 若使用 DummyVecEnv 回退，約需 10-15 小時
    print("=" * 60)
    print("  Ms. Pac-Man AI 訓練")
    print("  總步數: 10,000,000 | 並行環境: 8")
    print("  訓練期間會自動保存檢查點和最佳模型")
    print("=" * 60)

    env.train(
        model_class=PPO,
        model_params=model_params,
        total_timesteps=10_000_000,            # 再訓練 10M 步
        continue_from="models/MsPacman-v5.zip", # 從上次訓練的模型繼續
        n_envs=8
    )

    # ========== 載入最佳模型並遊玩 ==========
    # EvalCallback 在訓練期間會自動保存表現最好的模型
    best_model_path = "models/best/best_model.zip"
    if os.path.exists(best_model_path):
        print(f"\n載入訓練期間的最佳模型: {best_model_path}")
        env.load_model(PPO, best_model_path)
    else:
        print("\n使用最終訓練模型")

    # 使用原始環境玩遊戲並提交結果
    # 注意：遊玩時不會使用自定義獎勵，以確保評分公平性
    env.play(player_name, student_id)


if __name__ == '__main__':
    main()


"""
========== 進階使用說明 ==========

1. 繼續訓練以提高分數：
   如果 10M 步不夠，可以繼續訓練已有的模型：

   env.train(
       model_class=PPO,
       model_params=model_params,
       total_timesteps=5_000_000,
       continue_from="models/MsPacman-v5.zip",
       n_envs=8
   )

2. 使用自定義策略網絡：
   env.train(
       model_class=PPO,
       model_params=model_params,
       total_timesteps=10_000_000,
       force_train=True,
       use_custom_policy=True,    # 使用 custom_policy.py 中的 CNN
       n_envs=8
   )

3. 載入檢查點繼續訓練：
   env.train(
       model_class=PPO,
       model_params=model_params,
       total_timesteps=5_000_000,
       continue_from="models/checkpoints/pacman_5000000_steps.zip",
       n_envs=8
   )

4. 學生可自定義的部分：
   - 獎勵函數: 修改 custom_wrappers.py 中的 CustomRewardWrapper
   - CNN 架構: 修改 custom_policy.py 中的 CustomCNN
   - 超參數: 修改上方的 model_params 字典
   - 訓練步數: 增加 total_timesteps（更多步數 = 更高分數，但需更長時間）

5. 調參建議：
   - learning_rate: 嘗試 1e-4 ~ 5e-4
   - n_steps: 嘗試 128, 256, 512
   - ent_coef: 嘗試 0.005 ~ 0.05（更高 = 更多探索）
   - clip_range: 嘗試 0.1 ~ 0.3
   - 增加 total_timesteps 通常是提高分數最有效的方法
"""
