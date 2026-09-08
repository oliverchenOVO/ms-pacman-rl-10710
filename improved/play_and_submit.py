import sys
import io
# 修復 Windows 終端機中文亂碼，使用行緩衝確保即時輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from stable_baselines3 import PPO
from game_environment import GameEnvironment
import time
import os

def main():
    """
    載入最佳模型，遊玩 100 次並上傳結果
    """
    player_name = os.getenv('PACMAN_PLAYER_NAME', 'local-player')
    student_id = int(os.getenv('PACMAN_STUDENT_ID', '0'))

    # 創建遊戲環境（遊玩時不使用自定義獎勵）
    env = GameEnvironment(
        env_id="ALE/MsPacman-v5",
        use_custom_reward=False,
        use_custom_obs=False
    )

    # 載入最佳模型
    best_model_path = "models/best/best_model.zip"
    final_model_path = "models/MsPacman-v5.zip"

    if os.path.exists(best_model_path):
        print(f"載入最佳模型: {best_model_path}")
        env.load_model(PPO, best_model_path)
    elif os.path.exists(final_model_path):
        print(f"載入最終模型: {final_model_path}")
        env.load_model(PPO, final_model_path)
    else:
        print("找不到已訓練的模型！請先執行訓練。")
        return

    scores = []

    # 遊玩 100 次並上傳
    for i in range(1000):
        print(f"\n{'='*50}")
        print(f"  第 {i+1}/1000 次遊玩")
        print(f"{'='*50}")

        try:
            env.play(player_name, student_id, show_game=False)
        except Exception as e:
            print(f"第 {i+1} 次遊玩發生錯誤: {e}")

        # 每次遊玩之間等待 4 秒
        if i < 999: 
            print(f"等待 3 秒...")
            time.sleep(3)

    print(f"\n{'='*50}")
    print(f"  全部完成！共遊玩 1000 次")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
