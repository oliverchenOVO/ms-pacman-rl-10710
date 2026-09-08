"""
play_avg10500_until_target.py
用 avg10500 模型反覆玩 Ms. Pac-Man 並上傳，直到分數 ≥ 13,000。
每局都上傳，達里程碑時通知。
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from stable_baselines3 import PPO
from game_environment import GameEnvironment

PLAYER = os.getenv('PACMAN_PLAYER_NAME', 'local-player')
STUDENT_ID = int(os.getenv('PACMAN_STUDENT_ID', '0'))
MODEL_PATH = "models/MsPacman-v5_avg10500.zip"
MILESTONES = [12000, 12500, 13000]  # 通知點，最後一個達標就停

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 找不到: {MODEL_PATH}")
        return

    env = GameEnvironment(env_id="ALE/MsPacman-v5", use_custom_reward=False, use_custom_obs=False)
    env.load_model(PPO, MODEL_PATH)
    print(f"[OK] {MODEL_PATH}")

    total_games = 0
    best = 0
    hit = set()  # 已通知過的里程碑
    start = time.time()

    while True:
        total_games += 1
        print(f"\n{'='*40}")
        print(f"  第 {total_games} 局 | 最佳 {best:.0f} | 目標 13000")
        print(f"{'='*40}")

        try:
            score = env.play(PLAYER, STUDENT_ID, show_game=False)
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(3)
            continue

        best = max(best, score)
        elapsed_h = (time.time() - start) / 3600
        print(f"🎮 score={score:.0f} | best={best:.0f} | {total_games}局 | {elapsed_h:.1f}h")

        # 檢查里程碑
        for m in MILESTONES:
            if score >= m and m not in hit:
                hit.add(m)
                if m == MILESTONES[-1]:
                    print(f"\n{'🎉'*20}")
                    print(f"  🏆 達標！{score:.0f} ≥ {m}！")
                    print(f"  第 {total_games} 局 | 耗時 {elapsed_h:.1f}h")
                    print(f"{'🎉'*20}")
                    return
                else:
                    print(f"\n{'🔔'*10}")
                    print(f"  ✅ 破 {m}！score={score:.0f}，繼續衝 {MILESTONES[-1]}...")
                    print(f"{'🔔'*10}")

        time.sleep(3)

if __name__ == '__main__':
    main()
