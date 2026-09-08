"""
train_strategy1.py — Strategy 1: Score Delta Reward
=====================================================
從備份模型繼續訓練，使用新的 Score Delta 獎勵函數。
單一變數測試：只改 reward，其他不變。

Author: Hermes Agent
"""

import sys
import io
import os
import time
from datetime import datetime

# 修復 Windows 終端機中文亂碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from game_environment import GameEnvironment


# ===================== 配置 =====================

# 備份模型路徑 (Phase 0 備份的)
BACKUP_TIMESTAMP = "20260501_000639"
BACKUP_MODEL_PATH = f"models/backup/MsPacman-v5_backup_{BACKUP_TIMESTAMP}.zip"

# 輸出模型路徑
OUTPUT_MODEL_PATH = "models/MsPacman-v5_strategy1.zip"

# 訓練超參數
TOTAL_TIMESTEPS = 20_000_000   # 總步數
N_ENVS = 8                     # 並行環境數
EVAL_FREQ = 100_000 // N_ENVS  # 每 100K 步評估一次 (除以 n_envs)
N_EVAL_EPISODES = 10           # 每次評估 episode 數
CHECKPOINT_FREQ = 500_000 // N_ENVS  # 每 500K 保存檢查點

PPO_PARAMS = {
    'verbose': 1,
    'learning_rate': 1e-4,      # 中等學習率 — 我們要從舊模型適應新 reward
    'n_steps': 256,
    'batch_size': 256,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.15,
    'ent_coef': 0.02,           # 較高 entropy — 幫助 escape 舊的 local optimum
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

# 成功目標
TARGET_SCORE = 25000

# 監測用 log 檔案
MONITOR_LOG = "logs/strategy1_evals.csv"


# ===================== 自定義 Callback =====================

class EvalLogCallback(BaseCallback):
    """將每次 eval 的結果寫入 CSV 供監測 cronjob 讀取"""

    def __init__(self, eval_freq, n_eval_episodes, eval_env, log_path, best_model_save_path, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = eval_env
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf
        self.eval_history = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps

            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=False
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.eval_history.append({
                'step': self.num_timesteps,
                'mean': mean_reward,
                'std': std_reward,
                'time': timestamp
            })

            # 寫入 CSV log
            file_exists = os.path.exists(self.log_path)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("timestamp,step,mean_reward,std_reward\n")
                f.write(f"{timestamp},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f}\n")

            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_save_path)
                print(f"\n🏆 新最佳模型！step={self.num_timesteps:,} mean={mean_reward:.2f} std={std_reward:.2f}")

            print(f"📊 [Eval] step={self.num_timesteps:>12,} | mean={mean_reward:>10.2f} | std={std_reward:>8.2f} | best={self.best_mean_reward:>10.2f} | {timestamp}")

            # 檢查是否達到目標
            if mean_reward >= TARGET_SCORE:
                print(f"\n{'='*60}")
                print(f"  🎉🎉🎉 達到目標 {TARGET_SCORE} 分！🎉🎉🎉")
                print(f"  最終分數: {mean_reward:.2f}")
                print(f"{'='*60}")
                self.model.save(OUTPUT_MODEL_PATH)
                return False  # 停止訓練

        return True


# ===================== Main =====================

def main():
    print("=" * 60)
    print("  🎯 Strategy 1: Score Delta Reward")
    print("  Reward = ALE分數差 / 50")
    print(f"  總步數: {TOTAL_TIMESTEPS:,} | 環境數: {N_ENVS} | Eval: 每 {EVAL_FREQ * N_ENVS:,} 步")
    print(f"  備份模型: {BACKUP_MODEL_PATH}")
    print("=" * 60)

    # 初始化 log + PID
    os.makedirs("logs", exist_ok=True)
    pid_file = "logs/strategy1_pid.txt"
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    # 創建遊戲環境（使用新 reward wrapper）
    env = GameEnvironment(
        env_id="ALE/MsPacman-v5",
        use_custom_reward=True,    # ← 這裡吃到 Strategy 1 的 score delta reward
        use_custom_obs=False
    )

    # 創建評估環境（無自定義 reward，直接用原始分數）
    eval_env = env.create_eval_vec_env()

    # Callbacks
    eval_log_callback = EvalLogCallback(
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_env=eval_env,
        log_path=MONITOR_LOG,
        best_model_save_path="models/best/best_model_strategy1.zip"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./models/checkpoints/strategy1/",
        name_prefix="pacman_s1"
    )

    callbacks = CallbackList([eval_log_callback, checkpoint_callback])

    # 載入備份模型繼續訓練
    print(f"\n載入備份模型: {BACKUP_MODEL_PATH}")
    vec_env = env.create_vec_env(n_envs=N_ENVS)
    model = PPO.load(BACKUP_MODEL_PATH, env=vec_env)

    # 設定新的超參數（跳過 buffer-dependent 和 callable 屬性）
    skip_keys = {'n_steps', 'batch_size', 'clip_range', 'learning_rate'}
    for key, value in PPO_PARAMS.items():
        if key in skip_keys:
            continue
        if hasattr(model, key) and not callable(getattr(model, key)):
            setattr(model, key, value)
    model.n_epochs = PPO_PARAMS['n_epochs']
    model.ent_coef = PPO_PARAMS['ent_coef']
    model.vf_coef = PPO_PARAMS['vf_coef']
    model.max_grad_norm = PPO_PARAMS['max_grad_norm']

    print(f"\n[模型參數] lr={model.lr_schedule(1.0) if callable(getattr(model, 'lr_schedule', None)) else '?'}, "
          f"ent_coef={model.ent_coef}, "
          f"clip(原始)={model.clip_range(0.5) if callable(getattr(model, 'clip_range', None)) else model.clip_range}, "
          f"n_steps={model.n_steps}, n_epochs={model.n_epochs}")
    print(f"開始訓練 {TOTAL_TIMESTEPS:,} 步...\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            reset_num_timesteps=False,
            callback=callbacks
        )
    except KeyboardInterrupt:
        print("\n⚠️  訓練被手動中斷")
    except Exception as e:
        print(f"\n❌ 訓練發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    final_steps = model.num_timesteps

    print(f"\n{'='*60}")
    print(f"  訓練階段結束")
    print(f"  總耗時: {hours:.1f} 小時 | 總步數: {final_steps:,}")
    print(f"  最佳 eval 分數: {eval_log_callback.best_mean_reward:.2f}")
    print(f"{'='*60}")

    # 保存最終模型
    model.save(OUTPUT_MODEL_PATH)
    print(f"最終模型已保存: {OUTPUT_MODEL_PATH}")

    vec_env.close()
    eval_env.close()

    # 輸出 eval 歷史摘要
    if eval_log_callback.eval_history:
        print(f"\n📈 Eval 歷史 ({len(eval_log_callback.eval_history)} 次):")
        for h in eval_log_callback.eval_history:
            marker = " 🏆" if h['mean'] == eval_log_callback.best_mean_reward else ""
            print(f"   step={h['step']:>12,} | mean={h['mean']:>10.2f} | std={h['std']:>8.2f}{marker}")

    # 檢查是否達標
    if eval_log_callback.best_mean_reward >= TARGET_SCORE:
        print(f"\n🎉 成功達到 {TARGET_SCORE} 分！可以執行 play_and_submit.py 上傳結果。")
        return True
    else:
        print(f"\n⚠️  未達標 (best={eval_log_callback.best_mean_reward:.2f} < target={TARGET_SCORE})，需要進一步調整。")
        return False


if __name__ == '__main__':
    main()
