"""
train_from_scratch.py — 從頭訓練 + VecNormalize
================================================
Strategy D: 從零開始，使用 VecNormalize 消除 reward scale 問題。
從 Strategy 1 的崩潰學到：沒有 normalization，PPO 無法承受
60:2 的 reward spike（吃 4 鬼 vs 走 1000 步）。

Key changes vs Strategy 1:
  - VecNormalize(norm_obs=True, norm_reward=True, clip_reward=10.0)
  - 從 scratch 開始（無舊模型行為固化）
  - 保守超參數（標準 Atari PPO 配置）
  - 自動保存 VecNormalize stats
"""

import sys
import io
import os
import time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame
)
from custom_wrappers import CustomRewardWrapper
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)


# ===================== 配置 =====================

ENV_ID = "ALE/MsPacman-v5"
TOTAL_TIMESTEPS = 25_000_000
N_ENVS = 8
EVAL_FREQ = 100_000 // N_ENVS
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 500_000 // N_ENVS
TARGET_SCORE = 25000

OUTPUT_MODEL_PATH = "models/MsPacman-v5_scratch.zip"
VECNORM_PATH = "models/vecnormalize_scratch.pkl"
MONITOR_LOG = "logs/scratch_evals.csv"
PID_FILE = "logs/scratch_pid.txt"

PPO_PARAMS = {
    'verbose': 1,
    'learning_rate': 2.5e-4,
    'n_steps': 128,
    'batch_size': 256,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.1,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}


# ===================== 環境工廠 =====================

def make_env(env_id, training=True):
    """建立 Atari 環境（pickle-safe for SubprocVecEnv）"""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if training:
            env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        if training:
            env = CustomRewardWrapper(env)  # Score delta reward
        env = Monitor(env)
        return env
    return _init


# ===================== Callbacks =====================

class EvalLogCallback(BaseCallback):
    """Eval + CSV logging + best model save"""

    def __init__(self, eval_env, log_path, best_model_path, vecnorm_path=None, train_env=None):
        super().__init__()
        self.eval_env = eval_env
        self.log_path = log_path
        self.best_model_path = best_model_path
        self.vecnorm_path = vecnorm_path
        self.train_env = train_env
        self.best_mean_reward = -np.inf
        self.eval_history = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps

            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=False
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.eval_history.append({
                'step': self.num_timesteps,
                'mean': mean_reward,
                'std': std_reward,
                'time': timestamp
            })

            # Write CSV
            file_exists = os.path.exists(self.log_path)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("timestamp,step,mean_reward,std_reward\n")
                f.write(f"{timestamp},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f}\n")

            # Save best
            is_best = mean_reward > self.best_mean_reward
            if is_best:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                # Save VecNormalize stats
                if self.vecnorm_path and self.train_env:
                    self.train_env.save(self.vecnorm_path)
                print(f"\n🏆 新最佳！step={self.num_timesteps:,} mean={mean_reward:.2f} std={std_reward:.2f}")

            print(f"📊 [Eval] step={self.num_timesteps:>11,} | mean={mean_reward:>10.2f} | "
                  f"std={std_reward:>8.2f} | best={self.best_mean_reward:>10.2f} | {timestamp}")

            # Check target
            if mean_reward >= TARGET_SCORE:
                print(f"\n{'='*60}")
                print(f"  🎉🎉🎉 達到目標 {TARGET_SCORE} 分！🎉🎉🎉")
                print(f"  最終分數: {mean_reward:.2f}")
                print(f"{'='*60}")
                return False

        return True


# ===================== Main =====================

def main():
    print("=" * 60)
    print("  🎯 Strategy D: 從頭訓練 + VecNormalize")
    print(f"  環境: {ENV_ID} | 步數: {TOTAL_TIMESTEPS:,} | Envs: {N_ENVS}")
    print(f"  VecNormalize: obs=True, reward=True, clip_reward=10.0")
    print(f"  Reward: ALE score delta / 50 (CustomRewardWrapper)")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints/scratch", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    # Write PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # ========== Training Env (with VecNormalize) ==========
    env_fns = [make_env(ENV_ID, training=True) for _ in range(N_ENVS)]
    try:
        train_env = SubprocVecEnv(env_fns)
        print(f"[OK] SubprocVecEnv x {N_ENVS}")
    except Exception as e:
        print(f"[WARN] SubprocVecEnv failed ({e}), using DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize(
        train_env,
        norm_obs=False,        # Atari 圖像由 NatureCNN 內部處理
        norm_reward=True,      # ← 這才是解決 reward spike 的關鍵
        clip_reward=10.0,
        gamma=PPO_PARAMS['gamma']
    )
    print(f"[OK] VecNormalize applied (reward only, clip=10.0)")

    # ========== Eval Env (raw, no normalization) ==========
    eval_env_fn = make_env(ENV_ID, training=False)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    # NO VecNormalize on eval env — we want raw game scores

    # ========== Callbacks ==========
    eval_log_cb = EvalLogCallback(
        eval_env=eval_env,
        log_path=MONITOR_LOG,
        best_model_path="models/best/best_model_scratch.zip",
        vecnorm_path=VECNORM_PATH,
        train_env=train_env
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./models/checkpoints/scratch/",
        name_prefix="pacman_scratch"
    )

    callbacks = CallbackList([eval_log_cb, checkpoint_cb])

    # ========== Create Model ==========
    model = PPO('CnnPolicy', train_env, **PPO_PARAMS)
    print(f"\n模型建立: CnnPolicy, "
          f"lr={PPO_PARAMS['learning_rate']}, "
          f"ent_coef={PPO_PARAMS['ent_coef']}, "
          f"clip={PPO_PARAMS['clip_range']}")
    print(f"開始訓練 {TOTAL_TIMESTEPS:,} 步...\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks
        )
    except KeyboardInterrupt:
        print("\n⚠️  訓練被手動中斷")
    except Exception as e:
        print(f"\n❌ 訓練錯誤: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  訓練結束 | 耗時: {elapsed/3600:.1f}h | 步數: {model.num_timesteps:,}")
    print(f"  最佳 eval: {eval_log_cb.best_mean_reward:.2f}")
    print(f"{'='*60}")

    # Save final
    model.save(OUTPUT_MODEL_PATH)
    train_env.save(VECNORM_PATH)
    print(f"模型: {OUTPUT_MODEL_PATH}")
    print(f"VecNorm: {VECNORM_PATH}")

    train_env.close()
    eval_env.close()

    if eval_log_cb.best_mean_reward >= TARGET_SCORE:
        print(f"\n🎉 達標 {TARGET_SCORE}！可提交。")
        return True
    else:
        print(f"\n⚠️ 未達標 (best={eval_log_cb.best_mean_reward:.1f})，需進一步調整。")
        return False


if __name__ == '__main__':
    main()
