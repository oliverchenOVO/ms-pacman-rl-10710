"""
resume_pure_ppo.py — 從 checkpoint 接續訓練
"""
import sys, io, os, time
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, WarpFrame
)
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# Same config as train_pure_ppo.py
ENV_ID = "ALE/MsPacman-v5"
N_ENVS = 8
EVAL_FREQ = 100_000 // N_ENVS
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 1_000_000 // N_ENVS

CHECKPOINT_PATH = "models/checkpoints/pure_ppo/pacman_pure_ppo_11000000_steps.zip"
VECNORM_LOAD = "models/best/best_vecnormalize_pure_ppo.pkl"
EVAL_LOG = "logs/pure_ppo_evals.csv"
PID_FILE = "logs/pure_ppo_pid.txt"
CHECKPOINT_DIR = "models/checkpoints/pure_ppo"
BEST_MODEL = "models/best/best_model_pure_ppo.zip"
BEST_VECNORM = "models/best/best_vecnormalize_pure_ppo.pkl"
TARGET_SCORE = 15000
TOTAL_ADDITIONAL = 30_000_000  # Continue for 30M more

def _make_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = Monitor(env)
        return env
    return _init

class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fn, prev_best=-np.inf):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.best_mean_reward = prev_best  # preserve previous best
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps
            eval_env = DummyVecEnv([self.eval_env_fn])
            eval_env = VecFrameStack(eval_env, n_stack=4)
            mean_reward, std_reward = evaluate_policy(
                self.model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=False
            )
            eval_env.close()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ev = float('nan')
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                ev = self.model.logger.name_to_value.get('train/explained_variance', float('nan'))
            with open(EVAL_LOG, 'a', encoding='utf-8') as f:
                f.write(f"{ts},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f},{ev:.4f}\n")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(BEST_MODEL)
                self.training_env.save(BEST_VECNORM)
                print(f"\n🏆 新最佳！step={self.num_timesteps:,} mean={mean_reward:.2f}")
            print(f"📊 [Eval] step={self.num_timesteps:>11,} | mean={mean_reward:>10.2f} | "
                  f"std={std_reward:>8.2f} | best={self.best_mean_reward:>10.2f} | ev={ev:.3f} | {ts}")
            if mean_reward >= TARGET_SCORE:
                print(f"\n🎉 達標 {TARGET_SCORE}！")
                return False
        return True

def main():
    print(f"🔄 從 checkpoint 續跑: {CHECKPOINT_PATH}")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    # Read previous best score from eval log
    prev_best = -np.inf
    if os.path.exists(EVAL_LOG):
        with open(EVAL_LOG, 'r') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        s = float(parts[2])
                        if s > prev_best:
                            prev_best = s
                    except ValueError:
                        pass
    print(f"[OK] Previous best score: {prev_best:.1f}")

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    env_fns = [_make_env() for _ in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize.load(VECNORM_LOAD, train_env)
    train_env.training = True
    train_env.norm_reward = True
    print(f"[OK] Env + VecNormalize loaded")

    model = PPO.load(CHECKPOINT_PATH, env=train_env)
    print(f"[OK] Model loaded from 11M checkpoint")

    eval_cb = EvalCallback(eval_env_fn=_make_env(), prev_best=prev_best)
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_pure_ppo"
    )

    print(f"開始續跑 {TOTAL_ADDITIONAL:,} 步...\n")
    start_time = time.time()
    try:
        model.learn(total_timesteps=TOTAL_ADDITIONAL,
                    callback=CallbackList([eval_cb, checkpoint_cb]),
                    reset_num_timesteps=False)
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n結束 | {elapsed/3600:.1f}h | best={eval_cb.best_mean_reward:.2f}")

    model.save("models/MsPacman-v5_pure_ppo.zip")
    train_env.save("models/vecnormalize_pure_ppo.pkl")
    train_env.close()

if __name__ == '__main__':
    main()
