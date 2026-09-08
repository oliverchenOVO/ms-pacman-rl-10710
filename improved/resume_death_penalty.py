"""
resume_death_penalty.py — 方案一：死亡懲罰 + 生存獎勵，從 36M checkpoint 續跑
=========================================================================
DeathPenaltyWrapper: 死亡=-50, 存活+0.005/step
訓練環境用 wrapped reward，eval 環境用 raw score（測真實分數）
新 VecNormalize 重新適應 reward 分佈
10M 步微調，FPS ~400 → ~7 小時
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
from gymnasium import Wrapper
import ale_py
gym.register_envs(ale_py)


# ===================== Death Penalty Wrapper =====================

class DeathPenaltyWrapper(Wrapper):
    """
    死亡時 reward -= DEATH_PENALTY，每步 reward += SURVIVAL_BONUS。
    放在 Monitor 之後，讓 Monitor 記錄原始 ALE reward。
    """
    DEATH_PENALTY = -50.0
    SURVIVAL_BONUS = 0.005

    def __init__(self, env):
        super().__init__(env)
        self._last_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_lives = self.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = self.unwrapped.ale.lives()

        # 死亡懲罰：偵測 lives 下降
        if current_lives < self._last_lives:
            reward += self.DEATH_PENALTY

        # 生存獎勵：每步微量
        reward += self.SURVIVAL_BONUS

        self._last_lives = current_lives
        return obs, reward, terminated, truncated, info


# ===================== 配置 =====================

ENV_ID = "ALE/MsPacman-v5"
N_ENVS = 8
EVAL_FREQ = 100_000 // N_ENVS  # eval every 12.5K steps
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 1_000_000 // N_ENVS
TARGET_SCORE = 15000

CHECKPOINT_PATH = "models/checkpoints/death_penalty/pacman_death_penalty_44000000_steps.zip"
TOTAL_ADDITIONAL = 2_000_000  # 2M remaining (target: 46M)

# New output paths (不覆蓋 pure_ppo 的檔案)
EVAL_LOG = "logs/death_penalty_evals.csv"
PID_FILE = "logs/death_penalty_pid.txt"
CHECKPOINT_DIR = "models/checkpoints/death_penalty"
BEST_MODEL = "models/best/best_model_death_penalty.zip"
BEST_VECNORM = "models/best/best_vecnormalize_death_penalty.pkl"


# ===================== 環境工廠 =====================

def _make_train_env():
    """訓練環境: 含 DeathPenaltyWrapper"""
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = Monitor(env)
        env = DeathPenaltyWrapper(env)  # ← 死亡懲罰
        return env
    return _init


def _make_eval_env():
    """Eval 環境: 純 raw score，無死亡懲罰（測真實分數）"""
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = Monitor(env)
        return env
    return _init


# ===================== Callback =====================

STATS_LOG = "logs/death_penalty_stats.txt"


class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fn, prev_best=-np.inf):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.best_mean_reward = prev_best
        self._last_eval_step = 0
        # Manual tracking (SB3 logger timing unreliable for rollout stats)
        self._n_envs = N_ENVS
        self._ep_lengths = []              # completed episode lengths
        self._current_ep_lens = [0] * self._n_envs  # per-env counter
        self._start_time = time.time()
        self._last_fps_step = 0
        self._last_fps_time = time.time()
        self._fps = float('nan')

    def _on_step(self) -> bool:
        # Track episode lengths from dones (per-env counters)
        dones = self.locals.get('dones', None)
        if dones is not None:
            for i in range(self._n_envs):
                self._current_ep_lens[i] += 1
                if dones[i]:
                    self._ep_lengths.append(self._current_ep_lens[i])
                    self._current_ep_lens[i] = 0
            # Keep only last 400 episodes
            if len(self._ep_lengths) > 400:
                self._ep_lengths = self._ep_lengths[-400:]

        # Update FPS every ~30 seconds
        now = time.time()
        if now - self._last_fps_time > 30 and self.num_timesteps > self._last_fps_step:
            steps_delta = self.num_timesteps - self._last_fps_step
            time_delta = now - self._last_fps_time
            self._fps = steps_delta / time_delta if time_delta > 0 else float('nan')
            self._last_fps_step = self.num_timesteps
            self._last_fps_time = now

        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps

            eval_env = DummyVecEnv([self.eval_env_fn])
            eval_env = VecFrameStack(eval_env, n_stack=4)

            mean_reward, std_reward = evaluate_policy(
                self.model, eval_env,
                n_eval_episodes=N_EVAL_EPISODES, deterministic=False
            )
            eval_env.close()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ev from logger (train stats are available)
            ev = float('nan')
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                ev = self.model.logger.name_to_value.get('train/explained_variance', float('nan'))

            # ep_len from manual tracking
            ep_len = np.mean(self._ep_lengths[-50:]) if self._ep_lengths else float('nan')

            with open(EVAL_LOG, 'a', encoding='utf-8') as f:
                if os.path.getsize(EVAL_LOG) == 0:
                    f.write("timestamp,step,mean_reward,std_reward,explained_variance\n")
                f.write(f"{ts},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f},{ev:.4f}\n")

            # Write compact stats for cron monitoring
            with open(STATS_LOG, 'w', encoding='utf-8') as f:
                f.write(f"step={self.num_timesteps}\n"
                        f"mean={mean_reward:.1f}\n"
                        f"best={self.best_mean_reward:.1f}\n"
                        f"ev={ev:.4f}\n"
                        f"ep_len={ep_len:.1f}\n"
                        f"fps={self._fps:.1f}\n"
                        f"ts={ts}\n")

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


# ===================== Main =====================

def main():
    print("=" * 60)
    print("  💀 Pure PPO + Death Penalty (方案一)")
    print(f"  Death: {DeathPenaltyWrapper.DEATH_PENALTY} | Survival: +{DeathPenaltyWrapper.SURVIVAL_BONUS}/step")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Additional steps: {TOTAL_ADDITIONAL:,}")
    print(f"  Fresh VecNormalize (new reward distribution)")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # ===== Training Env (含死亡懲罰, 新 VecNormalize) =====
    env_fns = [_make_train_env() for _ in range(N_ENVS)]
    try:
        train_env = SubprocVecEnv(env_fns)
        print(f"[OK] SubprocVecEnv x {N_ENVS}")
    except Exception as e:
        print(f"[WARN] SubprocVecEnv failed: {e}, fallback DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=100.0, gamma=0.99)
    print("[OK] Fresh VecNormalize (reward only)")

    # ===== Load model (CNN weights preserved, value head adapts) =====
    model = PPO.load(CHECKPOINT_PATH, env=train_env)
    print(f"[OK] Model loaded from 36M checkpoint")

    # Update PPO params for fine-tuning phase
    model.ent_coef = 0.01
    model.learning_rate = 1e-4  # Lower LR for fine-tuning (was 2.5e-4→0)
    print(f"[OK] LR=1e-4, ent_coef=0.01 (fine-tuning mode)")

    # ===== Callbacks =====
    eval_cb = EvalCallback(eval_env_fn=_make_eval_env())
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_death_penalty"
    )

    # ===== Train =====
    print(f"\n開始死亡懲罰微調 {TOTAL_ADDITIONAL:,} 步...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=TOTAL_ADDITIONAL,
                    callback=CallbackList([eval_cb, checkpoint_cb]),
                    reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\n⚠️ 中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  結束 | {elapsed/3600:.1f}h | best={eval_cb.best_mean_reward:.2f}")
    print(f"  總步數: {model.num_timesteps:,}")
    print(f"{'='*60}")

    model.save("models/MsPacman-v5_death_penalty.zip")
    train_env.save("models/vecnormalize_death_penalty.pkl")
    train_env.close()


if __name__ == '__main__':
    main()
