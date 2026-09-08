"""
train_ram_ppo.py — PPO + Ghost Channel + CustomCNN
===================================================
Ghost channel (RAM[20-23]) + raw score delta, 無 reward shaping。

架構:
  每幀: WarpFrame(84×84 gray) + GhostStateObsWrapper → (84,84,2)
  VecFrameStack(4) → (84,84,8) = 4 gray + 4 ghost_channel
  CustomCNNv2 auto-detect 8 input channels

獎勵:
  Raw ALE score delta (dot=10, ghost=200-1600 clipped→10, pellet=50)
  VecNormalize handles scale, no manual shaping
  讓 PPO 自己發現吃豆 vs 吃鬼的風險/回報平衡

VecNormalize: norm_reward=True, norm_obs=False
"""

import sys, io, os, time
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
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame
)
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from ram_wrappers import GhostStateObsWrapper
from custom_policy_v2 import CustomCNNv2


# ===================== 配置 =====================

ENV_ID = "ALE/MsPacman-v5"
TOTAL_TIMESTEPS = 25_000_000
N_ENVS = 8
EVAL_FREQ = 100_000 // N_ENVS
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 500_000 // N_ENVS
TARGET_SCORE = 15000

OUTPUT_MODEL = "models/MsPacman-v5_ram_ppo.zip"
VECNORM_PATH = "models/vecnormalize_ram_ppo.pkl"
EVAL_LOG = "logs/ram_ppo_evals.csv"
PID_FILE = "logs/ram_ppo_pid.txt"
CHECKPOINT_DIR = "models/checkpoints/ram_ppo"
BEST_MODEL = "models/best/best_model_ram_ppo.zip"
BEST_VECNORM = "models/best/best_vecnormalize_ram_ppo.pkl"

PPO_PARAMS = {
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
    'verbose': 1,
}


# ===================== 環境工廠 ======================

def _make_training_env():
    """訓練環境: Ghost channel + raw score delta"""
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # No EpisodicLifeEnv — agent needs full-game episodes
        # No SteppedReward — raw score delta, let PPO discover risk/reward
        env = WarpFrame(env)               # → (84,84,1)
        env = GhostStateObsWrapper(env)    # → (84,84,2)
        env = Monitor(env)
        return env
    return _init


def _make_eval_env():
    """Eval: Ghost channel + raw game score"""
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)               # → (84,84,1)
        env = GhostStateObsWrapper(env)    # → (84,84,2)  needed for model input
        env = Monitor(env)
        return env
    return _init


# ===================== Callback =====================

class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fn):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.best_mean_reward = -np.inf
        self._last_eval_step = 0
        self._phase10_saved = False  # 2.5M checkpoint flag

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps

            # Create fresh eval env each time (avoids state leakage)
            eval_env = DummyVecEnv([self.eval_env_fn])
            eval_env = VecFrameStack(eval_env, n_stack=4)

            mean_reward, std_reward = evaluate_policy(
                self.model, eval_env,
                n_eval_episodes=N_EVAL_EPISODES, deterministic=False
            )

            eval_env.close()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Read explained_variance from SB3 logger
            ev = float('nan')
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                ev = self.model.logger.name_to_value.get('train/explained_variance', float('nan'))

            with open(EVAL_LOG, 'a', encoding='utf-8') as f:
                if os.path.getsize(EVAL_LOG) == 0:
                    f.write("timestamp,step,mean_reward,std_reward,explained_variance\n")
                f.write(f"{ts},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f},{ev:.4f}\n")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(BEST_MODEL)
                self.training_env.save(BEST_VECNORM)
                print(f"\n🏆 新最佳！step={self.num_timesteps:,} mean={mean_reward:.2f}")

            # Phase checkpoint at 10% (2.5M steps)
            if not self._phase10_saved and self.num_timesteps >= 2_500_000:
                self._phase10_saved = True
                phase_path = "models/checkpoints/ram_ppo/phase_10pct.zip"
                self.model.save(phase_path)
                self.training_env.save("models/checkpoints/ram_ppo/phase_10pct_vecnorm.pkl")
                print(f"\n📦 Phase 10% checkpoint saved: {phase_path}")

            print(f"📊 [Eval] step={self.num_timesteps:>11,} | mean={mean_reward:>10.2f} | "
                  f"std={std_reward:>8.2f} | best={self.best_mean_reward:>10.2f} | ev={ev:.3f} | {ts}")

            if mean_reward >= TARGET_SCORE:
                print(f"\n🎉 達標 {TARGET_SCORE}！")
                return False
        return True


# ===================== Main =====================

def main():
    print("=" * 60)
    print("  🧬 RAM PPO: Ghost Channel + Raw Score Delta")
    print(f"  Input: (8,84,84) = 4 gray + 4 ghost_state")
    print(f"  CNN: CustomCNNv2 (32→64→128+ResBlock×2)")
    print(f"  Reward: Raw ALE score (dot=10, ghost clipped→10)")
    print(f"  Steps: {TOTAL_TIMESTEPS:,} | Envs: {N_ENVS}")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # ===== Training Env =====
    env_fns = [_make_training_env() for _ in range(N_ENVS)]
    try:
        train_env = SubprocVecEnv(env_fns)
        print(f"[OK] SubprocVecEnv x {N_ENVS}")
    except Exception as e:
        print(f"[WARN] SubprocVecEnv failed: {e}, fallback DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=10.0, gamma=PPO_PARAMS['gamma'])
    print("[OK] VecNormalize (reward only)")

    # ===== Model =====
    policy_kwargs = {
        'features_extractor_class': CustomCNNv2,
        'features_extractor_kwargs': {'features_dim': 512}
    }
    model = PPO('CnnPolicy', train_env, policy_kwargs=policy_kwargs, **PPO_PARAMS)
    print(f"[OK] PPO + CustomCNNv2 ({train_env.observation_space.shape[0]}ch input)")

    # ===== Callbacks =====
    eval_cb = EvalCallback(eval_env_fn=_make_eval_env())
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_ram_ppo"
    )

    # ===== Train =====
    print(f"\n開始訓練 {TOTAL_TIMESTEPS:,} 步...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=CallbackList([eval_cb, checkpoint_cb]))
    except KeyboardInterrupt:
        print("\n⚠️ 中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  結束 | {elapsed/3600:.1f}h | best={eval_cb.best_mean_reward:.2f}")
    print(f"{'='*60}")

    model.save(OUTPUT_MODEL)
    train_env.save(VECNORM_PATH)
    train_env.close()


if __name__ == '__main__':
    main()
