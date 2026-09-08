"""
train_lstm.py — PPO + LSTM (RecurrentPPO)
===========================================
Strategy G: 使用 RecurrentPPO 的 LSTM 策略網路。
LSTM 能夠記住跨幀的因果鏈（吃能量豆 → 50步後吃鬼），
解決純 CNN 的 credit assignment 瓶頸。

Key config:
  - RecurrentPPO + CnnLstmPolicy
  - n_steps=256 (加倍，利於 BPTT)
  - lstm_hidden_size=256
  - VecNormalize (reward only)
  - SubprocVecEnv x 8 (使用模組級 env factory)
"""

import sys, io, os, time, csv, json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame
)
import gymnasium as gym
import ale_py

# 從 game_environment 匯入模組級 env factory（解決 SubprocVecEnv pickle 問題）
gym.register_envs(ale_py)

from custom_wrappers import CustomRewardWrapper


# ===================== 配置 =====================

ENV_ID = "ALE/MsPacman-v5"
TOTAL_TIMESTEPS = 25_000_000
N_ENVS = 8
EVAL_FREQ = 100_000 // N_ENVS   # 每 100K total steps eval
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 500_000 // N_ENVS
TARGET_SCORE = 25000

OUTPUT_MODEL = "models/MsPacman-v5_lstm.zip"
VECNORM_PATH = "models/vecnormalize_lstm.pkl"
EVAL_LOG = "logs/lstm_evals.csv"
STATE_FILE = "logs/lstm_state.json"
PID_FILE = "logs/lstm_pid.txt"
CHECKPOINT_DIR = "models/checkpoints/lstm"
BEST_MODEL_DIR = "models/best"
BEST_MODEL = f"{BEST_MODEL_DIR}/best_model_lstm.zip"
BEST_VECNORM = f"{BEST_MODEL_DIR}/best_vecnormalize_lstm.pkl"

PPO_PARAMS = {
    'learning_rate': 2.5e-4,
    'n_steps': 256,             # 加倍（利於 LSTM BPTT）
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

LSTM_KWARGS = {
    'lstm_hidden_size': 256,
    'n_lstm_layers': 1,
    'shared_lstm': False,
    'enable_critic_lstm': True,
}


# ===================== 環境工廠（模組級，SubprocVecEnv 安全）=====================

def _make_training_env():
    """訓練環境工廠（pickle-safe）"""
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = CustomRewardWrapper(env)
        env = Monitor(env)
        return env
    return _init


def _make_eval_env():
    """評估環境（無自訂 reward，無 EpisodicLife）"""
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = Monitor(env)
        return env
    return _init


# ===================== LSTM Eval（處理 hidden state）=====================

def evaluate_lstm(model, env_fn, n_episodes=10, deterministic=False):
    """
    LSTM-aware evaluation: 手動處理 hidden state。
    每個 episode 開始時重置 LSTM state。
    """
    # Create single env
    env = DummyVecEnv([env_fn])
    env = VecFrameStack(env, n_stack=4)
    # No VecNormalize on eval env

    episode_rewards = []
    n_envs = 1

    for ep in range(n_episodes):
        obs = env.reset()
        lstm_states = None
        ep_reward = 0.0
        done = [False]

        while not done[0]:
            # Predict with LSTM states
            action, lstm_states = model.predict(
                obs, state=lstm_states, deterministic=deterministic
            )
            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0]
            done = dones

            # Reset LSTM states on episode boundary
            if done[0]:
                lstm_states = None

        episode_rewards.append(ep_reward)

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


# ===================== Callbacks =====================

class LstmEvalCallback(BaseCallback):
    """LSTM-aware evaluation + logging"""

    def __init__(self, eval_env_fn, log_path, best_model_path, best_vecnorm_path,
                 vecnorm_path, train_env):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.log_path = log_path
        self.best_model_path = best_model_path
        self.best_vecnorm_path = best_vecnorm_path
        self.vecnorm_path = vecnorm_path
        self.train_env = train_env
        self.best_mean_reward = -np.inf
        self.eval_history = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps

            mean_reward, std_reward = evaluate_lstm(
                self.model, self.eval_env_fn,
                n_episodes=N_EVAL_EPISODES, deterministic=False
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.eval_history.append({
                'step': self.num_timesteps,
                'mean': mean_reward,
                'std': std_reward,
                'time': timestamp
            })

            # Save CSV
            file_exists = os.path.exists(self.log_path)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("timestamp,step,mean_reward,std_reward\n")
                f.write(f"{timestamp},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f}\n")

            is_best = mean_reward > self.best_mean_reward
            if is_best:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                self.train_env.save(self.best_vecnorm_path)
                print(f"\n🏆 新最佳！step={self.num_timesteps:,} mean={mean_reward:.2f} std={std_reward:.2f}")

            print(f"📊 [Eval] step={self.num_timesteps:>11,} | mean={mean_reward:>10.2f} | "
                  f"std={std_reward:>8.2f} | best={self.best_mean_reward:>10.2f} | {timestamp}")

            if mean_reward >= TARGET_SCORE:
                print(f"\n{'='*60}")
                print(f"  🎉 達到目標 {TARGET_SCORE} 分！")
                print(f"{'='*60}")
                return False

        return True


# ===================== Main =====================

def main():
    print("=" * 60)
    print("  🧠 Strategy G: PPO + LSTM (RecurrentPPO)")
    print(f"  n_steps={PPO_PARAMS['n_steps']} | envs={N_ENVS} | lstm_hidden={LSTM_KWARGS['lstm_hidden_size']}")
    print(f"  Reward: score delta / 50 + VecNormalize")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # ========== Training Env ==========
    env_fns = [_make_training_env() for _ in range(N_ENVS)]
    try:
        train_env = SubprocVecEnv(env_fns)
        print(f"[OK] SubprocVecEnv x {N_ENVS}")
    except Exception as e:
        print(f"[WARN] SubprocVecEnv failed ({e}), fallback DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=10.0, gamma=PPO_PARAMS['gamma'])
    print("[OK] VecNormalize (reward only)")

    # ========== Model ==========
    policy_kwargs = dict(LSTM_KWARGS)
    model = RecurrentPPO(
        'CnnLstmPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        **PPO_PARAMS
    )
    print(f"[OK] RecurrentPPO created (CnnLstmPolicy, lstm_hidden={LSTM_KWARGS['lstm_hidden_size']})")

    # ========== Callbacks ==========
    eval_cb = LstmEvalCallback(
        eval_env_fn=_make_eval_env(),
        log_path=EVAL_LOG,
        best_model_path=BEST_MODEL,
        best_vecnorm_path=BEST_VECNORM,
        vecnorm_path=VECNORM_PATH,
        train_env=train_env
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="pacman_lstm"
    )

    callbacks = CallbackList([eval_cb, checkpoint_cb])

    # ========== Train ==========
    print(f"\n開始訓練 {TOTAL_TIMESTEPS:,} 步...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("\n⚠️ 訓練被中斷")
    except Exception as e:
        print(f"\n❌ 訓練錯誤: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  訓練結束 | {elapsed/3600:.1f}h | steps={model.num_timesteps:,}")
    print(f"  最佳: {eval_cb.best_mean_reward:.2f}")
    print(f"{'='*60}")

    model.save(OUTPUT_MODEL)
    train_env.save(VECNORM_PATH)
    train_env.close()

    if eval_cb.best_mean_reward >= TARGET_SCORE:
        print(f"\n🎉 達標！")


if __name__ == '__main__':
    main()
