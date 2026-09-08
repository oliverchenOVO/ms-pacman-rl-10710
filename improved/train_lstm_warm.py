"""
train_lstm_warm.py — LSTM warm-start 從 avg10500
================================================
載入 avg10500 CNN 權重 → CnnLstmPolicy → RecurrentPPO
前 2M 步凍結 CNN 只練 LSTM + value head，後 8M 解凍全體微調。
Death Penalty + clip_reward=100，無 EpisodicLifeEnv，n_stack=4。
"""
import sys, io, os, time
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, WarpFrame
import gymnasium as gym
from gymnasium import Wrapper
import ale_py
gym.register_envs(ale_py)


# ===================== Death Penalty Wrapper =====================

class DeathPenaltyWrapper(Wrapper):
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
        if self.unwrapped.ale.lives() < self._last_lives:
            reward += self.DEATH_PENALTY
        reward += self.SURVIVAL_BONUS
        self._last_lives = self.unwrapped.ale.lives()
        return obs, reward, terminated, truncated, info


# ===================== 配置 =====================

ENV_ID = "ALE/MsPacman-v5"
N_ENVS = 8
N_STACK = 4
EVAL_FREQ = 100_000 // N_ENVS
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 1_000_000 // N_ENVS
TARGET_SCORE = 20000

SOURCE_MODEL = "models/MsPacman-v5_avg10500.zip"   # CNN warm-start source
TOTAL_TIMESTEPS = 20_000_000                          # total LSTM training
FREEZE_CNN_STEPS = 2_000_000                          # freeze CNN, train LSTM only

EVAL_LOG = "logs/lstm_warm_evals.csv"
PID_FILE = "logs/lstm_warm_pid.txt"
CHECKPOINT_DIR = "models/checkpoints/lstm_warm"
BEST_MODEL = "models/best/best_model_lstm_warm.zip"
BEST_VECNORM = "models/best/best_vecnormalize_lstm_warm.pkl"
STATS_LOG = "logs/lstm_warm_stats.txt"


# ===================== 環境工廠 =====================

def _make_train_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = Monitor(env)
        env = DeathPenaltyWrapper(env)
        return env
    return _init


def _make_eval_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = Monitor(env)
        return env
    return _init


# ===================== Callbacks =====================

class VecNormCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._last_save = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_save) >= self.save_freq:
            self._last_save = self.num_timesteps
            path = f"{self.save_path}/{self.name_prefix}_{self.num_timesteps}_steps_vecnorm.pkl"
            self.training_env.save(path)
        return True


class UnfreezeCallback(BaseCallback):
    """After FREEZE_CNN_STEPS, unfreeze CNN layers."""
    def __init__(self, unfreeze_at):
        super().__init__()
        self.unfreeze_at = unfreeze_at
        self._triggered = False

    def _on_step(self) -> bool:
        if not self._triggered and self.num_timesteps >= self.unfreeze_at:
            self._triggered = True
            for param in self.model.policy.features_extractor.parameters():
                param.requires_grad = True
            print(f"\n🔓 CNN 解凍！step={self.num_timesteps:,}")
        return True


class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fn):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.best_mean_reward = -np.inf
        self._last_eval_step = 0
        self._n_envs = N_ENVS
        self._ep_lengths = []
        self._current_ep_lens = [0] * self._n_envs
        self._last_fps_step = 0
        self._last_fps_time = time.time()
        self._fps = float('nan')

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', None)
        if dones is not None:
            for i in range(self._n_envs):
                self._current_ep_lens[i] += 1
                if dones[i]:
                    self._ep_lengths.append(self._current_ep_lens[i])
                    self._current_ep_lens[i] = 0
            if len(self._ep_lengths) > 400:
                self._ep_lengths = self._ep_lengths[-400:]

        now = time.time()
        if now - self._last_fps_time > 30 and self.num_timesteps > self._last_fps_step:
            d = self.num_timesteps - self._last_fps_step
            t = now - self._last_fps_time
            self._fps = d / t if t > 0 else float('nan')
            self._last_fps_step = self.num_timesteps
            self._last_fps_time = now

        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps

            eval_env = DummyVecEnv([self.eval_env_fn])
            eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
            mean_reward, std_reward = evaluate_policy(
                self.model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=False
            )
            eval_env.close()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ev = float('nan')
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                ev = self.model.logger.name_to_value.get('train/explained_variance', float('nan'))

            ep_len = np.mean(self._ep_lengths[-50:]) if self._ep_lengths else float('nan')

            with open(EVAL_LOG, 'a', encoding='utf-8') as f:
                if os.path.getsize(EVAL_LOG) == 0:
                    f.write("timestamp,step,mean_reward,std_reward,explained_variance\n")
                f.write(f"{ts},{self.num_timesteps},{mean_reward:.2f},{std_reward:.2f},{ev:.4f}\n")

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
    print("  🧠 LSTM Warm-Start from avg10500")
    print(f"  Source: {SOURCE_MODEL}")
    print(f"  Freeze CNN: 0→{FREEZE_CNN_STEPS/1e6:.0f}M steps")
    print(f"  Total: {TOTAL_TIMESTEPS/1e6:.0f}M steps")
    print(f"  Death: -50 | Survival: +0.005 | clip=100")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # ===== Env =====
    env_fns = [_make_train_env() for _ in range(N_ENVS)]
    try:
        train_env = SubprocVecEnv(env_fns)
        print(f"[OK] SubprocVecEnv x {N_ENVS}")
    except Exception as e:
        print(f"[WARN] SubprocVecEnv failed: {e}, fallback DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    train_env = VecFrameStack(train_env, n_stack=N_STACK)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=100.0, gamma=0.99)
    print("[OK] VecNormalize (fresh, clip=100)")

    # ===== Load source CNN model =====
    print(f"[*] Loading source CNN model: {SOURCE_MODEL}")
    src_model = PPO.load(SOURCE_MODEL)
    src_cnn_state = src_model.policy.features_extractor.state_dict()
    print("[OK] Source CNN weights extracted")

    # ===== Create RecurrentPPO with CnnLstmPolicy =====
    model = RecurrentPPO(
        RecurrentActorCriticCnnPolicy, train_env,
        n_steps=512, batch_size=512, n_epochs=4,
        gamma=0.99, gae_lambda=0.99, clip_range=0.1,
        ent_coef=0.01, vf_coef=1.5, max_grad_norm=0.5,
        learning_rate=1e-4,  # lower LR for fine-tuning
        verbose=1,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            enable_critic_lstm=False,
        )
    )
    print(f"[OK] RecurrentPPO + RecurrentActorCriticCnnPolicy (lstm_hidden=256)")

    # ===== Warm-start: copy CNN weights =====
    model.policy.features_extractor.load_state_dict(src_cnn_state)
    print("[OK] CNN weights transferred from avg10500")

    # ===== Freeze CNN =====
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
    print(f"[OK] CNN frozen for first {FREEZE_CNN_STEPS/1e6:.0f}M steps")

    # ===== Callbacks =====
    eval_cb = EvalCallback(eval_env_fn=_make_eval_env())
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_lstm_warm"
    )
    vecnorm_cb = VecNormCheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_lstm_warm"
    )
    unfreeze_cb = UnfreezeCallback(unfreeze_at=FREEZE_CNN_STEPS)

    print(f"\n開始 LSTM 訓練 {TOTAL_TIMESTEPS/1e6:.0f}M 步...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=CallbackList([eval_cb, checkpoint_cb, vecnorm_cb, unfreeze_cb]))
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

    model.save("models/MsPacman-v5_lstm_warm.zip")
    train_env.save("models/vecnormalize_lstm_warm.pkl")
    train_env.close()


if __name__ == '__main__':
    main()
