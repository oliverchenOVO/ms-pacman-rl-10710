"""
train_wide_ppo.py — 寬視野 Pure PPO + Death Penalty
=====================================================
VecFrameStack=32 (2.13s 視覺記憶) + DeathPenaltyWrapper + clip_reward=100
從頭訓練 40M 步。NatureCNN 從 (32,84,84) 輸入。
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
TOTAL_TIMESTEPS = 40_000_000
N_ENVS = 8
N_STACK = 32           # 2.13s visual memory
EVAL_FREQ = 100_000 // N_ENVS
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 1_000_000 // N_ENVS
TARGET_SCORE = 15000

EVAL_LOG = "logs/wide_ppo_evals.csv"
PID_FILE = "logs/wide_ppo_pid.txt"
CHECKPOINT_DIR = "models/checkpoints/wide_ppo"
BEST_MODEL = "models/best/best_model_wide_ppo.zip"
BEST_VECNORM = "models/best/best_vecnormalize_wide_ppo.pkl"
STATS_LOG = "logs/wide_ppo_stats.txt"


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
    """Save VecNormalize alongside model checkpoints."""
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


class EntCoefDecayCallback(BaseCallback):
    """Linear decay ent_coef: 0.02 → 0.001 over total_steps."""
    def __init__(self, initial=0.02, final=0.001, total_steps=40_000_000):
        super().__init__()
        self.initial = initial
        self.final = final
        self.total = total_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total, 1.0)
        self.model.ent_coef = self.initial + progress * (self.final - self.initial)
        return True


# ===================== Eval Callback =====================

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
    print(f"  🔭 Wide PPO: VecFrameStack={N_STACK} ({N_STACK*4/60:.1f}s) + Death Penalty")
    print(f"  Death: {DeathPenaltyWrapper.DEATH_PENALTY} | Survival: +{DeathPenaltyWrapper.SURVIVAL_BONUS}")
    print(f"  clip_reward=100 | N_ENVS={N_ENVS} | n_steps=512")
    print(f"  Steps: {TOTAL_TIMESTEPS:,} ~32-35h")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

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
    print(f"[OK] VecFrameStack({N_STACK}) + VecNormalize (clip=100)")

    model = PPO('CnnPolicy', train_env,
                n_steps=512, batch_size=512, n_epochs=4,
                gamma=0.99, gae_lambda=0.99, clip_range=0.1,
                ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5,
                learning_rate=lambda p: 2.5e-4 * p,  # linear: 2.5e-4 -> 0
                verbose=1)
    print(f"[OK] PPO + NatureCNN (input: {N_STACK}x84x84) gae_lambda=0.99")

    eval_cb = EvalCallback(eval_env_fn=_make_eval_env())
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_wide_ppo"
    )
    vecnorm_cb = VecNormCheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="pacman_wide_ppo"
    )
    ent_cb = EntCoefDecayCallback(
        initial=0.02, final=0.001, total_steps=TOTAL_TIMESTEPS
    )
    print(f"[OK] VecNorm checkpoint + ent_coef decay (0.02→0.001)")

    print(f"\n開始訓練 {TOTAL_TIMESTEPS:,} 步...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=CallbackList([eval_cb, checkpoint_cb, vecnorm_cb, ent_cb]))
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

    model.save("models/MsPacman-v5_wide_ppo.zip")
    train_env.save("models/vecnormalize_wide_ppo.pkl")
    train_env.close()


if __name__ == '__main__':
    main()
