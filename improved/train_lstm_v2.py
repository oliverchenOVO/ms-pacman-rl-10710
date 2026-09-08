"""
train_lstm_v2.py — RecurrentPPO + LSTM(512×2) + RND Curiosity
==============================================================
Root causes fixed vs train_lstm_warm.py:

  問題                     修正
  ─────────────────────────────────────────────────────────
  LSTM hidden=256 記憶瓶頸  → hidden=512, layers=2, critic_lstm=True
  無探索驅動 (EV=0.97 平台) → RND intrinsic reward (β=0.1)
  BPTT 窗口偏短             → n_steps 512→1024
  過早收斂                  → ent_coef 0.01→0.02
  LR 平台震盪               → cosine decay 1e-4→3e-5

Expected ceiling: avg 16,000–22,000 / best 可破 25,000

RND 整合方式:
  - 繼承 RecurrentPPO，monkey-patch collect_rollouts 捕捉 last_values
  - _on_rollout_end 時擴增 rollout_buffer.rewards，重算 advantage
  - RND predictor 每 rollout 更新一次
  - intrinsic reward 用 Welford online normalization 穩定化

Warm-start:
  - 從 best_model_lstm_warm.zip 萃取 CNN 權重
  - LSTM 全新初始化（size 改變無法繼承）
"""

import sys, io, os, time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
)
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, WarpFrame
)
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium import Wrapper
import ale_py

gym.register_envs(ale_py)


# ========================= CONFIG =========================

ENV_ID           = "ALE/MsPacman-v5"
N_ENVS           = 8
N_STACK          = 4
OBS_SHAPE        = (N_STACK, 84, 84)       # (C, H, W) after VecFrameStack
TOTAL_TIMESTEPS  = 30_000_000              # 多 10M 給 RND 探索
EVAL_FREQ        = 100_000 // N_ENVS
N_EVAL_EPISODES  = 10
CHECKPOINT_FREQ  = 1_000_000 // N_ENVS
TARGET_SCORE     = 20_000

SOURCE_MODEL     = "models/best/best_model_lstm_warm.zip"

EVAL_LOG         = "logs/lstm_v2_evals.csv"
PID_FILE         = "logs/lstm_v2_pid.txt"
CHECKPOINT_DIR   = "models/checkpoints/lstm_v2"
BEST_MODEL       = "models/best/best_model_lstm_v2.zip"
BEST_VECNORM     = "models/best/best_vecnormalize_lstm_v2.pkl"
STATS_LOG        = "logs/lstm_v2_stats.txt"

# PPO hyperparams
PPO_PARAMS = dict(
    n_steps        = 1024,      # 512→1024: BPTT 窗口加倍
    batch_size     = 512,
    n_epochs       = 4,
    gamma          = 0.99,
    gae_lambda     = 0.99,
    clip_range     = 0.1,
    ent_coef       = 0.02,      # 0.01→0.02: 防止過早收斂
    vf_coef        = 1.5,
    max_grad_norm  = 0.5,
    verbose        = 1,
)

# LSTM architecture (key upgrade)
LSTM_KWARGS = dict(
    lstm_hidden_size  = 512,    # 256→512: 2× 記憶容量
    n_lstm_layers     = 2,      # 1→2: 分層時間處理
    enable_critic_lstm = True,  # False→True: critic 也有記憶
)

# RND config
RND_BETA      = 0.1    # intrinsic reward 縮放因子
RND_DIM       = 256    # embedding 維度
RND_CNN_FLAT  = 3136   # 64×7×7 after 3-layer CNN on 84×84

# LR cosine schedule: 1e-4 → 3e-5 over TOTAL_TIMESTEPS
def _cosine_lr(progress_remaining: float) -> float:
    """progress_remaining: 1.0 (start) → 0.0 (end)"""
    lr_start, lr_end = 1e-4, 3e-5
    cos = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
    return lr_end + (lr_start - lr_end) * (1.0 - cos)


# ========================= RND MODULE =========================

class RNDModule(nn.Module):
    """
    Random Network Distillation:
      target    : fixed random CNN+MLP → 256-dim embedding
      predictor : trainable CNN+MLP → 256-dim embedding
      intrinsic reward = MSE(predictor(obs), target(obs))

    新狀態 → target 和 predictor 差距大 → 高 intrinsic reward
    熟悉狀態 → predictor 已學會 target → 低 intrinsic reward
    """

    def __init__(self, obs_shape: tuple = OBS_SHAPE, out_dim: int = RND_DIM):
        super().__init__()
        c = obs_shape[0]

        def _cnn_backbone():
            return nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(),
                nn.Flatten(),
            )

        # Target: 較小，固定隨機
        self.target = nn.Sequential(
            _cnn_backbone(),
            nn.Linear(RND_CNN_FLAT, 512), nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        # Predictor: 稍大（多一層），可訓練
        self.predictor = nn.Sequential(
            _cnn_backbone(),
            nn.Linear(RND_CNN_FLAT, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        # 凍結 target
        for p in self.target.parameters():
            p.requires_grad = False

        # Welford online stats for intrinsic reward normalization
        self._ir_count = 1e-6
        self._ir_mean  = 0.0
        self._ir_M2    = 1.0

    def _update_stats(self, x: np.ndarray):
        for v in x.flat:
            self._ir_count += 1
            delta = v - self._ir_mean
            self._ir_mean += delta / self._ir_count
            self._ir_M2   += delta * (v - self._ir_mean)

    @property
    def _ir_std(self) -> float:
        return max(np.sqrt(self._ir_M2 / self._ir_count), 1e-8)

    def compute_intrinsic(self, obs_t: th.Tensor) -> th.Tensor:
        """
        obs_t: (N, C, H, W) float32 in [0, 1]
        returns: (N,) intrinsic reward tensor (unnormalized)
        """
        with th.no_grad():
            tgt = self.target(obs_t)
        pred = self.predictor(obs_t)
        return (pred - tgt.detach()).pow(2).mean(dim=-1)

    def update_predictor(self, obs_t: th.Tensor, optimizer: th.optim.Optimizer,
                         max_samples: int = 1024):
        """Update predictor on a random subset of rollout observations."""
        idx = th.randperm(len(obs_t))[:max_samples]
        batch = obs_t[idx]

        with th.no_grad():
            tgt = self.target(batch)
        pred = self.predictor(batch)
        loss = F.mse_loss(pred, tgt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        optimizer.step()
        return loss.item()


# ========================= PPO SUBCLASS (last_values capture) =========================

class RNDRecurrentPPO(RecurrentPPO):
    """
    RecurrentPPO 子類：monkey-patch collect_rollouts 以捕捉
    compute_returns_and_advantage 的 last_values / dones 參數。

    RNDCallback 在 _on_rollout_end 中：
      1. 把 intrinsic reward 加進 rollout_buffer.rewards
      2. 用捕捉到的 last_values/dones 重算 advantage
    """

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        orig_fn = rollout_buffer.compute_returns_and_advantage
        captured: dict = {}

        def _patched(last_values, dones):
            captured['last_values'] = last_values.clone()
            captured['dones']       = np.array(dones, copy=True)
            return orig_fn(last_values, dones)

        rollout_buffer.compute_returns_and_advantage = _patched
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        rollout_buffer.compute_returns_and_advantage = orig_fn   # 還原
        self._rnd_captured = captured
        return result


# ========================= REWARD WRAPPER =========================

class EnhancedRewardWrapper(Wrapper):
    """
    vs train_lstm_warm.py 的 DeathPenaltyWrapper:
    - 保留 death=-50, survival=+0.005
    - 新增 idle penalty（鼓勵積極移動）
    - 新增 power pellet strategic bonus（吃 pellet 時鬼越多獎越多）
    """

    DEATH_PENALTY   = -50.0
    SURVIVAL_BONUS  = 0.005
    IDLE_PENALTY    = -0.02
    IDLE_THRESHOLD  = 60

    def __init__(self, env):
        super().__init__(env)
        self._last_lives        = None
        self._steps_no_reward   = 0
        self._combo             = 0     # consecutive ghosts eaten this power pellet
        self._power_steps       = 999   # steps since last power pellet

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_lives      = self.unwrapped.ale.lives()
        self._steps_no_reward = 0
        self._combo           = 0
        self._power_steps     = 999
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward

        self._power_steps += 1

        # Power pellet eaten
        if reward == 50:
            self._power_steps = 0
            self._combo       = 0

        # Ghost eaten (within frightened window ≈ 38 decision steps with skip=4)
        elif reward >= 200 and self._power_steps <= 38:
            self._combo += 1
            # Extra combo bonus: 3rd ghost +15, 4th ghost +60
            if self._combo == 3:
                shaped += 15.0
            elif self._combo >= 4:
                shaped += 60.0

        # Power pellet window expired → reset combo
        if self._power_steps > 38:
            self._combo = 0

        # Idle tracking
        if reward > 0:
            self._steps_no_reward = 0
        else:
            self._steps_no_reward += 1

        # Death penalty
        cur_lives = self.unwrapped.ale.lives()
        if self._last_lives is not None and cur_lives < self._last_lives:
            shaped += self.DEATH_PENALTY
        self._last_lives = cur_lives

        # Survival + idle
        if not (terminated or truncated):
            shaped += self.SURVIVAL_BONUS
        if self._steps_no_reward > self.IDLE_THRESHOLD:
            shaped += self.IDLE_PENALTY

        return obs, shaped, terminated, truncated, info


# ========================= ENV FACTORIES =========================

def _make_train_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = Monitor(env)
        env = EnhancedRewardWrapper(env)
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


# ========================= RND CALLBACK =========================

class RNDCallback(BaseCallback):
    """
    _on_rollout_end 執行三件事：
      1. 把 rollout_buffer 的 observations 轉成 tensor → 算 intrinsic reward
      2. 把 intrinsic reward 加進 buffer.rewards（正規化後縮放 β）
      3. 用捕捉到的 last_values/dones 重算 GAE advantage
      4. 更新 RND predictor
    """

    def __init__(self, rnd_module: RNDModule, beta: float = RND_BETA):
        super().__init__()
        self.rnd   = rnd_module
        self.beta  = beta
        self._dev  = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.rnd.to(self._dev)
        self._optimizer = th.optim.Adam(
            list(self.rnd.predictor.parameters()),
            lr=1e-4
        )
        self._total_updates = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        buf      = self.model.rollout_buffer
        captured = getattr(self.model, '_rnd_captured', {})
        if not captured:
            return

        # ── Step 1: 把 observations 轉成 float tensor ──────────────────
        obs_np = buf.observations                         # (T, N, C, H, W) uint8 or float32
        T, N   = obs_np.shape[0], obs_np.shape[1]

        obs_flat = obs_np.reshape(T * N, *OBS_SHAPE)
        obs_t    = th.as_tensor(obs_flat, dtype=th.float32, device=self._dev)

        # 確保像素歸一化到 [0, 1]
        if obs_t.max() > 1.5:
            obs_t = obs_t / 255.0

        # ── Step 2: 計算 intrinsic reward ─────────────────────────────
        CHUNK = 512
        ir_list = []
        self.rnd.eval()
        with th.no_grad():
            for i in range(0, len(obs_t), CHUNK):
                ir_chunk = self.rnd.compute_intrinsic(obs_t[i:i + CHUNK])
                ir_list.append(ir_chunk.cpu().numpy())

        ir_np = np.concatenate(ir_list)                  # (T*N,)

        # Welford normalize
        self.rnd._update_stats(ir_np)
        ir_norm  = ir_np / self.rnd._ir_std
        ir_2d    = ir_norm.reshape(T, N)                 # (T, N)

        # ── Step 3: 擴增 rewards，重算 advantage ──────────────────────
        buf.rewards += self.beta * ir_2d

        last_values = captured['last_values'].to(self.model.device)
        dones       = captured['dones']
        buf.compute_returns_and_advantage(last_values, dones)

        # ── Step 4: 更新 RND predictor ────────────────────────────────
        self.rnd.train()
        loss = self.rnd.update_predictor(obs_t, self._optimizer, max_samples=1024)
        self._total_updates += 1

        if self._total_updates % 50 == 0:
            print(f"  [RND] updates={self._total_updates} | "
                  f"ir_mean={self.rnd._ir_mean:.4f} | "
                  f"ir_std={self.rnd._ir_std:.4f} | "
                  f"pred_loss={loss:.4f}")


# ========================= EVAL CALLBACK =========================

class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fn):
        super().__init__()
        self.eval_env_fn     = eval_env_fn
        self.best_mean_reward = -np.inf
        self._last_eval_step = 0
        self._ep_lengths     = []
        self._cur_ep_lens    = [0] * N_ENVS
        self._last_fps_step  = 0
        self._last_fps_time  = time.time()
        self._fps            = float('nan')

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        if dones is not None:
            for i in range(N_ENVS):
                self._cur_ep_lens[i] += 1
                if dones[i]:
                    self._ep_lengths.append(self._cur_ep_lens[i])
                    self._cur_ep_lens[i] = 0
            if len(self._ep_lengths) > 400:
                self._ep_lengths = self._ep_lengths[-400:]

        now = time.time()
        if now - self._last_fps_time > 30 and self.num_timesteps > self._last_fps_step:
            elapsed = now - self._last_fps_time
            self._fps = (self.num_timesteps - self._last_fps_step) / elapsed
            self._last_fps_step = self.num_timesteps
            self._last_fps_time = now

        if (self.num_timesteps - self._last_eval_step) >= EVAL_FREQ:
            self._last_eval_step = self.num_timesteps
            self._run_eval()

        return True

    def _run_eval(self):
        eval_env = DummyVecEnv([self.eval_env_fn])
        eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=False
        )
        eval_env.close()

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ev = self.model.logger.name_to_value.get('train/explained_variance', float('nan')) \
            if hasattr(self.model, 'logger') and self.model.logger else float('nan')
        ep_len = np.mean(self._ep_lengths[-50:]) if self._ep_lengths else float('nan')

        is_best = mean_reward > self.best_mean_reward
        if is_best:
            self.best_mean_reward = mean_reward
            self.model.save(BEST_MODEL)
            self.training_env.save(BEST_VECNORM)
            print(f"\n  新最佳！step={self.num_timesteps:,} mean={mean_reward:.2f}")

        print(f"[Eval] step={self.num_timesteps:>12,} | "
              f"mean={mean_reward:>9.2f} | std={std_reward:>8.2f} | "
              f"best={self.best_mean_reward:>9.2f} | "
              f"ev={ev:.3f} | ep_len={ep_len:.0f} | fps={self._fps:.0f} | {ts}")

        file_exists = os.path.exists(EVAL_LOG)
        with open(EVAL_LOG, 'a', encoding='utf-8') as f:
            if not file_exists:
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

        if mean_reward >= TARGET_SCORE:
            print(f"\n{'='*60}")
            print(f"  達標 {TARGET_SCORE}！")
            print(f"{'='*60}")
            return False
        return True


# ========================= MAIN =========================

def main():
    print("=" * 60)
    print("  RecurrentPPO + LSTM(512×2) + RND Curiosity")
    print(f"  n_steps={PPO_PARAMS['n_steps']} | envs={N_ENVS} | "
          f"lstm={LSTM_KWARGS['lstm_hidden_size']}×{LSTM_KWARGS['n_lstm_layers']}")
    print(f"  RND β={RND_BETA} | ent_coef={PPO_PARAMS['ent_coef']} | "
          f"LR cosine 1e-4→3e-5")
    print(f"  Total steps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # ── Training env ──────────────────────────────────────────────
    env_fns = [_make_train_env() for _ in range(N_ENVS)]
    try:
        train_env = SubprocVecEnv(env_fns)
        print(f"[OK] SubprocVecEnv ×{N_ENVS}")
    except Exception as e:
        print(f"[WARN] SubprocVecEnv 失敗 ({e})，改用 DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    train_env = VecFrameStack(train_env, n_stack=N_STACK)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=200.0, gamma=0.99)
    print("[OK] VecNormalize (reward only, clip=200)")

    # ── RND module ────────────────────────────────────────────────
    rnd_module = RNDModule(obs_shape=OBS_SHAPE, out_dim=RND_DIM)
    print(f"[OK] RND module (dim={RND_DIM}, β={RND_BETA})")

    # ── Model ─────────────────────────────────────────────────────
    model = RNDRecurrentPPO(
        RecurrentActorCriticCnnPolicy,
        train_env,
        learning_rate=_cosine_lr,
        policy_kwargs=dict(**LSTM_KWARGS),
        **PPO_PARAMS,
    )
    print(f"[OK] RNDRecurrentPPO created")
    print(f"     LSTM: hidden={LSTM_KWARGS['lstm_hidden_size']}, "
          f"layers={LSTM_KWARGS['n_lstm_layers']}, "
          f"critic_lstm={LSTM_KWARGS['enable_critic_lstm']}")

    # ── CNN warm-start ─────────────────────────────────────────────
    if os.path.exists(SOURCE_MODEL):
        try:
            # Load source as RecurrentPPO (same CNN architecture)
            src = RecurrentPPO.load(
                SOURCE_MODEL,
                env=DummyVecEnv([_make_eval_env()])
            )
            src_cnn_sd = src.policy.features_extractor.state_dict()
            model.policy.features_extractor.load_state_dict(src_cnn_sd)
            print(f"[OK] CNN warm-start from {SOURCE_MODEL}")
            del src
        except Exception as e:
            print(f"[WARN] CNN warm-start 失敗：{e}，從隨機初始化開始")
    else:
        print(f"[INFO] {SOURCE_MODEL} 不存在，從隨機初始化開始")

    # ── Callbacks ─────────────────────────────────────────────────
    rnd_cb   = RNDCallback(rnd_module, beta=RND_BETA)
    eval_cb  = EvalCallback(eval_env_fn=_make_eval_env())
    ckpt_cb  = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="pacman_lstm_v2"
    )
    callbacks = CallbackList([rnd_cb, eval_cb, ckpt_cb])

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n開始訓練 {TOTAL_TIMESTEPS/1e6:.0f}M 步...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("\n  訓練中斷")
    except Exception as e:
        import traceback
        print(f"\n  訓練錯誤: {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  結束 | {elapsed/3600:.1f}h | best={eval_cb.best_mean_reward:.2f}")
    print(f"  總步數: {model.num_timesteps:,}")
    print(f"{'='*60}")

    model.save("models/MsPacman-v5_lstm_v2.zip")
    train_env.save("models/vecnormalize_lstm_v2.pkl")
    train_env.close()

    if eval_cb.best_mean_reward >= TARGET_SCORE:
        print("  達標！")


if __name__ == '__main__':
    main()
