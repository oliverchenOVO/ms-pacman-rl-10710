"""
train_dqn.py — Double Dueling DQN + PER + avg10500 warm-start
================================================================
Double DQN, Dueling architecture, Prioritized Experience Replay.
NatureCNN warm-started from avg10500. 10M steps.
"""
import sys, io, os, time, random
from datetime import datetime
from collections import deque
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, WarpFrame
import ale_py
gym.register_envs(ale_py)


# ==================== Death Penalty Wrapper ====================

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


# ==================== Dueling Q-Network ====================

class DuelingQNetwork(nn.Module):
    """NatureCNN backbone + dueling heads (V and A streams)."""
    def __init__(self, n_actions, n_input_channels=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_shared = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.fc_value = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.fc_advantage = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        x = x / 255.0
        f = self.cnn(x)
        s = self.fc_shared(f)
        v = self.fc_value(s)
        a = self.fc_advantage(s)
        return v + a - a.mean(dim=1, keepdim=True)


# ==================== Prioritized Replay Buffer ====================

class SumTree:
    """Binary tree for O(log n) weighted sampling."""
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def total(self):
        return self.tree[1]

    def add(self, priority):
        idx = self.pos + self.capacity
        self.tree[idx] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._propagate(idx)

    def update(self, idx, priority):
        idx += self.capacity
        self.tree[idx] = priority
        self._propagate(idx)

    def _propagate(self, idx):
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def sample(self, value):
        """Find leaf index for given cumulative value."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=3e-7,
                 eps=1e-6, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.n_step = n_step
        self.gamma = gamma

        self.tree = SumTree(capacity)
        self.buffer = [None] * capacity
        self.n_step_buffer = deque(maxlen=n_step)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def _get_n_step(self, reward, next_state, done):
        """Compute n-step return and final state."""
        self.n_step_buffer.append((reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return None
        ret, ns, d = 0.0, self.n_step_buffer[0][1], False
        for i, (r, s_, dn) in enumerate(self.n_step_buffer):
            ret += (self.gamma ** i) * r
            if dn:
                ns = s_
                d = True
                break
            ns = s_
        return ret, ns, d

    def push(self, state, action, reward, next_state, done):
        n_step_result = self._get_n_step(reward, next_state, done)
        if n_step_result is None:
            return
        n_reward, n_next_state, n_done = n_step_result

        self.buffer[self.pos] = (state, action, n_reward, n_next_state, n_done)
        self.tree.add(self.max_priority ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        segment = self.tree.total() / batch_size
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        batch = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = random.uniform(a, b)
            idx = self.tree.sample(value)
            # SumTree capacity is rounded to power-of-2; wrap to buffer capacity
            idx = idx % self.capacity
            indices[i] = idx
            prob = self.tree.tree[idx + self.tree.capacity] / self.tree.total()
            weights[i] = (prob * self.size) ** (-self.beta) if prob > 0 else 1.0
            batch.append(self.buffer[idx])

        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32), np.array(next_states),
                np.array(dones, dtype=np.float32), indices, weights)

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size


# ==================== Environment Helpers ====================

class FrameStack:
    """4-frame stack, outputs (4,84,84) as expected by NatureCNN."""
    def __init__(self, env, n_stack=4):
        self.env = env
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)

    def reset(self):
        obs, info = self.env.reset()
        obs = np.squeeze(obs, axis=-1)  # (84,84,1) → (84,84)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.squeeze(obs, axis=-1)
        self.frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self):
        return np.stack(list(self.frames), axis=0)  # (4,84,84)


def make_env():
    def _init():
        env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = DeathPenaltyWrapper(env)
        return env
    return _init


# ==================== DQN Agent ====================

class DQNAgent:
    def __init__(self, n_actions, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_actions = n_actions
        self.q_online = DuelingQNetwork(n_actions).to(device)
        self.q_target = DuelingQNetwork(n_actions).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=1e-4)
        self.memory = PrioritizedReplayBuffer(capacity=200000, alpha=0.6, beta=0.4,
                                              n_step=3, gamma=0.99)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 250000
        self.epsilon_mid = 0.05
        self.epsilon_decay_mid = 3000000
        self.batch_size = 64
        self.gamma = 0.99
        self.target_update = 8000
        self.steps = 0
        self.train_start = 10000
        self.train_freq = 4

    def warm_start_cnn(self, ppo_path):
        ppo = PPO.load(ppo_path)
        src = ppo.policy.features_extractor.state_dict()
        # SB3 NatureCNN uses 'cnn.0.weight' keys; our Sequential needs '0.weight'
        fixed = {}
        for k, v in src.items():
            if k.startswith('cnn.'):
                fixed[k[4:]] = v  # strip 'cnn.' prefix
        self.q_online.cnn.load_state_dict(fixed)
        self.q_target.cnn.load_state_dict(fixed)
        print(f"[OK] CNN warm-started from {ppo_path}")

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_online(s).argmax(dim=1).item()

    def update_epsilon(self):
        if self.steps < self.epsilon_decay:
            self.epsilon = 1.0 - (1.0 - self.epsilon_mid) * self.steps / self.epsilon_decay
        elif self.steps < self.epsilon_decay_mid:
            self.epsilon = self.epsilon_mid - (self.epsilon_mid - self.epsilon_min) * \
                            (self.steps - self.epsilon_decay) / (self.epsilon_decay_mid - self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_min

    def train_step(self):
        if len(self.memory) < self.train_start:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Double DQN: online picks action, target evaluates
        with torch.no_grad():
            next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.q_target(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.q_online(states).gather(1, actions)
        td_errors = (target - current_q).detach().cpu().numpy().flatten()
        loss = (weights * F.smooth_l1_loss(current_q, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        if self.steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({'q_online': self.q_online.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'steps': self.steps,
                    'epsilon': self.epsilon}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(ckpt['q_online'])
        self.q_target.load_state_dict(ckpt['q_online'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps = ckpt['steps']
        self.epsilon = ckpt['epsilon']


# ==================== Evaluation ====================

def evaluate(agent, env_fn, n_episodes=10):
    env = FrameStack(env_fn())
    scores = []
    ep_lengths = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_score, ep_len = 0, 0
        while not done:
            action = agent.act(state, training=False)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_score += reward
            ep_len += 1
        scores.append(ep_score)
        ep_lengths.append(ep_len)
    env.env.close()
    return np.mean(scores), np.std(scores), np.mean(ep_lengths)


# ==================== Main ====================

def main():
    TOTAL_STEPS = 10_000_000
    EVAL_FREQ = 50000
    CHECKPOINT_FREQ = 500000
    TARGET_SCORE = 22000
    EVAL_LOG = "logs/dqn_scratch_evals.csv"
    STATS_LOG = "logs/dqn_scratch_stats.txt"
    MODEL_DIR = "models/checkpoints/dqn"
    BEST_MODEL = "models/best/best_model_dqn.pt"

    os.makedirs("logs", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    print("=" * 60)
    print("  🧠 Double Dueling DQN + PER (from scratch)")
    print(f"  Replay: 200K | n-step: 3 | PER α=0.6 β=0.4→1")
    print(f"  ε: 1.0→0.05 over 250K → 0.01 over 3M")
    print(f"  Target update: 8K | Batch: 64 | LR: 1e-4")
    print(f"  Total: {TOTAL_STEPS/1e6:.0f}M steps")
    print("=" * 60)

    env_fn = make_env()
    env = FrameStack(env_fn())
    n_actions = env.env.action_space.n
    agent = DQNAgent(n_actions)

    best_mean = -np.inf
    state, _ = env.reset()
    episode_reward = 0
    episode_len = 0
    losses = []

    eval_log_header = not os.path.exists(EVAL_LOG)
    if eval_log_header:
        with open(EVAL_LOG, 'w') as f:
            f.write("timestamp,step,mean_reward,std_reward,ep_len\n")

    start_time = time.time()
    print(f"\n開始訓練 {TOTAL_STEPS/1e6:.0f}M 步...\n")

    for step in range(1, TOTAL_STEPS + 1):
        agent.steps = step
        agent.update_epsilon()

        action = agent.act(state, training=True)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        episode_len += 1

        # Train every train_freq steps (reduces GPU idle, keeps fps high)
        if step % agent.train_freq == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        # Heartbeat — prevents background process timeout
        if step % 5000 == 0:
            print(f'⏱ step={step:,} fps≈{step/(time.time()-start_time):.0f}', flush=True)

        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_len = 0

        # Eval
        if step % EVAL_FREQ == 0:
            mean_r, std_r, mean_ep = evaluate(agent, make_env())
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg_loss = np.mean(losses[-100:]) if losses else 0
            elapsed = (time.time() - start_time) / 3600

            with open(EVAL_LOG, 'a') as f:
                f.write(f"{ts},{step},{mean_r:.2f},{std_r:.2f},{mean_ep:.1f}\n")

            with open(STATS_LOG, 'w') as f:
                f.write(f"step={step}\nmean={mean_r:.1f}\nbest={max(best_mean, mean_r):.1f}\n"
                        f"ep_len={mean_ep:.1f}\nepsilon={agent.epsilon:.4f}\n"
                        f"loss={avg_loss:.4f}\nfps={step/elapsed/3600:.0f}\nts={ts}\n")

            if mean_r > best_mean:
                best_mean = mean_r
                agent.save(BEST_MODEL)
                print(f"\n🏆 新最佳！step={step:,} mean={mean_r:.2f}")

            print(f"📊 step={step:>10,} | mean={mean_r:>10.2f} | best={best_mean:>10.2f} | "
                  f"ep_len={mean_ep:.0f} | ε={agent.epsilon:.3f} | loss={avg_loss:.4f} | {ts}")

            if mean_r >= TARGET_SCORE:
                print(f"\n🎉 達標 {TARGET_SCORE}！")
                break

        # Checkpoint
        if step % CHECKPOINT_FREQ == 0:
            agent.save(f"{MODEL_DIR}/dqn_{step}_steps.pt")
            print(f"📦 checkpoint: {step:,}")

        if step % 100000 == 0:
            elapsed = (time.time() - start_time) / 3600
            fps = step / (elapsed * 3600) if elapsed > 0 else 0
            print(f"⏱ step={step/1e6:.1f}M | {elapsed:.1f}h | fps={fps:.0f} | "
                  f"ε={agent.epsilon:.4f} | mem={len(agent.memory):,}")

    elapsed = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"  結束 | {elapsed:.1f}h | best={best_mean:.2f}")
    print(f"  總步數: {step:,}")
    print(f"{'='*60}")

    agent.save("models/dqn_final.pt")
    env.env.close()


if __name__ == '__main__':
    main()
