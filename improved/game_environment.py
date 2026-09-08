import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import cv2
import numpy as np
from collections import deque
import os
import ale_py
import requests
import time
import hashlib
import json
import hmac
from custom_wrappers import CustomRewardWrapper, CustomObservationWrapper
from custom_policy import CustomCNN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

gym.register_envs(ale_py)
# Upload-safe default: local evaluation does not contact the course server.
# Set this explicitly only when you are authorized to submit a score.
SERVER_URL = os.getenv('PACMAN_SCORE_SERVER_URL', '').rstrip('/')


class FrameStack(gym.Wrapper):
    """
    將最近 k 幀沿通道維度堆疊，產生 (H, W, C*k) 形狀的觀察值。
    用於單一環境遊玩模式，行為與 VecFrameStack 一致。
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 使用零填充 + 初始觀察，與 VecFrameStack 的 reset 行為一致
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)


class MaxAndSkipWithRecord(gym.Wrapper):
    """
    與 MaxAndSkipEnv 功能一致（重複動作 + 最後兩幀取最大值），
    但同時會把每一步的中間渲染幀存下來，供視頻錄製使用。
    這樣模型看到的觀察與訓練時一致，但視頻幀數足夠讓伺服器驗證。
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.recorded_frames = []  # 存儲中間渲染幀

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        self.recorded_frames = []

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # 錄製中間幀
            frame = self.env.render()
            if frame is not None:
                self.recorded_frames.append(frame)

            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs

            if terminated or truncated:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.recorded_frames = []
        return self.env.reset(**kwargs)


def _make_atari_env(env_id, is_training=True, use_custom_reward=False, use_custom_obs=False):
    """
    模組級別的工廠函式，用於建立預處理過的 Atari 環境。
    定義在模組級別以確保可被 SubprocVecEnv 序列化（pickle safe）。

    預處理流程:
    1. NoopResetEnv - 開始時隨機執行 no-op 動作
    2. MaxAndSkipEnv - 每個動作重複4幀，取最後2幀最大值
    3. EpisodicLifeEnv - 失去生命視為 episode 結束（僅訓練時）
    4. FireResetEnv - 需要時按 FIRE 開始遊戲
    5. WarpFrame - 縮放至 84x84 灰階
    6. 自定義包裝器（僅訓練時）
    7. Monitor - 追蹤 episode 統計
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")

        # 標準 Atari 預處理
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

        if is_training:
            env = EpisodicLifeEnv(env)

        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = WarpFrame(env)  # 縮放至 84x84 灰階

        # 自定義包裝器（僅訓練時使用）
        if is_training:
            if use_custom_reward:
                env = CustomRewardWrapper(env)
            if use_custom_obs:
                env = CustomObservationWrapper(env)

        env = Monitor(env)
        return env
    return _init


class GameEnvironment:
    def __init__(self, env_id="ALE/MsPacman-v5", use_custom_reward=False, use_custom_obs=False):
        """
        初始化遊戲環境
        Args:
            env_id: 遊戲環境ID
            use_custom_reward: 是否使用自定義獎勵
            use_custom_obs: 是否使用自定義觀察處理
        """
        # Validate environment ID
        if not isinstance(env_id, str) or not env_id.startswith("ALE/"):
            raise ValueError("Invalid environment ID. Must be a valid ALE environment.")

        self._env_id = env_id
        self._use_custom_reward = bool(use_custom_reward)
        self._use_custom_obs = bool(use_custom_obs)
        self._model = None
        self._ensure_directories()
        self._game_data = []  # 用於存儲遊戲過程中的數據
        self._secret_key = "RL_GAME_ENV_2026"  # 用於生成簽名的密鑰

    @property
    def env_id(self):
        return self._env_id

    @property
    def model(self):
        return self._model

    def _ensure_directories(self):
        """確保必要的目錄存在"""
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)
        os.makedirs("models/best", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def _validate_model_integrity(self, model_path):
        """驗證模型文件的完整性"""
        if not os.path.exists(model_path):
            return False

        try:
            with open(model_path, 'rb') as f:
                content = f.read()
            current_hash = hashlib.sha256(content).hexdigest()

            # 如果是新模型，保存其哈希值
            if not hasattr(self, '_model_hash'):
                self._model_hash = current_hash
                return True

            # 驗證模型是否被修改
            return current_hash == self._model_hash
        except Exception:
            return False

    def create_env(self, use_wrappers=True):
        """
        創建單一環境。
        - 訓練時 (use_wrappers=True): 完整 Atari 預處理管線
        - 遊玩時 (use_wrappers=False): 使用 MaxAndSkipWithRecord 保持模型觀察一致，
          同時錄製中間幀供視頻使用
        """
        if use_wrappers:
            # 訓練環境：完整預處理
            make_fn = _make_atari_env(
                self._env_id,
                is_training=True,
                use_custom_reward=self._use_custom_reward,
                use_custom_obs=self._use_custom_obs
            )
            env = make_fn()
        else:
            # 遊玩環境：保留 ALE 預設 frameskip=4，加上 MaxAndSkipWithRecord(skip=4)
            # 模型觀察與訓練一致: frameskip=4 × skip=4 = 每步 16 ALE 幀
            # 每步錄製 4 個中間幀 → video_frames ≈ total_env_frames / 4（與原始一致）
            env = gym.make(self._env_id, render_mode="rgb_array")
            env = MaxAndSkipWithRecord(env, skip=4)  # 錄製 4 個中間幀
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = WarpFrame(env)  # 84x84 灰階
            env = Monitor(env)

        env = FrameStack(env, k=4)
        return env

    def create_vec_env(self, n_envs=8):
        """
        創建向量化訓練環境，支持多進程並行加速訓練。
        使用 SubprocVecEnv 實現真正的並行（自動回退到 DummyVecEnv）。
        觀察格式: (n_envs, 84, 84, 4) → SB3 自動轉置為 (n_envs, 4, 84, 84)
        """
        env_fns = [
            _make_atari_env(
                self._env_id,
                is_training=True,
                use_custom_reward=self._use_custom_reward,
                use_custom_obs=self._use_custom_obs
            )
            for _ in range(n_envs)
        ]

        try:
            env = SubprocVecEnv(env_fns)
            print(f"[OK] 使用 SubprocVecEnv 創建 {n_envs} 個並行環境")
        except Exception as e:
            print(f"[WARN] SubprocVecEnv 失敗 ({e})，回退到 DummyVecEnv")
            env = DummyVecEnv(env_fns)

        env = VecFrameStack(env, n_stack=4)
        return env

    def create_eval_vec_env(self):
        """創建評估用向量化環境（單一環境，無訓練專用包裝器）"""
        env_fn = _make_atari_env(
            self._env_id,
            is_training=False,
            use_custom_reward=False,
            use_custom_obs=False
        )
        env = DummyVecEnv([env_fn])
        env = VecFrameStack(env, n_stack=4)
        return env

    def load_model(self, model_class, model_path="models/best/best_model.zip"):
        """
        載入已保存的模型
        Args:
            model_class: 模型類別（如 PPO）
            model_path: 模型檔案路徑
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        self._model = model_class.load(model_path)
        print(f"模型已載入: {model_path}")
        return self._model

    def train(self, model_class, model_params, total_timesteps=10000000, force_train=False,
              continue_from=None, use_custom_policy=False, n_envs=8):
        """
        訓練模型
        Args:
            model_class: 模型類別（如 PPO）
            model_params: 模型超參數字典
            total_timesteps: 總訓練步數（所有環境合計）
            force_train: 是否強制重新訓練（忽略已存在的模型）
            continue_from: 繼續訓練的模型路徑
            use_custom_policy: 是否使用自定義策略網絡
            n_envs: 並行環境數量（建議 4-16）
        """
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")

        default_model_path = f"models/{self._env_id.split('/')[-1]}.zip"
        model_path = continue_from if continue_from else default_model_path

        # 設置訓練回調
        checkpoint_callback = CheckpointCallback(
            save_freq=max(500_000 // n_envs, 1),  # 每 50 萬步保存一次
            save_path='./models/checkpoints/',
            name_prefix='pacman'
        )

        eval_env = self.create_eval_vec_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/best/',
            log_path='./logs/',
            eval_freq=max(100_000 // n_envs, 1),  # 每 10 萬步評估一次
            n_eval_episodes=10,
            deterministic=False
        )

        callbacks = CallbackList([checkpoint_callback, eval_callback])

        # 驗證模型完整性
        if os.path.exists(model_path) and not force_train:
            if not self._validate_model_integrity(model_path):
                raise ValueError("Model file has been tampered with")

            print(f"載入已存在的模型: {model_path}")
            env = self.create_vec_env(n_envs=n_envs)
            self._model = model_class.load(model_path, env=env)

            if continue_from:
                print(f"繼續訓練模型 {total_timesteps:,} 步...")
                self._model.learn(
                    total_timesteps=total_timesteps,
                    reset_num_timesteps=False,
                    callback=callbacks
                )
                print(f"繼續訓練完成，評估模型中...")
                mean_reward, std_reward = evaluate_policy(
                    self._model, eval_env, n_eval_episodes=20
                )
                print(f"評估結果 - 平均獎勵: {mean_reward:.2f} (+/- {std_reward:.2f})")
                self._model.save(model_path)
                print(f"更新後的模型已保存至: {model_path}")

            env.close()
            eval_env.close()
            return self._model

        print(f"開始訓練新模型（{total_timesteps:,} 步，{n_envs} 個並行環境）...")
        env = self.create_vec_env(n_envs=n_envs)

        if use_custom_policy:
            policy_kwargs = {
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': {'features_dim': 512}
            }
            model_params['policy_kwargs'] = policy_kwargs

        self._model = model_class('CnnPolicy', env, **model_params)

        # 訓練模型（含定期保存檢查點和自動評估）
        self._model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # 最終評估
        mean_reward, std_reward = evaluate_policy(
            self._model, eval_env, n_eval_episodes=20
        )
        print(f"最終評估結果 - 平均獎勵: {mean_reward:.2f} (+/- {std_reward:.2f})")

        # 保存最終模型
        self._model.save(default_model_path)
        print(f"最終模型已保存至: {default_model_path}")

        env.close()
        eval_env.close()
        return self._model

    def _generate_game_signature(self, total_reward, total_frames, game_data):
        """
        生成遊戲數據的簽名，現在包含了真實的 total_frames 以確保物理時間不可竄改
        """
        data = {
            'total_reward': float(total_reward),
            'total_frames': int(total_frames),  # 新增：真實物理總幀數
            'game_data': game_data,
            'timestamp': int(time.time())
        }
        # 排序 key 確保 json 字串一致性
        data_str = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            self._secret_key.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature, data

    def play(self, player_name, student_id, show_game=True, window_scale=4.0):
        """
        使用訓練好的模型玩遊戲並提交結果。
        使用原始環境（無自定義包裝器），但保留標準 Atari 預處理以確保與訓練環境相容。
        """
        if not isinstance(player_name, str) or not player_name.strip():
            raise ValueError("Invalid player name")

        if self._model is None:
            raise ValueError("請先訓練或載入模型")

        # 遊玩時使用原始環境，不使用自定義包裝器，但保持相同的視覺預處理
        env = self.create_env(use_wrappers=False)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        frames = []
        step_count = 0
        self._game_data = []  # 重置遊戲數據

        print(f"開始遊玩：{self._env_id}")

        # 找到 MaxAndSkipWithRecord 包裝器以取得中間幀
        skip_wrapper = None
        check_env = env
        while hasattr(check_env, 'env'):
            if isinstance(check_env, MaxAndSkipWithRecord):
                skip_wrapper = check_env
                break
            check_env = check_env.env

        while not done:
            # 轉置 HWC → CHW 以匹配模型期望的格式
            obs_chw = np.transpose(obs, (2, 0, 1))
            action, _states = self._model.predict(obs_chw, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 記錄基本遊戲數據
            step_data = {
                'step': step_count,
                'action': int(action),
                'reward': float(reward),
                'done': terminated or truncated
            }
            self._game_data.append(step_data)

            # 收集視頻幀：優先使用 MaxAndSkipWithRecord 的中間幀
            if skip_wrapper is not None and skip_wrapper.recorded_frames:
                frames.extend(skip_wrapper.recorded_frames)
            else:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            total_reward += reward
            done = terminated or truncated
            step_count += 1

            if show_game:
                try:
                    display_frame = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        None, fx=window_scale, fy=window_scale,
                        interpolation=cv2.INTER_NEAREST
                    )
                    cv2.imshow('Game', display_frame)
                    cv2.waitKey(1)
                except cv2.error:
                    # opencv-python-headless 不支援顯示視窗，自動關閉顯示
                    show_game = False
                    print("(opencv-headless: 略過即時顯示，繼續錄影和提交)")

        if show_game:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        # 必須在 env.close() 之前獲取
        total_env_frames = env.unwrapped.ale.getEpisodeFrameNumber()

        env.close()
        print(f"遊戲結束！總分：{total_reward}，決策步數：{step_count}，真實物理幀數：{total_env_frames}")

        # 生成遊戲數據簽名，傳入真實物理幀數，回傳的 signed_package 已包含所有資料
        signature, signed_package = self._generate_game_signature(total_reward, total_env_frames, self._game_data)

        # 保存視頻並提交結果
        video_path = self._save_video(frames, player_name)
        if video_path:
            self._submit_score(player_name, student_id, total_reward, video_path, signature, signed_package)
        else:
            print("由於視頻保存失敗，無法提交分數")
        return total_reward

    def _validate_video(self, video_path):
        """驗證視頻文件的完整性"""
        if not os.path.exists(video_path):
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                return False

            cap.release()
            return True
        except Exception:
            return False

    def _save_video(self, frames, player_name):
        """保存遊戲視頻"""
        if not frames:
            print("No frames to save")
            return None

        video_path = f'recordings/{player_name}_gameplay.mp4'
        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        if not out.isOpened():
            print("嘗試使用其他編碼器...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_path = f'recordings/{player_name}_gameplay.avi'
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

            if not out.isOpened():
                print("無法創建視頻文件")
                return None

        try:
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()

            if self._validate_video(video_path):
                print(f"視頻成功保存到: {video_path}")
                return video_path
            else:
                print("視頻保存失敗或文件損壞")
                return None

        except Exception as e:
            print(f"保存視頻時發生錯誤: {str(e)}")
            out.release()
            return None

    def _submit_score(self, player, student_id, score, video_path, signature, signed_package):
        """
        提交分數和視頻到伺服器
        """
        if not SERVER_URL:
            print("未設定 PACMAN_SCORE_SERVER_URL；略過成績提交。")
            return

        url = f'{SERVER_URL}/submit_score'
        try:
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
                # 將包含 total_frames 且被加密過資料包整包轉為 JSON 送出
                data = {
                    'player': player,
                    'student_id': student_id,
                    'score': score,
                    'signature': signature,
                    'game_data': json.dumps(signed_package)
                }
                response = requests.post(url, data=data, files=files)

            if response.status_code == 200:
                print("分數和影片提交成功！")
            else:
                print("提交失敗:", response.json().get('message', '未知錯誤'))
        except Exception as e:
            print("無法連接到伺服器:", e)
