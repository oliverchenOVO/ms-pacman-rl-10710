import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import cv2
import numpy as np
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
SERVER_URL = 'http://163.13.136.86:5000'  # 伺服器地址

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
        os.makedirs("recordings", exist_ok=True)
        
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
        創建並返回一個環境
        Args:
            use_wrappers: 是否使用自定義包裝器（訓練時使用，遊玩時不使用）
        """
        try:
            env = gym.make(self._env_id, render_mode="rgb_array")
        except Exception as e:
            raise RuntimeError(f"Failed to create environment: {str(e)}")
            
        # 只在訓練時使用自定義包裝器
        if use_wrappers:
            if self._use_custom_reward:
                env = CustomRewardWrapper(env)
            if self._use_custom_obs:
                env = CustomObservationWrapper(env)
                
        return Monitor(env)
        
    def train(self, model_class, model_params, total_timesteps=100000, force_train=False, 
              continue_from=None, use_custom_policy=False):
        """訓練模型 (略過註解保持原樣)"""
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
            
        default_model_path = f"models/{self._env_id.split('/')[-1]}.zip"
        model_path = continue_from if continue_from else default_model_path
        
        # 驗證模型完整性
        if os.path.exists(model_path) and not force_train:
            if not self._validate_model_integrity(model_path):
                raise ValueError("Model file has been tampered with")
                
            print(f"載入已存在的模型: {model_path}")
            env = self.create_env(use_wrappers=True)
            self._model = model_class.load(model_path, env=env)
            
            if continue_from:
                print(f"繼續訓練模型...")
                self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
                print(f"繼續訓練完成，評估模型中...")
                eval_env = self.create_env(use_wrappers=True)
                mean_reward, std_reward = evaluate_policy(self._model, eval_env, n_eval_episodes=5)
                print(f"評估結果 - 平均獎勵: {mean_reward:.2f} (+/- {std_reward:.2f})")
                self._model.save(model_path)
                print(f"更新後的模型已保存至: {model_path}")
                eval_env.close()
                
            return self._model
            
        print("開始訓練新模型...")
        env = self.create_env(use_wrappers=True)
        
        if use_custom_policy:
            policy_kwargs = {
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': {'features_dim': 512}
            }
            model_params['policy_kwargs'] = policy_kwargs
            
        self._model = model_class('CnnPolicy', env, **model_params)
        
        # 訓練模型
        eval_env = self.create_env(use_wrappers=True)
        self._model.learn(total_timesteps=total_timesteps)
        mean_reward, std_reward = evaluate_policy(self._model, eval_env, n_eval_episodes=5)
        print(f"評估結果 - 平均獎勵: {mean_reward:.2f} (+/- {std_reward:.2f})")
        
        # 保存模型
        self._model.save(default_model_path)
        print(f"模型已保存至: {default_model_path}")
        
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
        使用訓練好的模型玩遊戲並提交結果
        """
        if not isinstance(player_name, str) or not player_name.strip():
            raise ValueError("Invalid player name")
            
        if self._model is None:
            raise ValueError("請先訓練或載入模型")
            
        # 遊玩時使用原始環境，不使用自定義包裝器
        env = self.create_env(use_wrappers=False)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        frames = []
        step_count = 0
        self._game_data = []  # 重置遊戲數據
        
        print(f"開始遊玩：{self._env_id}")

        while not done:
            action, _states = self._model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 記錄基本遊戲數據（不用管 skip）
            step_data = {
                'step': step_count,
                'action': int(action),
                'reward': float(reward),
                'done': terminated or truncated
            }
            self._game_data.append(step_data)
            
            frame = env.render()
            frames.append(frame)
            total_reward += reward
            done = terminated or truncated
            step_count += 1
            
            if show_game:
                display_frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    None, fx=window_scale, fy=window_scale,
                    interpolation=cv2.INTER_NEAREST
                )
                cv2.imshow('Game', display_frame)
                cv2.waitKey(1)
            
        if show_game:
            cv2.destroyAllWindows()
            
        
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