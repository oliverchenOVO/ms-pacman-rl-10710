# Baseline：課程原始參考程式

此目錄是 `D:\Pacman training\pacman v0.0` 的檔案快照，用來保留修改前的比較基準。

## 內容

- `game_environment(2).py`：遊戲環境、訓練、播放、錄影及課程成績提交流程。
- `custom_policy.py`：三層 CNN 特徵提取器，輸出 512 維特徵。
- `custom_wrappers.py`：示範型 reward wrapper，主要將正獎勵加倍。
- `example_usage.py`：PPO 訓練與續訓範例。
- `example_usage.ipynb`：Notebook 形式的原始範例。
- `requirements.txt`：原始依賴清單。

## 原始測試結果

保存的最初單局測試影片顯示 **430 分**：

[觀看 baseline 測試影片](../media/baseline-test.mp4)

## 注意

此目錄刻意保留參考程式原貌，可能包含已失效的課程伺服器位址。請勿直接執行成績提交功能。原始檔案沒有附 LICENSE；在確認授權前，只应保存在私人 repository 中。
