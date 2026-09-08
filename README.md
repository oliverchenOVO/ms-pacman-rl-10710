# Ms. Pac-Man 強化學習：從參考程式到 10,710 分

這個資料庫記錄我如何從課程提供的 Ms. Pac-Man 強化學習參考程式出發，逐步調整獎勵函數、特徵提取網路、PPO 超參數與訓練流程，最後在保存的遊戲影片中取得 **10,710 分**。

> 目前建議先以 Private GitHub repository 保存。`baseline/` 的原始參考程式未附授權條款；在取得老師或原作者允許前，不建議將包含該目錄的 repository 改成 Public。

## 成果對照

| 版本 | 保存影片分數 | 說明 |
|---|---:|---|
| 原始測試 | 430 | 最初用參考架構進行的單局測試 |
| 改良版本 | **10,710** | 修改訓練方法後保存的最高分影片 |

### 原始版本：430 分

<video controls preload="metadata" src="/oliverchenOVO/ms-pacman-rl-10710/raw/refs/heads/main/media/baseline-test.mp4"></video>

原始參考程式保存在 [`baseline/`](baseline/README.md)。它提供基本的遊戲環境、PPO 使用範例、自訂 CNN，以及可供學生修改的 reward/observation wrapper。

### 改良版本：10,710 分

<video controls preload="metadata" src="/oliverchenOVO/ms-pacman-rl-10710/raw/refs/heads/main/media/improved-10710.mp4"></video>

改良程式保存在 [`improved/`](improved/README.md)，最终使用的模型为 `improved/models/MsPacman-v5_avg10500.zip`。

## 我修改了什麼

1. **重新設計獎勵函數**：保留遊戲分數比例，讓連續吃鬼的高價值能反映在訓練訊號中，並加入死亡、長時間無進展與存活訊號。
2. **加深 CNN**：增加通道數、加入 Residual Blocks，並將特徵維度由 512 提升至 1024。
3. **調整 PPO 參數**：縮短 rollout、增加 batch size、降低 clip range，並加入 value coefficient 與 gradient clipping。
4. **改善 Atari 訓練流程**：使用灰階 84×84、frame stack、並行環境、最佳模型保存、checkpoint、定期評估與續訓。
5. **擴充實驗路線**：測試 RAM-based reward、死亡懲罰、Wide PPO、LSTM PPO，以及 DQN 等方法。
6. **分離訓練與正式評分**：訓練時可使用 shaped reward；播放與計分時使用原始遊戲 reward。

完整對照請見 [`docs/changes.md`](docs/changes.md)。

## 成果應如何解讀

10,710 分由 `media/improved-10710.mp4` 的結束畫面確認。訓練紀錄中曾出現更高的單局數值，但目前沒有保留相對應的完整影片，因此本資料庫只把 **10,710** 列為可由影片直接驗證的成果。

這些修改共同構成取得 10,710 分的最終流程；目前沒有逐項消融實驗，因此不主張任何單一修改必然造成特定幅度的提升。

## 快速開始

```powershell
cd improved
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python play_local.py
```

執行者需自行合法取得 Atari ROM。這個資料庫不包含 ROM。模型播放預設不會向課程伺服器送出成績。

## 資料庫結構

```text
baseline/       原始參考程式快照
improved/       修改後的訓練、續訓、評估及播放程式
media/          前後對照影片與預覽圖
results/        經影片核對的成績摘要
docs/           修改內容與權利說明
```

## 使用與權利提醒

- 本資料庫不提供 Ms. Pac-Man ROM、遊戲素材或 Atari 執行檔。
- 影片包含 Ms. Pac-Man 遊戲畫面，權利屬原權利人；目前僅供私人學習成果整理。
- `baseline/` 為課程參考程式快照，來源授權尚待確認。
- 自行修改的程式在確認基礎程式授權前，也不另行宣告開源授權。

詳見 [`docs/RIGHTS_AND_PRIVACY.md`](docs/RIGHTS_AND_PRIVACY.md)。
