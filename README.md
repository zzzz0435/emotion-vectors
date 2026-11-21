# emotional

在開源大模型（Qwen3-8B 等）內部提取情緒方向，用「調整（steering）」改變模型的內部情緒狀態，並測量其道德判斷的變化。
目前覆蓋階段 0–2：環境、情緒向量的提取與驗證、情緒軸的擬合。

## 目錄結構

- `datasets/`：提示描述、故事主題、對照概念、生成的故事
- `scripts/emolab/`：共用模塊（可測試的邏輯）
- `scripts/6.x_*/`：按編號執行的腳本
- `results/`：各腳本的輸出
- `tests/`：pytest 測試（`-m gpu` 需要顯卡）

## 執行順序（階段 0–2）

所有命令都在專案根目錄執行，先啟用虛擬環境（放在 OneDrive 之外：`C:\Users\xgame\.venvs\emotional`）。
先用 `qwen3-8b`（正式）；`qwen3-1.7b` 可用來先試跑。需要顯卡的步驟是 01、03、05、6.9/02、08。

```
# 0. 檢查環境
python scripts/6.0_env/01_check_environment.py --models qwen3-8b

# 1. 詞表：先下載 NRC VAD v2.1（見下方外部資料），再提取 20 種情緒的評分
python scripts/6.3_axes/00_extract_vad.py

# 2. 生成故事，然後過濾
python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b
#    補足循環：若 shortfall.json 還有缺口，就補生成再過濾，重複到核心情緒的缺口為 0 為止。
#    每一輪補生成都用新的隨機種子（42 + 1000 × 起始 replicate），並按 filter_report.csv 裡該情緒
#    的通過率多生成（缺口 ÷ 通過率，通過率下限 0.1，上限 5 倍缺口），所以通常 2–4 輪就夠。
python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b --top-up
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b

# 3. 提取激活、構建向量
python scripts/6.2_emotion_vectors/03_extract_activations.py --model qwen3-8b
python scripts/6.2_emotion_vectors/04_build_vectors.py --model qwen3-8b

# 4. Pain-axis 疼痛向量、驗證、汙染檢查
python scripts/6.2_emotion_vectors/05_pain_axis_vector.py --model qwen3-8b
#    注意：05 會就地修改 vectors.pt；之後若重跑 04，pain_painaxis 會丟失，需要再跑一次 05。
python scripts/6.2_emotion_vectors/06_validate_vectors.py --model qwen3-8b
python scripts/6.2_emotion_vectors/07_contamination.py --model qwen3-8b

# 5. 情緒軸（V/A/D）與殘差向量
python scripts/6.3_axes/01_fit_axes.py --model qwen3-8b
python scripts/6.3_axes/02_residual_vectors.py --model qwen3-8b

# 6. 交叉確認：SAE 最近特徵、與 Pain-axis 論文向量比較
#    6.9/01 會下載約 2 GB 的 SAE 權重，需要約 2–3 GB 記憶體，但不需要顯卡。
python scripts/6.9_crosscheck/01_sae_nearest_features.py --model qwen3-8b
python scripts/6.9_crosscheck/02_compare_pain_axis.py --model qwen3-8b

# 7. 檢查點 1：範例文字（72 段）與中文摘要
python scripts/6.2_emotion_vectors/08_checkpoint_examples.py --model qwen3-8b
```

最後會得到 `results/6.2_emotion_vectors/<模型名>/checkpoint1_summary.md`（給你讀的摘要）和
`checkpoint1_examples.md`（範例文字）。範例裡的調整強度 α 只是暫定值。

測試：`python -m pytest -v`（gpu 與 network 測試預設跳過）。

## 外部資料

- NRC VAD Lexicon v2.1：到 https://saifmohammad.com/WebPages/nrc-vad.html 下載，解壓到
  `<緩存目錄>/nrc_vad/NRC-VAD-Lexicon-v2.1/`（緩存目錄預設 `C:\Users\xgame\.cache\emotional`，
  可用環境變數 `EMOTIONAL_CACHE` 改，但不能在專案目錄內）。詞表不可再散佈，不進 git；
  提取出的 `datasets/6.3_nrc_vad_emotions.csv` 同樣不進 git。
- Pain-axis（valen-research/Pain-axis，固定在 commit `213fb6c`）：其句子資料放在
  `datasets/6.2_pain_axis_3.1_pain_and_control_datasets.json`，其公開的疼痛向量由 6.9/02 自動下載到緩存目錄。
- Qwen-Scope SAE（`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`）：6.9/01 從 Hugging Face 自動下載；
  它訓練於基礎版模型，只當粗略參考。
- 各資料集的來源與授權說明見 `datasets/ATTRIBUTION.md`。
