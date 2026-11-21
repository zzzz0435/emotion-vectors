# 計劃一：環境、情緒向量與三個軸（階段 0–2）實現計劃

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Qwen3-8B 上建立全部情緒、概念、汙染向量和「正負面 / 激動程度 / 掌控感」三個軸，完成可分辨性、汙染和特徵字典的檢查，產出「使用者檢查點 1」的材料。

**Architecture:** 目錄結構仿照 GitHub 上星數最高的情緒相關專案 [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis)（76 星，commit `213fb6c`）：`datasets/` 存放輸入資料，`scripts/<設計文件章節號>_<名稱>/NN_<動作>.py` 按執行順序編號，`results/<章節號>_<名稱>/<模型名>/` 存放輸出，模型名採用 Pain-axis 的寫法（例如 `Qwen_3_8B_instruct`）。和 Pain-axis 不同的地方：可測試的邏輯統一放在 `scripts/emolab/` 共用模塊裡，編號腳本只負責命令行參數和文件讀寫。這樣做是因為設計要求每個階段的調整方式完全一致，Pain-axis 那種每個腳本各寫一份的做法容易出現不一致。

**Tech Stack:** Python 3.12、PyTorch（CUDA 12.8 以上版本）、transformers、bitsandbytes、scikit-learn、pandas、matplotlib、pytest。不使用 transformer_lens，因為它對 4-bit 量化模型的支援不完整，改用原生 PyTorch forward hook。

**Spec:** `docs/superpowers/specs/2026-11-21-emotion-moral-steering-design.md`（已定案）

**本計劃的範圍**：設計文件的階段 0、1、2，到使用者檢查點 1 為止。階段 3–5（校準、道德題、分析）和階段 6–7（Gemma 重做、報告）分別是計劃二和計劃三，等檢查點 1 的結果出來後再寫，因為它們的細節取決於這裡的結果（實際速度、憤怒和恐懼能否分開、哪些軸可靠）。

## Global Constraints

- 所有命令都在專案根目錄 `C:\Users\xgame\OneDrive\Desktop\emotional` 下執行，並先啟用虛擬環境。
- 虛擬環境放在 OneDrive 之外：`C:\Users\xgame\.venvs\emotional`。
- 模型權重使用 Hugging Face 預設緩存；激活值等大型中間文件放在 `emolab.paths.cache_dir()`。該路徑預設是 `C:\Users\xgame\.cache\emotional`，可以用環境變數 `EMOTIONAL_CACHE` 覆蓋，**但不得位於專案目錄內**。
- PyTorch 必須從 CUDA 12.8 以上的官方索引安裝（RTX 5060 是 sm_120 架構）；transformers ≥ 4.51（支援 Qwen3）。安裝完成後，用 `pip freeze` 把確切版本寫進 `requirements.txt`。
- 模型登記（key → Hugging Face repo → 輸出名 → 量化方式）：
  - `qwen3-8b` → `Qwen/Qwen3-8B` → `Qwen_3_8B_instruct` → 4-bit NF4（compute dtype 為 bfloat16，開啟 double quant）
  - `qwen3-1.7b` → `Qwen/Qwen3-1.7B` → `Qwen_3_1.7B_instruct` → 不量化，bfloat16（先導用）
  - `gemma3-4b` → `google/gemma-3-4b-it` → `Gemma_3_4B_instruct` → 8-bit（只載入文字部分）
  - `gemma3-1b` → `google/gemma-3-1b-it` → `Gemma_3_1B_instruct` → 不量化，bfloat16
- Qwen3 一律關閉思考：`apply_chat_template(..., enable_thinking=False)`。
- **層號約定**：層 L 指第 L 個 decoder block 的輸出，等於 transformer_lens 的 `blocks.L.hook_resid_post`，也等於 `output_hidden_states[L+1]`。
- **候選層**：`sorted({int(n_layers * f + 0.5) for f in (0.25, 0.40, 0.50, 0.65)})`。Qwen3-8B（36 層）→ [9, 14, 18, 23]。
- **取平均與調整時都排除每一行第一個非 padding 位置。**
- **調整量** = α × resid_norm[L] × 單位方向向量。resid_norm[L] 是該層在中性故事（訓練集）上逐詞元 L2 範數的平均值。
- 隨機種子 42。生成故事使用 temperature 0.8、top_p 0.95、max_new_tokens 300。
- 所有給模型看的文字一律用英文；給使用者看的 README、檢查點摘要一律用簡體中文。
- **倫理**：疼痛方向只用於短篇生成（本計劃只有檢查點 1 的 12 段範例），不做任何持續或互動式的施加。
- **憤怒 vs 恐懼可分辨的門檻**：驗證集 AUROC ≥ 0.75。**軸可靠的門檻**：留一法相關係數 ≥ 0.5。
- Git 只在本機使用，不推送到遠端。

## Review Focus

1. **長時間生成中途中斷**（當機、睡眠、顯示記憶體不足）：重跑時要從中斷處繼續，不能重複寫入或遺漏故事。→ Task 5 `test_resume_skips_existing`
2. **顯示記憶體不足（8GB）**：批次太大時要自動減半重試，而不是整個程式崩潰。→ Task 5 `test_oom_halves_batch`
3. **Qwen3 輸出殘留思考標籤**（`<think>`）：這類故事必須被剔除。→ Task 2 `test_format_chat_qwen_disables_thinking`、Task 6 `test_flags_thinking_leak`
4. **故事以變形詞洩漏情緒名稱**（例如 "angrily"、"furious"），而且不能誤判（例如 "made" 不能被當成 "mad"）。→ Task 6 `test_flags_emotion_inflections`
5. **4-bit 模型產生 NaN/Inf 激活值**：要立即報錯並指出是哪一層，不能悄悄寫進向量。→ Task 3 `test_nan_raises`

---

## 文件結構

```
emotional/
├── README.md                         # 簡體中文：專案說明、目錄結構、執行順序（仿照 Pain-axis README）
├── requirements.txt                  # 安裝後用 pip freeze 寫入確切版本
├── pytest.ini                        # pythonpath = scripts；markers：gpu、network
├── .gitignore                        # 複製 Pain-axis 的 Python 模板，另加 .venv/
├── datasets/
│   ├── ATTRIBUTION.md                # 外部資料的來源與授權
│   ├── 6.2_emotions.json             # 20 種情緒 + 中性：提示描述、禁用詞、限制條款
│   ├── 6.2_story_topics.json         # 50 個日常主題，其中 10 個標記為驗證用（heldout）
│   ├── 6.2_concepts.json             # 40 個對照概念 + 2 個汙染概念
│   ├── 6.2_pain_axis_3.1_pain_and_control_datasets.json   # Pain-axis 資料集（MIT），commit 213fb6c
│   ├── 6.2_stories.jsonl             # 由 Qwen3-8B 生成並過濾後的最終故事（Gemma 也會用）
│   └── 6.3_nrc_vad_emotions.csv      # 20 個情緒詞的 V/A/D 評分（摘自 NRC VAD v2）
├── scripts/
│   ├── emolab/                       # 共用模塊（可測試的邏輯都在這裡）
│   │   ├── __init__.py
│   │   ├── paths.py                  # 目錄、緩存位置
│   │   ├── models.py                 # 模型登記、載入、聊天格式、取得層
│   │   ├── batching.py               # 顯示記憶體不足時自動減半批次
│   │   ├── hooks.py                  # 讀取激活值、調整（steering）
│   │   ├── stories.py                # 生成任務、提示模板、過濾、平衡挑選
│   │   ├── vectors.py                # 對比平均、Pain-axis 去噪、AUROC、可分辨性、汙染
│   │   ├── axes.py                   # 三個軸的擬合、留一法驗證、殘差
│   │   ├── sae.py                    # 特徵字典解碼器讀取、最近特徵
│   │   └── report.py                 # 檢查點 1 中文摘要
│   ├── 6.0_env/01_check_environment.py
│   ├── 6.2_emotion_vectors/
│   │   ├── 01_generate_stories.py
│   │   ├── 02_filter_stories.py
│   │   ├── 03_extract_activations.py
│   │   ├── 04_build_vectors.py
│   │   ├── 05_pain_axis_vector.py
│   │   ├── 06_validate_vectors.py
│   │   ├── 07_contamination.py
│   │   └── 08_checkpoint_examples.py
│   ├── 6.3_axes/
│   │   ├── 01_fit_axes.py
│   │   └── 02_residual_vectors.py
│   └── 6.9_crosscheck/
│       ├── 01_sae_nearest_features.py
│       └── 02_compare_pain_axis.py
├── results/
│   ├── 6.0_env/env_report.json
│   ├── 6.2_emotion_vectors/<模型名>/
│   ├── 6.3_axes/<模型名>/
│   └── 6.9_crosscheck/<模型名>/
└── tests/
    ├── conftest.py                   # tiny_qwen3、tiny_gemma3 fixture（用 config 建立隨機小模型，不需下載）
    └── test_*.py
```

每個編號腳本都以 docstring 開頭，說明讀取什麼、寫出什麼、需要什麼（Pain-axis 的寫法），並用以下兩行引入共用模塊：

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

每個編號腳本都接受參數 `--model {qwen3-8b,qwen3-1.7b,gemma3-4b,gemma3-1b}`；會處理大量資料的腳本另外接受 `--limit N`，供先導測試使用。

---

### Task 1: 環境與專案骨架

**Files:**
- Create: `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `scripts/emolab/__init__.py`, `scripts/6.0_env/01_check_environment.py`, `tests/test_env.py`

**Interfaces:**
- Produces: 可用的虛擬環境；`results/6.0_env/env_report.json`，包含 `gpu`、`vram_total_mb`、`torch`、`cuda`、`transformers`、`bitsandbytes`，以及 `models: {key: {"loaded": bool, "peak_vram_mb": float, "tokens_per_s": float, "error": str|None}}`

- [ ] **Step 1: 確認磁盤空間**

Run: `powershell -c "(Get-PSDrive C).Free/1GB"`
Expected: ≥ 50（整個研究約需下載 Qwen3-8B 16GB、Qwen3-1.7B 3.4GB、Gemma 3 4B 8.6GB、Gemma 3 1B 2GB、SAE 每層 2GB）。不足就停下來回報使用者。

- [ ] **Step 2: 建立虛擬環境並安裝套件**

```bash
python -m venv /c/Users/xgame/.venvs/emotional
source /c/Users/xgame/.venvs/emotional/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes safetensors huggingface_hub scikit-learn scipy pandas numpy matplotlib tqdm pytest
pip freeze > requirements.txt
git init && git add -A && git commit -m "chore: project skeleton and design docs"
```

`.gitignore` 先複製 Pain-axis 的版本（commit 213fb6c 的 `.gitignore`），再加上 `.venv/`。`pytest.ini` 的內容：`[pytest]`、`pythonpath = scripts`、`addopts = -m "not gpu and not network"`，並登記 `gpu` 和 `network` 兩個 marker。

- [ ] **Step 3: 寫 GPU 測試**

```python
# tests/test_env.py
import pytest, torch

@pytest.mark.gpu
def test_cuda_supports_this_gpu():
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()

@pytest.mark.gpu
def test_bnb_4bit_forward():
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
    out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (2, 64) and torch.isfinite(out).all()
```

- [ ] **Step 4: 執行 GPU 測試**

Run: `pytest tests/test_env.py -m gpu -v`
Expected: 2 passed。失敗時，記下錯誤並換用較新的 CUDA 索引或 bitsandbytes 版本；仍然失敗就回報使用者。

- [ ] **Step 5: 實現 `01_check_environment.py`**

參數 `--models qwen3-1.7b qwen3-8b [gemma3-1b gemma3-4b]`。依序載入每個模型（模型載入在 Task 2 完成後才能用；這一步先只寫 GPU 和版本資訊，模型部分在 Task 2 Step 6 補跑），量測 `torch.cuda.max_memory_allocated()`，以及用 64 個新詞元算出 tokens/s，最後寫入 `env_report.json`。Gemma 載入失敗（未登錄）時，`error` 欄寫「需要使用者登錄 Hugging Face 並同意 Gemma 條款」，不中斷其他模型。

- [ ] **Step 6: 使用者操作（可以延後到計劃三）**

請使用者自己執行 `hf auth login`，並在瀏覽器中同意 `google/gemma-3-4b-it` 和 `google/gemma-3-1b-it` 的使用條款。Claude 不代為輸入任何 token。

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: environment check and test config"
```

---

### Task 2: 路徑與模型載入（`paths.py`、`models.py`）

**Files:**
- Create: `scripts/emolab/paths.py`, `scripts/emolab/models.py`, `tests/conftest.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `paths.REPO_ROOT: Path`, `paths.DATASETS: Path`, `paths.RESULTS: Path`
  - `paths.cache_dir() -> Path`：建立資料夾並回傳；如果位於 `REPO_ROOT` 之內，拋出 `ValueError`
  - `paths.results_dir(section: str, model_name: str | None = None) -> Path`：例如 `results_dir("6.2_emotion_vectors", "Qwen_3_8B_instruct")`，會建立資料夾
  - `models.ModelSpec(key: str, repo_id: str, name: str, quant: Literal["4bit", "8bit", "none"])`（frozen dataclass）
  - `models.MODELS: dict[str, ModelSpec]`，四個 key 見 Global Constraints
  - `models.load(key: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`：eval 模式、`device_map={"": 0}`、`tokenizer.padding_side = "left"`；沒有 pad token 時，用 eos 作為 pad token。`gemma3-*` 用 `Gemma3ForCausalLM`（只載入文字部分）。
  - `models.get_layers(model) -> torch.nn.ModuleList`：依序嘗試 `model.model.layers`、`model.model.language_model.layers`、`model.language_model.model.layers`
  - `models.candidate_layers(n_layers: int) -> list[int]`
  - `models.format_chat(tokenizer, user_msg: str, system_msg: str | None = None) -> str`

- [ ] **Step 1: 寫測試**

`conftest.py` 提供 `tiny_qwen3` fixture：`Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8))`，float32、eval、固定種子。同樣方式提供 `tiny_gemma3`（`Gemma3ForCausalLM` + `Gemma3TextConfig`）。

```python
def test_candidate_layers():
    assert candidate_layers(36) == [9, 14, 18, 23]
    assert candidate_layers(28) == [7, 11, 14, 18]
    assert candidate_layers(34) == [9, 14, 17, 22]
    assert candidate_layers(26) == [7, 10, 13, 17]

def test_get_layers_qwen_and_gemma(tiny_qwen3, tiny_gemma3):
    assert len(get_layers(tiny_qwen3)) == 4
    assert len(get_layers(tiny_gemma3)) == 4

def test_cache_dir_must_be_outside_repo(monkeypatch):
    monkeypatch.setenv("EMOTIONAL_CACHE", str(REPO_ROOT / "cache"))
    with pytest.raises(ValueError):
        cache_dir()

@pytest.mark.network
def test_format_chat_qwen_disables_thinking():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    s = format_chat(tok, "Hello there")
    assert "Hello there" in s
    assert s.endswith("<think>\n\n</think>\n\n")
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_models.py -v -m "not gpu"`
Expected: FAIL（ImportError）

- [ ] **Step 3: 實現 `paths.py` 和 `models.py`**

`format_chat` 調用 `tokenizer.apply_chat_template([...], tokenize=False, add_generation_prompt=True, enable_thinking=False)`；如果 tokenizer 不接受 `enable_thinking`，就不傳這個參數。

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_models.py -v -m "not gpu"`，以及 `pytest tests/test_models.py -v -m network`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: model registry, loading and paths"
```

- [ ] **Step 6: 補跑環境檢查的模型部分**

Run: `python scripts/6.0_env/01_check_environment.py --models qwen3-1.7b qwen3-8b`
Expected: 兩個模型的 `loaded` 都是 true，`qwen3-8b` 的 `peak_vram_mb` < 7500。如果放不下，停下來回報使用者（設計文件 6.1 有備案）。把 `env_report.json` 一起 commit。

---

### Task 3: 讀取激活值與調整（`hooks.py`）

**Files:**
- Create: `scripts/emolab/hooks.py`, `tests/test_hooks.py`

**Interfaces:**
- Consumes: `models.get_layers`
- Produces:
  - `capture_mean_activations(model, input_ids: LongTensor[B,T], attention_mask: LongTensor[B,T], layers: list[int]) -> dict[int, Tensor[B, d]]`：float32，在 CPU 上；對每一行中 `attention_mask==1` 的位置取平均，排除第一個非 padding 位置。只要出現 NaN 或 Inf，就拋出 `FloatingPointError`，訊息包含層號。
  - `mean_token_norm(model, input_ids, attention_mask, layers) -> dict[int, float]`：同樣的位置規則，回傳逐詞元 L2 範數的平均值。
  - `steer(model, layer: int, direction: Tensor[d], alpha: float, resid_norm: float) -> ContextManager[None]`：在 layer 的輸出加上 `alpha * resid_norm * direction / ||direction||`。prefill 階段（序列長度 > 1）跳過第一個非 padding 位置；逐字生成階段（序列長度 == 1）一律加上。離開 context 時移除 hook；`alpha == 0` 時不註冊 hook。

- [ ] **Step 1: 寫測試**

```python
def test_capture_matches_hidden_states(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 7)); mask = torch.ones_like(ids)
    got = capture_mean_activations(tiny_qwen3, ids, mask, [1])[1]
    hs = tiny_qwen3(ids, attention_mask=mask, output_hidden_states=True).hidden_states[2]
    torch.testing.assert_close(got[0], hs[0, 1:].mean(0), atol=1e-5, rtol=1e-4)

def test_capture_respects_left_padding(tiny_qwen3):
    # 同一序列單獨輸入 vs 左側補 3 個 pad 後批次輸入 → 平均值相同（atol 1e-4）

def test_steer_adds_scaled_unit_vector(tiny_qwen3):
    # direction = 3*e0, alpha=0.5, resid_norm=2.0 → 第 1 層輸出在位置 ≥1 增加 1.0*e0，位置 0 不變

def test_steer_zero_alpha_registers_nothing(tiny_qwen3): ...
def test_steer_hook_removed_after_exit(tiny_qwen3): ...   # 離開 context 後輸出與原本完全相同

def test_nan_raises(tiny_qwen3, monkeypatch):
    # 用 forward hook 讓第 2 層輸出 NaN → capture_mean_activations 拋出 FloatingPointError，訊息含 "layer 2"
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_hooks.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 實現 `hooks.py`**

兩個函式都在 `get_layers(model)[L]` 上註冊 `register_forward_hook`。decoder layer 的輸出可能是 tuple，只修改第一個元素。

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_hooks.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: activation capture and steering hooks"
```

---

### Task 4: 輸入資料集

**Files:**
- Create: `datasets/6.2_emotions.json`, `datasets/6.2_story_topics.json`, `datasets/6.2_concepts.json`, `datasets/6.2_pain_axis_3.1_pain_and_control_datasets.json`, `datasets/6.3_nrc_vad_emotions.csv`, `datasets/ATTRIBUTION.md`, `tests/test_datasets.py`

**Interfaces:**
- Produces（schema，後面的 Task 都依賴這些欄位）：
  - `6.2_emotions.json`：list，每個元素是 `{"label": str, "group": "core"|"map"|"neutral", "description": str, "banned": [regex, ...], "restriction": str|null, "subtypes": [str, ...]|null}`
  - `6.2_story_topics.json`：list，每個元素是 `{"idx": int, "topic": str, "heldout": bool, "concept_topic": bool}`
  - `6.2_concepts.json`：list，每個元素是 `{"label": str, "role": "control"|"contamination", "description": str}`
  - `6.3_nrc_vad_emotions.csv`：欄位 `label,word,valence,arousal,dominance,lexicon_version`

- [ ] **Step 1: 寫測試**

```python
CORE = {"anger", "fear", "sadness", "joy", "compassion", "pain"}
MAP = {"calm", "pride", "gratitude", "love", "hope", "excitement", "amusement",
       "jealousy", "disgust", "shame", "guilt", "contempt", "anxiety", "desperation"}

def test_emotions_schema():
    e = load_json("6.2_emotions.json")
    assert {x["label"] for x in e if x["group"] == "core"} == CORE
    assert {x["label"] for x in e if x["group"] == "map"} == MAP
    assert [x["label"] for x in e if x["group"] == "neutral"] == ["neutral"]
    assert all(x["banned"] for x in e if x["group"] != "neutral")
    pain = next(x for x in e if x["label"] == "pain")
    assert len(pain["subtypes"]) == 4          # physical, psychological, loneliness, mental strain; no moral

def test_topics():
    t = load_json("6.2_story_topics.json")
    assert len(t) == 50 and sum(x["heldout"] for x in t) == 10
    assert sum(x["concept_topic"] for x in t) == 20
    assert sum(x["concept_topic"] and x["heldout"] for x in t) == 4

def test_concepts():
    c = load_json("6.2_concepts.json")
    assert sum(x["role"] == "control" for x in c) == 40
    assert {x["label"] for x in c if x["role"] == "contamination"} == {"punishment", "harming_others"}

def test_vad_covers_all_emotions():
    df = pd.read_csv(DATASETS / "6.3_nrc_vad_emotions.csv")
    assert set(df.label) == CORE | MAP
    assert df[["valence", "arousal", "dominance"]].notna().all().all()
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL（找不到文件）

- [ ] **Step 3: 寫資料**

- **主題**：50 個情緒中性的日常主題（例如 "waiting for a delayed train"、"repairing a bicycle"、"cooking dinner for one"）。不能涉及任何人做錯事。
- **限制條款**：一般情緒的 `restriction` 為 null，使用通用條款。以下情緒另訂條款：
  - guilt、shame：`"The story may involve a minor personal slip that hurts no one, but must not involve blame from others, punishment, revenge, harm to other people, or moral decisions. "`
  - jealousy、contempt：`"The feeling is directed at another person's luck, success or habits, but nobody does anything wrong. The story must not involve blame, punishment, revenge, harm to other people, or moral decisions. "`
- **憤怒和恐懼的 description** 要符合設計 6.2：憤怒針對事物或情境，恐懼針對危險情境，都不能針對某個人的惡意。
- **疼痛的 4 個 subtype**：`physical pain from an accidental injury or an illness`、`deep inner psychological pain`、`the pain of loneliness and feeling left out`、`the pain of an exhausting mental struggle`。
- **banned**：每種情緒約 5–10 個 regex，例如憤怒 `[r"\bangr\w*", r"\banger\w*", r"\bfurious\w*", r"\bfury\b", r"\brage\w*", r"\benrag\w*", r"\bmad\b", r"\birritat\w*"]`。
- **概念**：40 個對照概念（設計 6.7 的清單）。汙染概念的 description：punishment → `"someone being punished for something they did"`；harming_others → `"one person physically harming another person"`。
- **Pain-axis 資料集**：從 `https://raw.githubusercontent.com/valen-research/Pain-axis/213fb6c89c96004dbe93d1704c636ad8c9d801f3/datasets/3.1_pain_and_control_datasets.json` 下載，原樣保存。
- **NRC VAD**：從官方頁面下載 NRC VAD Lexicon v2，確認授權允許研究使用；只摘錄 20 個情緒詞的評分寫進 CSV，不要轉存整個詞表。如果條款不允許摘錄，就停下來回報使用者。
- **ATTRIBUTION.md**：寫明 Pain-axis（MIT，commit 213fb6c）、NRC VAD（版本和授權條款）、MoralChoice（CC-BY-4.0，計劃二才會用到）。

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_datasets.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "data: emotion, topic, concept and VAD datasets"
```

---

### Task 5: 生成故事（`stories.py` 前半、`batching.py`、`01_generate_stories.py`）

**Files:**
- Create: `scripts/emolab/batching.py`, `scripts/emolab/stories.py`, `scripts/6.2_emotion_vectors/01_generate_stories.py`, `tests/test_stories_generate.py`

**Interfaces:**
- Consumes: Task 4 的 schema；`models.load`、`models.format_chat`
- Produces:
  - `batching.run_with_oom_backoff(fn: Callable[[list[T]], list[R]], items: list[T], batch_size: int) -> Iterator[tuple[list[T], list[R]]]`：遇到 `torch.cuda.OutOfMemoryError` 時，執行 `torch.cuda.empty_cache()`，批次減半後重試；批次已經是 1 還失敗，就把錯誤往上拋。
  - `stories.StoryJob`（frozen dataclass）：`id: str, label: str, kind: Literal["emotion","neutral","concept","contamination"], topic_idx: int, topic: str, subtype: str | None, replicate: int, prompt: str`
  - `stories.build_jobs(emotions: list[dict], topics: list[dict], concepts: list[dict]) -> list[StoryJob]`
  - `stories.build_topup_jobs(shortfall: dict[tuple[str, int, int], int], emotions, topics, concepts, replicate_offset: int) -> list[StoryJob]`：為每個缺額格子 `(label, topic_idx, subtype_idx)` 產生對應數量的新任務。新任務沿用該格子原本的 subtype，replicate 編號從 `replicate_offset` 開始。
  - `stories.run_generation(jobs: list[StoryJob], out_path: Path, generate_fn: Callable[[list[str]], list[str]], batch_size: int, model_name: str) -> int`：以 JSONL 逐行追加寫入，每行包含 `id, label, kind, topic_idx, topic, subtype, replicate, text, model`；已經存在的 id 會跳過；回傳這次新寫入的筆數。

**數量規則**（每個 id 是 `f"{label}__t{topic_idx:02d}__s{subtype_idx}__r{replicate}"`，subtype_idx 在沒有 subtype 時是 0）：

| 種類 | 主題 | 每個主題的篇數 | 合計 |
|---|---|---|---|
| 核心情緒（除疼痛外） | 全部 50 個 | 2 | 100 |
| 疼痛 | 全部 50 個；4 個 subtype 依 `(topic_idx * 2 + replicate) % 4` 輪流分配 | 2 | 100（每個 subtype 25） |
| 地圖情緒 | 全部 50 個 | 1 | 50 |
| 中性 | 全部 50 個 | 2 | 100 |
| 對照概念、汙染概念 | `concept_topic` 為 true 的 20 個 | 2 | 每個 40 |

**提示模板**（確切文字）：

```python
GENERAL_RESTRICTION = ("The story must not involve anyone doing something wrong, blame, punishment, "
                       "revenge, harm to other people, or moral decisions. ")
EMOTION = ("Write a short story of about 150 words about {topic}. Throughout the story, the main character "
           "feels {description}. Never name this feeling or any word for it; show it only through the "
           "character's thoughts, body sensations and actions. {restriction}Output only the story.")
NEUTRAL = ("Write a short story of about 150 words about {topic}. The main character feels no particular "
           "emotion; describe what happens in a calm, matter-of-fact way. {restriction}Output only the story.")
CONCEPT = ("Write a short story of about 150 words about {topic}, in which {description} plays a central "
           "role. The main character feels no particular emotion; describe what happens in a calm, "
           "matter-of-fact way. {restriction}Output only the story.")
CONTAMINATION = ("Write a short story of about 150 words about {topic}, in which {description} plays a "
                 "central role. Describe what happens in a calm, matter-of-fact way without describing "
                 "anyone's feelings. Output only the story.")
```

疼痛的 `{description}` 使用它的 subtype 文字。`{restriction}` 優先使用該情緒自己的 `restriction`，否則使用 `GENERAL_RESTRICTION`。

- [ ] **Step 1: 寫測試**

```python
def test_build_jobs_counts():
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)
    c = Counter(j.label for j in jobs)
    assert c["anger"] == 100 and c["pain"] == 100 and c["neutral"] == 100
    assert c["jealousy"] == 50 and c["weather"] == 40 and c["punishment"] == 40
    assert set(Counter(j.subtype for j in jobs if j.label == "pain").values()) == {25}

def test_topup_keeps_cell_subtype():
    jobs = build_topup_jobs({("pain", 3, 2): 2}, EMOTIONS, TOPICS, CONCEPTS, replicate_offset=5)
    assert len(jobs) == 2 and all(j.subtype == PAIN_SUBTYPES[2] and j.topic_idx == 3 for j in jobs)
    assert [j.replicate for j in jobs] == [5, 6]

def test_ids_unique_and_deterministic():
    a, b = build_jobs(EMOTIONS, TOPICS, CONCEPTS), build_jobs(EMOTIONS, TOPICS, CONCEPTS)
    assert [j.id for j in a] == [j.id for j in b] and len({j.id for j in a}) == len(a)

def test_contamination_prompt_has_no_restriction():
    j = next(j for j in build_jobs(EMOTIONS, TOPICS, CONCEPTS) if j.label == "punishment")
    assert "must not involve" not in j.prompt

def test_resume_skips_existing(tmp_path):
    # 第一次執行時，fake generate_fn 在第 5 次調用拋出 KeyboardInterrupt；第二次正常執行
    # → 文件中每個 id 只出現一次，總數 == len(jobs)

def test_oom_halves_batch(tmp_path):
    # 當 len(prompts) > 2 時，fake generate_fn 拋出 torch.cuda.OutOfMemoryError
    # → 以 batch_size=8 執行完畢，所有 id 都已寫入
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_stories_generate.py -v`
Expected: FAIL

- [ ] **Step 3: 實現 `batching.py` 和 `stories.py` 的生成部分，以及腳本 `01_generate_stories.py`**

- 腳本參數：`--model`、`--out`（預設為 `results/6.2_emotion_vectors/<模型名>/stories_raw.jsonl`）、`--batch-size 16`、`--limit`、`--top-up`。`--top-up` 會讀取 Task 6 寫出的 `shortfall.json`（格子 → 缺額），以 `build_topup_jobs` 產生任務，`replicate_offset` 設為現有最大 replicate + 1。
- `generate_fn`：對每個 prompt 先套 `format_chat`，左側補齊 padding，執行 `model.generate(do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=300)`，每個批次之前執行 `torch.manual_seed(42 + 批次序號)`，只解碼新生成的詞元。

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_stories_generate.py -v`
Expected: 6 passed

- [ ] **Step 5: 先導執行**

Run: `python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-1.7b --limit 40`
Expected: 寫入 40 行；人工打開 3 行確認內容是故事、沒有 `<think>`。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: story generation with resume and OOM backoff"
```

- [ ] **Step 7: 正式執行（背景執行，預計約 1 小時）**

Run: `python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b`
Expected: 寫入約 3,080 行。中途中斷就用同一個命令重跑。

---

### Task 6: 過濾故事與抽查（`stories.py` 後半、`02_filter_stories.py`）

**Files:**
- Modify: `scripts/emolab/stories.py`
- Create: `scripts/6.2_emotion_vectors/02_filter_stories.py`, `tests/test_stories_filter.py`

**Interfaces:**
- Produces:
  - `stories.CONTAMINATION_PATTERNS: list[str]` = `[r"\bpunish\w*", r"\brevenge\w*", r"\baveng\w*", r"\bretaliat\w*", r"\bblam\w*", r"\bguilty\b", r"\bcrim\w*", r"\bsteal\w*", r"\bstole\w*", r"\btheft\w*", r"\bassault\w*", r"\bkill\w*", r"\bmurder\w*", r"\babus\w*", r"\bcheat\w*", r"\bbetray\w*", r"\bjail\w*", r"\bprison\w*", r"\barrest\w*"]`
  - `stories.check_story(text: str, banned: list[str], kind: str, min_words: int = 80, max_words: int = 300) -> list[str]`：回傳不通過的原因，可能是 `"thinking_leak"`、`"names_emotion"`、`"contamination"`（只檢查 kind ≠ contamination 的故事）、`"too_short"`、`"too_long"`。比對時不分大小寫。
  - `stories.select_balanced(rows: list[dict], targets: dict[tuple[str, int, int], int]) -> tuple[list[dict], dict[tuple, int]]`：每個 `(label, topic_idx, subtype_idx)` 格子依 replicate 順序取前 N 篇通過的故事；回傳 `(選中的故事, 缺額)`。
- 輸出：
  - `datasets/6.2_stories.jsonl`：只在 `--model qwen3-8b` 時寫入；會多一個 `heldout` 欄位，取自主題檔。
  - `results/6.2_emotion_vectors/<模型名>/filter_report.csv`：欄位為 label、n_raw、各原因的剔除數、n_kept、shortfall。
  - `results/6.2_emotion_vectors/<模型名>/shortfall.json`：`[{"label", "topic_idx", "subtype_idx", "missing"}]`，供 `--top-up` 使用。
  - `results/6.2_emotion_vectors/<模型名>/review_samples.md`：中文標題；每種核心情緒隨機抽 5 篇，每個汙染概念抽 2 篇，種子 42。

- [ ] **Step 1: 寫測試**

```python
def test_flags_thinking_leak():
    assert "thinking_leak" in check_story("<think>hmm</think> " + "word " * 100, [], "emotion")

def test_flags_emotion_inflections():
    pats = [r"\bangr\w*", r"\bmad\b"]
    assert "names_emotion" in check_story("He answered Angrily. " + "word " * 100, pats, "emotion")
    assert check_story("She made tea. " + "word " * 100, pats, "emotion") == []

def test_contamination_only_for_non_contamination_kind():
    t = "He was punished. " + "word " * 100
    assert "contamination" in check_story(t, [], "emotion")
    assert check_story(t, [], "contamination") == []

def test_word_bounds():
    assert "too_short" in check_story("word " * 50, [], "neutral")
    assert "too_long" in check_story("word " * 400, [], "neutral")

def test_select_balanced_reports_shortfall():
    # 格子 ("anger", 0, 0) 目標 2 篇，只有 1 篇通過 → 缺額 {("anger", 0, 0): 1}
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_stories_filter.py -v`
Expected: FAIL

- [ ] **Step 3: 實現過濾邏輯與腳本**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_stories_filter.py -v`
Expected: 5 passed

- [ ] **Step 5: 先導執行，再正式執行；有缺額就補生成**

```bash
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-1.7b
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b
# 如果 filter_report.csv 中有 shortfall > 0：
python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b --top-up
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b
```

Expected: shortfall 全部為 0。補生成兩輪後仍有缺額，就在報告中記錄並繼續（地圖情緒可以接受少於 50 篇，核心情緒不行）。

- [ ] **Step 6: Commit**（包含 `datasets/6.2_stories.jsonl`、`filter_report.csv`、`review_samples.md`）

```bash
git add -A && git commit -m "feat: story filtering; final story dataset"
```

---

### Task 7: 讀取激活值（`03_extract_activations.py`）

**Files:**
- Create: `scripts/6.2_emotion_vectors/03_extract_activations.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `hooks.capture_mean_activations`、`hooks.mean_token_norm`、`batching.run_with_oom_backoff`、`paths.cache_dir`
- Produces:
  - 在 `03_extract_activations.py` 中定義 `extract(model, encode_fn: Callable[[list[str]], dict], rows: list[dict], layers: list[int], batch_size: int, out_dir: Path) -> pd.DataFrame`；測試用 `importlib` 依文件路徑載入這個腳本。
  - 每層寫一個 `<cache_dir>/6.2/<模型名>/acts_L{L}.npy`（float32，形狀 [N, d]，行順序與 index 相同）。
  - `<cache_dir>/6.2/<模型名>/index.csv`，欄位為 row、id、label、kind、topic_idx、subtype、heldout。
  - `results/6.2_emotion_vectors/<模型名>/resid_norms.json`：`{"<L>": float}`，只用中性、非 heldout 的故事計算。
- 腳本參數：`--model`、`--stories`（預設為 `datasets/6.2_stories.jsonl`）、`--batch-size 16`、`--limit`、`--force`。
- 輸入文字直接用故事原文，不套聊天格式；`add_special_tokens=True`。依長度排序後組成批次，以減少 padding。
- 已經存在、而且和 index 的 id 一致的文件會被跳過；加 `--force` 可以重算。

- [ ] **Step 1: 寫測試**（用 `tiny_qwen3` 和一個把字元映射成 id 的 fake `encode_fn`）

```python
def test_extract_shapes_and_order(tiny_qwen3, tmp_path):
    idx = extract(tiny_qwen3, fake_encode, ROWS_5, [1, 2], batch_size=2, out_dir=tmp_path)
    a = np.load(tmp_path / "acts_L1.npy")
    assert a.shape == (5, 32) and list(idx.id) == [r["id"] for r in ROWS_5]

def test_extract_skips_when_done(tiny_qwen3, tmp_path):
    # 執行兩次；第二次時讓 capture 被呼叫就拋錯 → 仍然正常回傳（證明有跳過）
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL

- [ ] **Step 3: 實現**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_extract.py -v`
Expected: 2 passed

- [ ] **Step 5: 先導執行，再正式執行**

```bash
python scripts/6.2_emotion_vectors/03_extract_activations.py --model qwen3-1.7b --stories results/6.2_emotion_vectors/Qwen_3_1.7B_instruct/stories_raw.jsonl --limit 40
python scripts/6.2_emotion_vectors/03_extract_activations.py --model qwen3-8b
```

Expected: 第 9、14、18、23 層的 npy 都已產生；`resid_norms.json` 有 4 個數值。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: activation extraction"
```

---

### Task 8: 建立向量（`vectors.py` 前半、`04_build_vectors.py`、`05_pain_axis_vector.py`）

**Files:**
- Create: `scripts/emolab/vectors.py`, `scripts/6.2_emotion_vectors/04_build_vectors.py`, `scripts/6.2_emotion_vectors/05_pain_axis_vector.py`, `tests/test_vectors_build.py`

**Interfaces:**
- Consumes: Task 7 的輸出
- Produces:
  - `vectors.diff_of_means(acts: np.ndarray, labels: np.ndarray, target: str, baseline: str, mask: np.ndarray) -> np.ndarray`：只用 `mask` 為 True 的行。
  - `vectors.pain_axis_vector(pain_acts: np.ndarray, control_acts: np.ndarray, denoise_variance: float = 0.5) -> np.ndarray`：移植自 Pain-axis commit 213fb6c 中 `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py` 的 `compute_pain_vector`（第 151–185 行，`baseline="all_controls"`、`denoise=True`）。docstring 要註明出處與 MIT 授權。
  - `results/6.2_emotion_vectors/<模型名>/vectors.pt`：`{"layers": list[int], "resid_norm": {L: float}, "raw": {label: Tensor[n_layers, d]}, "unit": {label: Tensor[n_layers, d]}, "meta": {"model": str, "n_train": {label: int}, "heldout_topics": list[int]}}`。label 包括 20 種情緒、40 個對照概念、2 個汙染概念；Task 8 的 05 腳本會再加入 `pain_painaxis`。
- **建立方式**：所有向量都是「該類訓練集平均 − 中性訓練集平均」。訓練集 = 主題 `heldout == False` 的故事。
- **05 腳本**：使用 Pain-axis 資料集的 `S2_1P`。疼痛類別取 A1、A2、A3、A5，排除 A4（Moral Injury）；對照類別取 B、C1、C2、D、E。輸入句子原文，以同一套 `capture_mean_activations` 讀取候選層的激活值；結果以 `pain_painaxis` 寫回 `vectors.pt`。

- [ ] **Step 1: 寫測試**

```python
def test_diff_of_means_simple():
    acts = np.array([[1., 0], [1, 0], [0, 0], [0, 0]]); labels = np.array(["a", "a", "n", "n"])
    np.testing.assert_allclose(diff_of_means(acts, labels, "a", "n", np.ones(4, bool)), [1, 0])

def test_mask_excludes_heldout():
    # heldout 行設成極端值，mask 把它排除 → 結果不受影響

def test_pain_axis_denoise_removes_dominant_control_direction():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=(200, 4)) * [10, 1, 1, 1]          # 對照組在 e0 上變異極大
    pain = ctrl[:50] + [3, 2, 0, 0]                           # 疼痛組在 e0、e1 上都有偏移
    v = pain_axis_vector(pain, ctrl, 0.5)
    assert abs(v[0]) < 0.1 * abs(v[1])
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_vectors_build.py -v`
Expected: FAIL

- [ ] **Step 3: 實現**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_vectors_build.py -v`
Expected: 3 passed

- [ ] **Step 5: 正式執行**

```bash
python scripts/6.2_emotion_vectors/04_build_vectors.py --model qwen3-8b
python scripts/6.2_emotion_vectors/05_pain_axis_vector.py --model qwen3-8b
```

Expected: `vectors.pt` 包含 63 個 label（20 + 40 + 2 + `pain_painaxis`），每個都是 4 層。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: emotion, concept and Pain-axis vectors"
```

---

### Task 9: 驗證向量（`vectors.py`、`06_validate_vectors.py`）

**Files:**
- Modify: `scripts/emolab/vectors.py`
- Create: `scripts/6.2_emotion_vectors/06_validate_vectors.py`, `tests/test_vectors_validate.py`

**Interfaces:**
- Produces:
  - `vectors.projection_auroc(acts, labels, unit_vec, pos: str, neg: str, mask) -> float`
  - `vectors.separability(acts, labels, train_mask, test_mask, a: str = "anger", b: str = "fear") -> float`：用訓練集算出方向 `unit(mean_a − mean_b)`，在測試集上計算 a 對 b 的 AUROC。
  - `SEPARABLE_THRESHOLD = 0.75`
- 輸出：
  - `validation.csv`：欄位 label、layer、auroc_vs_neutral，涵蓋全部 20 種情緒和 `pain_painaxis`，只用 heldout 資料。
  - `separability.json`：`{"<L>": {"auroc": float, "separable": bool}}`
  - `similarity_L{L}.csv` 和 `similarity_L{L}.png`：20 種情緒加 `pain_painaxis` 的餘弦相似度熱圖。

- [ ] **Step 1: 寫測試**

```python
def test_separability_high_when_distinct():   # 兩群在 e0 上相距 5 → AUROC > 0.95
def test_separability_chance_when_identical(): # 同一分佈 → 0.35 < AUROC < 0.65，且 < SEPARABLE_THRESHOLD
def test_projection_auroc_direction():         # pos 群在單位向量方向上較大 → AUROC > 0.9
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_vectors_validate.py -v`
Expected: FAIL

- [ ] **Step 3: 實現**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_vectors_validate.py -v`
Expected: 3 passed

- [ ] **Step 5: 正式執行**

Run: `python scripts/6.2_emotion_vectors/06_validate_vectors.py --model qwen3-8b`
Expected: 4 個輸出都已產生。如果某種核心情緒在所有層的 auroc_vs_neutral 都低於 0.7，在 Task 13 的摘要中標示出來。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: vector validation and anger-fear separability"
```

---

### Task 10: 汙染檢查（`vectors.py`、`07_contamination.py`）

**Files:**
- Modify: `scripts/emolab/vectors.py`
- Create: `scripts/6.2_emotion_vectors/07_contamination.py`, `tests/test_contamination.py`

**Interfaces:**
- Produces:
  - `vectors.contamination_flags(unit: dict[str, np.ndarray], emotions: list[str], concepts: list[str], probe: str, reference: str = "sadness") -> dict[str, dict]`：對每種情緒回傳 `{"cos": float, "ref_cos": float, "p95_concepts": float, "flagged": bool}`。flagged 的條件是 `cos > ref_cos` 且 `cos > p95_concepts`。
  - `vectors.remove_component(v: np.ndarray, probe: np.ndarray) -> np.ndarray`：回傳去掉 probe 成分後的單位向量。
- 兩個 probe 都要檢查：`punishment` 和 `harming_others`，逐層計算。被標示的情緒，依序去掉所有被標示的 probe 成分，得到 `<label>__clean`。
- 輸出：`contamination.json`（逐層、逐 probe、逐情緒）和 `vectors_clean.pt`（只包含 `__clean` 向量，格式與 `vectors.pt["unit"]` 相同）。

- [ ] **Step 1: 寫測試**

```python
def test_flags_contaminated_and_cleans():
    # anger = 0.8*p + 0.6*u；sadness 與 joy 和 p 正交；40 個概念在 p 上的餘弦都 < 0.1
    # → anger 被標示、joy 沒有；remove_component 後與 p 的 |cos| < 1e-6，而且長度為 1
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_contamination.py -v`
Expected: FAIL

- [ ] **Step 3: 實現**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_contamination.py -v`
Expected: PASS

- [ ] **Step 5: 正式執行**

Run: `python scripts/6.2_emotion_vectors/07_contamination.py --model qwen3-8b`
Expected: 產生 `contamination.json`；只有在有情緒被標示時才會產生 `vectors_clean.pt`。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: contamination check and clean vectors"
```

---

### Task 11: 三個軸與殘差向量（`axes.py`、`6.3_axes/`）

**Files:**
- Create: `scripts/emolab/axes.py`, `scripts/6.3_axes/01_fit_axes.py`, `scripts/6.3_axes/02_residual_vectors.py`, `tests/test_axes.py`

**Interfaces:**
- Consumes: `vectors.pt["unit"]`、`datasets/6.3_nrc_vad_emotions.csv`
- Produces:
  - `axes.ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]`、`axes.RELIABLE_R = 0.5`
  - `axes.fit_axis(X: np.ndarray[n, d], y: np.ndarray[n]) -> tuple[np.ndarray, float]`：用 `RidgeCV(alphas=ALPHAS, fit_intercept=True)`，回傳單位化的 `coef_` 和選中的 alpha。
  - `axes.loo_r(X, y) -> float`：每次留下一個樣本不參與擬合，用 `fit_axis` 的完整流程（包括截距）預測它的分數，最後計算所有預測值與真實值的 Pearson r。
  - `axes.fit_axes(unit: dict[str, np.ndarray], vad: pd.DataFrame, exclude: set[str]) -> dict`：回傳 `{"V"|"A"|"D": {"axis": ndarray, "alpha": float, "loo_r": float, "reliable": bool}}`。三個軸依 V → A → D 的順序做 Gram–Schmidt 正交化。
  - `axes.residual(v: np.ndarray, axes: list[np.ndarray]) -> np.ndarray`：回傳去掉所有給定軸成分後的單位向量。
- **擬合組**（每一層都要做）：`all`（不排除）、`for_anger`（排除 anger、fear）、`for_pain`（排除 pain）、`for_compassion`（排除 compassion）。
- **殘差向量**：
  - `anger__resid3`：用 `for_anger` 中可靠的軸。
  - `anger__resid2`：只用 V、A。
  - `pain__resid3`：用 `for_pain` 中可靠的軸。
  - `compassion__resid3`：用 `for_compassion` 中可靠的軸。
  - 如果該組的 V 軸不可靠，就不產生這個殘差，並在 `axes.json` 中記錄 `"untestable"`。
- 輸出：
  - `results/6.3_axes/<模型名>/axes.pt`：`{"layers": [...], "fits": {fit_set: {L: {"V"|"A"|"D": Tensor[d]}}}}`。其中 `all` 組的三個軸，在計劃二中會用作調整方向 `axis_V`、`axis_A`、`axis_D`。
  - `axes.json`：每組、每層、每軸的 alpha、loo_r、reliable。
  - `residuals.pt`：格式與 `vectors.pt["unit"]` 相同。

- [ ] **Step 1: 寫測試**

```python
def test_recovers_planted_axis():
    rng = np.random.default_rng(0); a = unit(rng.normal(size=64))
    y = rng.uniform(-1, 1, 20); X = np.outer(y, a) + 0.05 * rng.normal(size=(20, 64))
    axis, _ = fit_axis(X, y); assert abs(axis @ a) > 0.9
    assert loo_r(X, y) > 0.8

def test_random_ratings_unreliable():
    # X 為隨機向量，y 為與 X 無關的隨機評分 → loo_r < RELIABLE_R

def test_exclude_really_excludes():
    # 把被排除的 label 換成垃圾向量 → 擬合出的軸完全相同

def test_residual_orthogonal_and_unit():
    # 對每個軸的 |residual @ axis| < 1e-6，而且 ||residual|| == 1
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_axes.py -v`
Expected: FAIL

- [ ] **Step 3: 實現**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_axes.py -v`
Expected: 4 passed

- [ ] **Step 5: 正式執行**

```bash
python scripts/6.3_axes/01_fit_axes.py --model qwen3-8b
python scripts/6.3_axes/02_residual_vectors.py --model qwen3-8b
```

Expected: 產生 `axes.json`，並且至少有一層的 V 軸可靠。如果所有層的 V 軸都不可靠，在 Task 13 的摘要中標示「H4 無法檢驗」。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: VAD axes, LOO validation and residual vectors"
```

---

### Task 12: 交叉確認（`sae.py`、`6.9_crosscheck/`）

**Files:**
- Create: `scripts/emolab/sae.py`, `scripts/6.9_crosscheck/01_sae_nearest_features.py`, `scripts/6.9_crosscheck/02_compare_pain_axis.py`, `tests/test_sae.py`

**Interfaces:**
- Produces:
  - `sae.load_decoder(path: Path, d_model: int) -> np.ndarray`：回傳 [n_features, d_model] 的矩陣，每一列都單位化。如果矩陣存成 [d_model, n] 就轉置。key 名稱請先實際打開文件確認，常見的是 `W_dec`。
  - `sae.nearest_features(vec: np.ndarray, W_dec: np.ndarray, k: int = 10) -> list[tuple[int, float]]`：依餘弦由高到低排序。
- **01 腳本**：
  - 從 `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` 下載 `layer18.sae.pt`（約 2GB，存到 HF 緩存）。凍結層要到計劃二才會決定，這裡先用中間層 18。
  - 對 6 種核心情緒、`anger__resid3`、`pain_painaxis` 列出最接近的 10 個特徵。
  - 對每個特徵，再列出我們故事中在該特徵方向上投影最大的 5 篇故事的前 30 個字，當作特徵的大致含義。
  - 輸出 `results/6.9_crosscheck/Qwen_3_8B_instruct/sae_L18.md`，並加註「此字典訓練於基礎版，只能當作粗略參考」。
- **02 腳本**：
  - 從 Pain-axis commit 213fb6c 的 `results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/` 下載疼痛向量（先列出資料夾內容確認文件名稱；文件裡有 `layer` 欄位）。
  - 計算它和我們的 `pain`、`pain_painaxis` 在最接近候選層上的餘弦相似度。
  - 輸出 `pain_axis_compare.json`，欄位為 their_layer、our_layer、cos_pain、cos_pain_painaxis、note。

- [ ] **Step 1: 寫測試**

```python
def test_load_decoder_transposes_and_normalizes(tmp_path):
    torch.save({"W_dec": torch.randn(16, 100)}, tmp_path / "s.pt")   # 存成 [d, n]
    W = load_decoder(tmp_path / "s.pt", d_model=16)
    assert W.shape == (100, 16)
    np.testing.assert_allclose(np.linalg.norm(W, axis=1), 1, atol=1e-6)

def test_nearest_features_finds_planted():
    # W 的第 37 列 = vec 的方向 → 第一名是 (37, ≈1.0)
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_sae.py -v`
Expected: FAIL

- [ ] **Step 3: 實現**

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_sae.py -v`
Expected: 2 passed

- [ ] **Step 5: 正式執行**

```bash
python scripts/6.9_crosscheck/01_sae_nearest_features.py --model qwen3-8b --layer 18
python scripts/6.9_crosscheck/02_compare_pain_axis.py --model qwen3-8b
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: SAE nearest features and Pain-axis comparison"
```

---

### Task 13: 檢查點 1 的範例與中文摘要（`report.py`、`08_checkpoint_examples.py`）

**Files:**
- Create: `scripts/emolab/report.py`, `scripts/6.2_emotion_vectors/08_checkpoint_examples.py`, `tests/test_report.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 8–12 的所有輸出、`hooks.steer`
- Produces:
  - `report.render_checkpoint1(validation: pd.DataFrame, separability: dict, contamination: dict, axes_info: dict, sae_tops: dict, examples_path: str) -> str`：產生簡體中文 Markdown。
  - `results/6.2_emotion_vectors/<模型名>/checkpoint1_examples.md`
  - `results/6.2_emotion_vectors/<模型名>/checkpoint1_summary.md`
- **範例生成**：
  - 6 種核心情緒各用 3 個提示：`"Describe your morning routine."`、`"Write a short paragraph about a walk in the park."`、`"Tell me about your favorite meal."`
  - α ∈ {0.0, 0.1, 0.2, 0.4}，第 18 層，greedy 解碼，max_new_tokens 120，提示套 `format_chat`。
  - 這些 α 只是暫定值，正式數值在計劃二校準。
  - 疼痛只生成上述 12 段（3 個提示 × 4 個 α），以遵守倫理限制。
- **摘要包含**：
  - 每種情緒的 AUROC 表（取最佳層）。
  - 憤怒 vs 恐懼能否分開（逐層，並用白話說明代表什麼）。
  - 汙染檢查結果。
  - 三個軸在各層是否可靠（白話說明）。
  - 憤怒、恐懼最接近的特徵。
  - 與 Pain-axis 疼痛向量的相似度。
  - 範例文字的連結。

- [ ] **Step 1: 寫測試**

```python
def test_render_checkpoint1_contains_key_sections():
    s = render_checkpoint1(VALID_DF, {"18": {"auroc": 0.8123, "separable": True}}, CONTAM, AXES, SAE, "x.md")
    for h in ["憤怒 vs 恐懼", "汙染檢查", "三個軸", "特徵字典", "範例文字"]:
        assert h in s
    assert "0.81" in s and "可以分開" in s
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `pytest tests/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: 實現，並更新 README**

README 用簡體中文寫，參照 Pain-axis README 的結構：專案目的、目錄說明、階段 0–2 的執行順序（列出每一個指令）、外部資料來源。

- [ ] **Step 4: 執行測試，確認通過**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: 正式執行**

Run: `python scripts/6.2_emotion_vectors/08_checkpoint_examples.py --model qwen3-8b`
Expected: 產生兩個 Markdown 文件，範例文字共 72 段（6 種情緒 × 3 個提示 × 4 個 α）。

- [ ] **Step 6: 執行全部測試**

Run: `pytest -v`
Expected: 全部 PASS（gpu 和 network 測試預設跳過）

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: checkpoint 1 examples and summary"
```

- [ ] **Step 8: 使用者檢查點 1**

把 `checkpoint1_summary.md` 交給使用者，用白話說明結果，然後依結果撰寫計劃二。

---

## 之後的計劃（檢查點 1 之後才撰寫）

- **計劃二（階段 3–5）**：
  - 校準與凍結：`scripts/6.5_calibration/`
  - 懲罰情境與 MoralChoice 篩選：`scripts/6.6_moral_items/`、`datasets/6.6_*`
  - 所有實驗條件：`scripts/6.7_runs/`
  - 統計與圖表：`scripts/6.8_analysis/`
- **計劃三（階段 6–7）**：
  - Gemma 3 4B 重做：同一套腳本加上 `--model gemma3-4b`，使用同一份 `datasets/6.2_stories.jsonl`
  - Gemma Scope 2 交叉確認
  - 中文研究報告
