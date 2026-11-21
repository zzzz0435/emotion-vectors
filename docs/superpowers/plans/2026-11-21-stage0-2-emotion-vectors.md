# 计划一：环境、情绪向量与三个轴（阶段 0–2）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Qwen3-8B 上建立全部情绪、概念、污染向量和「正负面 / 激动程度 / 掌控感」三个轴，完成可分辨性、污染和特征字典的检查，产出「用户检查点 1」的材料。

**Architecture:** 目录结构仿照 GitHub 上星数最高的情绪相关项目 [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis)（76 星，commit `213fb6c`）：`datasets/` 存放输入数据，`scripts/<设计文件章节号>_<名称>/NN_<动作>.py` 按执行顺序编号，`results/<章节号>_<名称>/<模型名>/` 存放输出，模型名采用 Pain-axis 的写法（例如 `Qwen_3_8B_instruct`）。和 Pain-axis 不同的地方：可测试的逻辑统一放在 `scripts/emolab/` 共用模块里，编号脚本只负责命令行参数和文件读写。这样做是因为设计要求每个阶段的调整方式完全一致，Pain-axis 那种每个脚本各写一份的做法容易出现不一致。

**Tech Stack:** Python 3.12、PyTorch（CUDA 12.8 以上版本）、transformers、bitsandbytes、scikit-learn、pandas、matplotlib、pytest。不使用 transformer_lens，因为它对 4-bit 量化模型的支持不完整，改用原生 PyTorch forward hook。

**Spec:** `docs/superpowers/specs/2026-11-21-emotion-moral-steering-design.md`（已定案）

**本计划的范围**：设计文件的阶段 0、1、2，到用户检查点 1 为止。阶段 3–5（校准、道德题、分析）和阶段 6–7（Gemma 重做、报告）分别是计划二和计划三，等检查点 1 的结果出来后再写，因为它们的细节取决于这里的结果（实际速度、愤怒和恐惧能否分开、哪些轴可靠）。

## Global Constraints

- 所有命令都在项目根目录 `C:\Users\xgame\OneDrive\Desktop\emotional` 下执行，并先启用虚拟环境。
- 虚拟环境放在 OneDrive 之外：`C:\Users\xgame\.venvs\emotional`。
- 模型权重使用 Hugging Face 默认缓存；激活值等大型中间文件放在 `emolab.paths.cache_dir()`。该路径默认是 `C:\Users\xgame\.cache\emotional`，可以用环境变量 `EMOTIONAL_CACHE` 覆盖，**但不得位于项目目录内**。
- PyTorch 必须从 CUDA 12.8 以上的官方索引安装（RTX 5060 是 sm_120 架构）；transformers ≥ 4.51（支持 Qwen3）。安装完成后，用 `pip freeze` 把确切版本写进 `requirements.txt`。
- 模型登记（key → Hugging Face repo → 输出名 → 量化方式）：
  - `qwen3-8b` → `Qwen/Qwen3-8B` → `Qwen_3_8B_instruct` → 4-bit NF4（compute dtype 为 bfloat16，开启 double quant）
  - `qwen3-1.7b` → `Qwen/Qwen3-1.7B` → `Qwen_3_1.7B_instruct` → 不量化，bfloat16（先导用）
  - `gemma3-4b` → `google/gemma-3-4b-it` → `Gemma_3_4B_instruct` → 8-bit（只载入文字部分）
  - `gemma3-1b` → `google/gemma-3-1b-it` → `Gemma_3_1B_instruct` → 不量化，bfloat16
- Qwen3 一律关闭思考：`apply_chat_template(..., enable_thinking=False)`。
- **层号约定**：层 L 指第 L 个 decoder block 的输出，等于 transformer_lens 的 `blocks.L.hook_resid_post`，也等于 `output_hidden_states[L+1]`。
- **候选层**：`sorted({int(n_layers * f + 0.5) for f in (0.25, 0.40, 0.50, 0.65)})`。Qwen3-8B（36 层）→ [9, 14, 18, 23]。
- **取平均与调整时都排除每一行第一个非 padding 位置。**
- **调整量** = α × resid_norm[L] × 单位方向向量。resid_norm[L] 是该层在中性故事（训练集）上逐词元 L2 范数的平均值。
- 随机种子 42。生成故事使用 temperature 0.8、top_p 0.95、max_new_tokens 300。
- 所有给模型看的文字一律用英文；给用户看的 README、检查点摘要一律用简体中文。
- **伦理**：疼痛方向只用于短篇生成（本计划只有检查点 1 的 12 段范例），不做任何持续或互动式的施加。
- **愤怒 vs 恐惧可分辨的门槛**：验证集 AUROC ≥ 0.75。**轴可靠的门槛**：留一法相关系数 ≥ 0.5。
- Git 只在本机使用，不推送到远端。

## Review Focus

1. **长时间生成中途中断**（当机、睡眠、显存不足）：重跑时要从中断处继续，不能重复写入或遗漏故事。→ Task 5 `test_resume_skips_existing`
2. **显存不足（8GB）**：批次太大时要自动减半重试，而不是整个程序崩溃。→ Task 5 `test_oom_halves_batch`
3. **Qwen3 输出残留思考标签**（`<think>`）：这类故事必须被剔除。→ Task 2 `test_format_chat_qwen_disables_thinking`、Task 6 `test_flags_thinking_leak`
4. **故事以变形词泄漏情绪名称**（例如 "angrily"、"furious"），而且不能误判（例如 "made" 不能被当成 "mad"）。→ Task 6 `test_flags_emotion_inflections`
5. **4-bit 模型产生 NaN/Inf 激活值**：要立即报错并指出是哪一层，不能悄悄写进向量。→ Task 3 `test_nan_raises`

---

## 文件结构

```
emotional/
├── README.md                         # 简体中文：项目说明、目录结构、执行顺序（仿照 Pain-axis README）
├── requirements.txt                  # 安装后用 pip freeze 写入确切版本
├── pytest.ini                        # pythonpath = scripts；markers：gpu、network
├── .gitignore                        # 复制 Pain-axis 的 Python 模板，另加 .venv/
├── datasets/
│   ├── ATTRIBUTION.md                # 外部数据的来源与授权
│   ├── 6.2_emotions.json             # 20 种情绪 + 中性：提示描述、禁用词、限制条款
│   ├── 6.2_story_topics.json         # 50 个日常主题，其中 10 个标记为验证用（heldout）
│   ├── 6.2_concepts.json             # 40 个对照概念 + 2 个污染概念
│   ├── 6.2_pain_axis_3.1_pain_and_control_datasets.json   # Pain-axis 资料集（MIT），commit 213fb6c
│   ├── 6.2_stories.jsonl             # 由 Qwen3-8B 生成并过滤后的最终故事（Gemma 也会用）
│   └── 6.3_nrc_vad_emotions.csv      # 20 个情绪词的 V/A/D 评分（摘自 NRC VAD v2）
├── scripts/
│   ├── emolab/                       # 共用模块（可测试的逻辑都在这里）
│   │   ├── __init__.py
│   │   ├── paths.py                  # 目录、缓存位置
│   │   ├── models.py                 # 模型登记、载入、聊天格式、取得层
│   │   ├── batching.py               # 显存不足时自动减半批次
│   │   ├── hooks.py                  # 读取激活值、调整（steering）
│   │   ├── stories.py                # 生成任务、提示模板、过滤、平衡挑选
│   │   ├── vectors.py                # 对比平均、Pain-axis 去噪、AUROC、可分辨性、污染
│   │   ├── axes.py                   # 三个轴的拟合、留一法验证、残差
│   │   ├── sae.py                    # 特征字典解码器读取、最近特征
│   │   └── report.py                 # 检查点 1 中文摘要
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
    ├── conftest.py                   # tiny_qwen3、tiny_gemma3 fixture（用 config 建立随机小模型，不需下载）
    └── test_*.py
```

每个编号脚本都以 docstring 开头，说明读取什么、写出什么、需要什么（Pain-axis 的写法），并用以下两行引入共用模块：

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

每个编号脚本都接受参数 `--model {qwen3-8b,qwen3-1.7b,gemma3-4b,gemma3-1b}`；会处理大量资料的脚本另外接受 `--limit N`，供先导测试使用。

---

### Task 1: 环境与项目骨架

**Files:**
- Create: `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `scripts/emolab/__init__.py`, `scripts/6.0_env/01_check_environment.py`, `tests/test_env.py`

**Interfaces:**
- Produces: 可用的虚拟环境；`results/6.0_env/env_report.json`，包含 `gpu`、`vram_total_mb`、`torch`、`cuda`、`transformers`、`bitsandbytes`，以及 `models: {key: {"loaded": bool, "peak_vram_mb": float, "tokens_per_s": float, "error": str|None}}`

- [ ] **Step 1: 确认磁盘空间**

Run: `powershell -c "(Get-PSDrive C).Free/1GB"`
Expected: ≥ 50（整个研究约需下载 Qwen3-8B 16GB、Qwen3-1.7B 3.4GB、Gemma 3 4B 8.6GB、Gemma 3 1B 2GB、SAE 每层 2GB）。不足就停下来回报用户。

- [ ] **Step 2: 建立虚拟环境并安装套件**

```bash
python -m venv /c/Users/xgame/.venvs/emotional
source /c/Users/xgame/.venvs/emotional/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes safetensors huggingface_hub scikit-learn scipy pandas numpy matplotlib tqdm pytest
pip freeze > requirements.txt
git init && git add -A && git commit -m "chore: project skeleton and design docs"
```

`.gitignore` 先复制 Pain-axis 的版本（commit 213fb6c 的 `.gitignore`），再加上 `.venv/`。`pytest.ini` 的内容：`[pytest]`、`pythonpath = scripts`、`addopts = -m "not gpu and not network"`，并登记 `gpu` 和 `network` 两个 marker。

- [ ] **Step 3: 写 GPU 测试**

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

- [ ] **Step 4: 执行 GPU 测试**

Run: `pytest tests/test_env.py -m gpu -v`
Expected: 2 passed。失败时，记下错误并换用较新的 CUDA 索引或 bitsandbytes 版本；仍然失败就回报用户。

- [ ] **Step 5: 实现 `01_check_environment.py`**

参数 `--models qwen3-1.7b qwen3-8b [gemma3-1b gemma3-4b]`。依序载入每个模型（模型载入在 Task 2 完成后才能用；这一步先只写 GPU 和版本资讯，模型部分在 Task 2 Step 6 补跑），量测 `torch.cuda.max_memory_allocated()`，以及用 64 个新词元算出 tokens/s，最后写入 `env_report.json`。Gemma 载入失败（未登录）时，`error` 栏写「需要用户登录 Hugging Face 并同意 Gemma 条款」，不中断其他模型。

- [ ] **Step 6: 用户操作（可以延后到计划三）**

请用户自己执行 `hf auth login`，并在浏览器中同意 `google/gemma-3-4b-it` 和 `google/gemma-3-1b-it` 的使用条款。Claude 不代为输入任何 token。

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: environment check and test config"
```

---

### Task 2: 路径与模型载入（`paths.py`、`models.py`）

**Files:**
- Create: `scripts/emolab/paths.py`, `scripts/emolab/models.py`, `tests/conftest.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `paths.REPO_ROOT: Path`, `paths.DATASETS: Path`, `paths.RESULTS: Path`
  - `paths.cache_dir() -> Path`：建立资料夹并回传；如果位于 `REPO_ROOT` 之内，抛出 `ValueError`
  - `paths.results_dir(section: str, model_name: str | None = None) -> Path`：例如 `results_dir("6.2_emotion_vectors", "Qwen_3_8B_instruct")`，会建立资料夹
  - `models.ModelSpec(key: str, repo_id: str, name: str, quant: Literal["4bit", "8bit", "none"])`（frozen dataclass）
  - `models.MODELS: dict[str, ModelSpec]`，四个 key 见 Global Constraints
  - `models.load(key: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`：eval 模式、`device_map={"": 0}`、`tokenizer.padding_side = "left"`；没有 pad token 时，用 eos 作为 pad token。`gemma3-*` 用 `Gemma3ForCausalLM`（只载入文字部分）。
  - `models.get_layers(model) -> torch.nn.ModuleList`：依序尝试 `model.model.layers`、`model.model.language_model.layers`、`model.language_model.model.layers`
  - `models.candidate_layers(n_layers: int) -> list[int]`
  - `models.format_chat(tokenizer, user_msg: str, system_msg: str | None = None) -> str`

- [ ] **Step 1: 写测试**

`conftest.py` 提供 `tiny_qwen3` fixture：`Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8))`，float32、eval、固定种子。同样方式提供 `tiny_gemma3`（`Gemma3ForCausalLM` + `Gemma3TextConfig`）。

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

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_models.py -v -m "not gpu"`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 `paths.py` 和 `models.py`**

`format_chat` 调用 `tokenizer.apply_chat_template([...], tokenize=False, add_generation_prompt=True, enable_thinking=False)`；如果 tokenizer 不接受 `enable_thinking`，就不传这个参数。

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_models.py -v -m "not gpu"`，以及 `pytest tests/test_models.py -v -m network`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: model registry, loading and paths"
```

- [ ] **Step 6: 补跑环境检查的模型部分**

Run: `python scripts/6.0_env/01_check_environment.py --models qwen3-1.7b qwen3-8b`
Expected: 两个模型的 `loaded` 都是 true，`qwen3-8b` 的 `peak_vram_mb` < 7500。如果放不下，停下来回报用户（设计文件 6.1 有备案）。把 `env_report.json` 一起 commit。

---

### Task 3: 读取激活值与调整（`hooks.py`）

**Files:**
- Create: `scripts/emolab/hooks.py`, `tests/test_hooks.py`

**Interfaces:**
- Consumes: `models.get_layers`
- Produces:
  - `capture_mean_activations(model, input_ids: LongTensor[B,T], attention_mask: LongTensor[B,T], layers: list[int]) -> dict[int, Tensor[B, d]]`：float32，在 CPU 上；对每一行中 `attention_mask==1` 的位置取平均，排除第一个非 padding 位置。只要出现 NaN 或 Inf，就抛出 `FloatingPointError`，讯息包含层号。
  - `mean_token_norm(model, input_ids, attention_mask, layers) -> dict[int, float]`：同样的位置规则，回传逐词元 L2 范数的平均值。
  - `steer(model, layer: int, direction: Tensor[d], alpha: float, resid_norm: float) -> ContextManager[None]`：在 layer 的输出加上 `alpha * resid_norm * direction / ||direction||`。prefill 阶段（序列长度 > 1）跳过第一个非 padding 位置；逐字生成阶段（序列长度 == 1）一律加上。离开 context 时移除 hook；`alpha == 0` 时不注册 hook。

- [ ] **Step 1: 写测试**

```python
def test_capture_matches_hidden_states(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 7)); mask = torch.ones_like(ids)
    got = capture_mean_activations(tiny_qwen3, ids, mask, [1])[1]
    hs = tiny_qwen3(ids, attention_mask=mask, output_hidden_states=True).hidden_states[2]
    torch.testing.assert_close(got[0], hs[0, 1:].mean(0), atol=1e-5, rtol=1e-4)

def test_capture_respects_left_padding(tiny_qwen3):
    # 同一序列单独输入 vs 左侧补 3 个 pad 后批次输入 → 平均值相同（atol 1e-4）

def test_steer_adds_scaled_unit_vector(tiny_qwen3):
    # direction = 3*e0, alpha=0.5, resid_norm=2.0 → 第 1 层输出在位置 ≥1 增加 1.0*e0，位置 0 不变

def test_steer_zero_alpha_registers_nothing(tiny_qwen3): ...
def test_steer_hook_removed_after_exit(tiny_qwen3): ...   # 离开 context 后输出与原本完全相同

def test_nan_raises(tiny_qwen3, monkeypatch):
    # 用 forward hook 让第 2 层输出 NaN → capture_mean_activations 抛出 FloatingPointError，讯息含 "layer 2"
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_hooks.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 `hooks.py`**

两个函数都在 `get_layers(model)[L]` 上注册 `register_forward_hook`。decoder layer 的输出可能是 tuple，只修改第一个元素。

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_hooks.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: activation capture and steering hooks"
```

---

### Task 4: 输入资料集

**Files:**
- Create: `datasets/6.2_emotions.json`, `datasets/6.2_story_topics.json`, `datasets/6.2_concepts.json`, `datasets/6.2_pain_axis_3.1_pain_and_control_datasets.json`, `datasets/6.3_nrc_vad_emotions.csv`, `datasets/ATTRIBUTION.md`, `tests/test_datasets.py`

**Interfaces:**
- Produces（schema，后面的 Task 都依赖这些栏位）：
  - `6.2_emotions.json`：list，每个元素是 `{"label": str, "group": "core"|"map"|"neutral", "description": str, "banned": [regex, ...], "restriction": str|null, "subtypes": [str, ...]|null}`
  - `6.2_story_topics.json`：list，每个元素是 `{"idx": int, "topic": str, "heldout": bool, "concept_topic": bool}`
  - `6.2_concepts.json`：list，每个元素是 `{"label": str, "role": "control"|"contamination", "description": str}`
  - `6.3_nrc_vad_emotions.csv`：栏位 `label,word,valence,arousal,dominance,lexicon_version`

- [ ] **Step 1: 写测试**

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

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL（找不到文件）

- [ ] **Step 3: 写资料**

- **主题**：50 个情绪中性的日常主题（例如 "waiting for a delayed train"、"repairing a bicycle"、"cooking dinner for one"）。不能涉及任何人做错事。
- **限制条款**：一般情绪的 `restriction` 为 null，使用通用条款。以下情绪另订条款：
  - guilt、shame：`"The story may involve a minor personal slip that hurts no one, but must not involve blame from others, punishment, revenge, harm to other people, or moral decisions. "`
  - jealousy、contempt：`"The feeling is directed at another person's luck, success or habits, but nobody does anything wrong. The story must not involve blame, punishment, revenge, harm to other people, or moral decisions. "`
- **愤怒和恐惧的 description** 要符合设计 6.2：愤怒针对事物或情境，恐惧针对危险情境，都不能针对某个人的恶意。
- **疼痛的 4 个 subtype**：`physical pain from an accidental injury or an illness`、`deep inner psychological pain`、`the pain of loneliness and feeling left out`、`the pain of an exhausting mental struggle`。
- **banned**：每种情绪约 5–10 个 regex，例如愤怒 `[r"\bangr\w*", r"\banger\w*", r"\bfurious\w*", r"\bfury\b", r"\brage\w*", r"\benrag\w*", r"\bmad\b", r"\birritat\w*"]`。
- **概念**：40 个对照概念（设计 6.7 的清单）。污染概念的 description：punishment → `"someone being punished for something they did"`；harming_others → `"one person physically harming another person"`。
- **Pain-axis 资料集**：从 `https://raw.githubusercontent.com/valen-research/Pain-axis/213fb6c89c96004dbe93d1704c636ad8c9d801f3/datasets/3.1_pain_and_control_datasets.json` 下载，原样保存。
- **NRC VAD**：从官方页面下载 NRC VAD Lexicon v2，确认授权允许研究使用；只摘录 20 个情绪词的评分写进 CSV，不要转存整个词表。如果条款不允许摘录，就停下来回报用户。
- **ATTRIBUTION.md**：写明 Pain-axis（MIT，commit 213fb6c）、NRC VAD（版本和授权条款）、MoralChoice（CC-BY-4.0，计划二才会用到）。

- [ ] **Step 4: 执行测试，确认通过**

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
  - `batching.run_with_oom_backoff(fn: Callable[[list[T]], list[R]], items: list[T], batch_size: int) -> Iterator[tuple[list[T], list[R]]]`：遇到 `torch.cuda.OutOfMemoryError` 时，执行 `torch.cuda.empty_cache()`，批次减半后重试；批次已经是 1 还失败，就把错误往上抛。
  - `stories.StoryJob`（frozen dataclass）：`id: str, label: str, kind: Literal["emotion","neutral","concept","contamination"], topic_idx: int, topic: str, subtype: str | None, replicate: int, prompt: str`
  - `stories.build_jobs(emotions: list[dict], topics: list[dict], concepts: list[dict]) -> list[StoryJob]`
  - `stories.build_topup_jobs(shortfall: dict[tuple[str, int, int], int], emotions, topics, concepts, replicate_offset: int) -> list[StoryJob]`：为每个缺额格子 `(label, topic_idx, subtype_idx)` 产生对应数量的新任务。新任务沿用该格子原本的 subtype，replicate 编号从 `replicate_offset` 开始。
  - `stories.run_generation(jobs: list[StoryJob], out_path: Path, generate_fn: Callable[[list[str]], list[str]], batch_size: int, model_name: str) -> int`：以 JSONL 逐行追加写入，每行包含 `id, label, kind, topic_idx, topic, subtype, replicate, text, model`；已经存在的 id 会跳过；回传这次新写入的笔数。

**数量规则**（每个 id 是 `f"{label}__t{topic_idx:02d}__s{subtype_idx}__r{replicate}"`，subtype_idx 在没有 subtype 时是 0）：

| 种类 | 主题 | 每个主题的篇数 | 合计 |
|---|---|---|---|
| 核心情绪（除疼痛外） | 全部 50 个 | 2 | 100 |
| 疼痛 | 全部 50 个；4 个 subtype 依 `(topic_idx * 2 + replicate) % 4` 轮流分配 | 2 | 100（每个 subtype 25） |
| 地图情绪 | 全部 50 个 | 1 | 50 |
| 中性 | 全部 50 个 | 2 | 100 |
| 对照概念、污染概念 | `concept_topic` 为 true 的 20 个 | 2 | 每个 40 |

**提示模板**（确切文字）：

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

疼痛的 `{description}` 使用它的 subtype 文字。`{restriction}` 优先使用该情绪自己的 `restriction`，否则使用 `GENERAL_RESTRICTION`。

- [ ] **Step 1: 写测试**

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
    # 第一次执行时，fake generate_fn 在第 5 次调用抛出 KeyboardInterrupt；第二次正常执行
    # → 文件中每个 id 只出现一次，总数 == len(jobs)

def test_oom_halves_batch(tmp_path):
    # 当 len(prompts) > 2 时，fake generate_fn 抛出 torch.cuda.OutOfMemoryError
    # → 以 batch_size=8 执行完毕，所有 id 都已写入
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_stories_generate.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `batching.py` 和 `stories.py` 的生成部分，以及脚本 `01_generate_stories.py`**

- 脚本参数：`--model`、`--out`（默认为 `results/6.2_emotion_vectors/<模型名>/stories_raw.jsonl`）、`--batch-size 16`、`--limit`、`--top-up`。`--top-up` 会读取 Task 6 写出的 `shortfall.json`（格子 → 缺额），以 `build_topup_jobs` 产生任务，`replicate_offset` 设为现有最大 replicate + 1。
- `generate_fn`：对每个 prompt 先套 `format_chat`，左侧补齐 padding，执行 `model.generate(do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=300)`，每个批次之前执行 `torch.manual_seed(42 + 批次序号)`，只解码新生成的词元。

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_stories_generate.py -v`
Expected: 6 passed

- [ ] **Step 5: 先导执行**

Run: `python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-1.7b --limit 40`
Expected: 写入 40 行；人工打开 3 行确认内容是故事、没有 `<think>`。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: story generation with resume and OOM backoff"
```

- [ ] **Step 7: 正式执行（背景执行，预计约 1 小时）**

Run: `python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b`
Expected: 写入约 3,080 行。中途中断就用同一个命令重跑。

---

### Task 6: 过滤故事与抽查（`stories.py` 后半、`02_filter_stories.py`）

**Files:**
- Modify: `scripts/emolab/stories.py`
- Create: `scripts/6.2_emotion_vectors/02_filter_stories.py`, `tests/test_stories_filter.py`

**Interfaces:**
- Produces:
  - `stories.CONTAMINATION_PATTERNS: list[str]` = `[r"\bpunish\w*", r"\brevenge\w*", r"\baveng\w*", r"\bretaliat\w*", r"\bblam\w*", r"\bguilty\b", r"\bcrim\w*", r"\bsteal\w*", r"\bstole\w*", r"\btheft\w*", r"\bassault\w*", r"\bkill\w*", r"\bmurder\w*", r"\babus\w*", r"\bcheat\w*", r"\bbetray\w*", r"\bjail\w*", r"\bprison\w*", r"\barrest\w*"]`
  - `stories.check_story(text: str, banned: list[str], kind: str, min_words: int = 80, max_words: int = 300) -> list[str]`：回传不通过的原因，可能是 `"thinking_leak"`、`"names_emotion"`、`"contamination"`（只检查 kind ≠ contamination 的故事）、`"too_short"`、`"too_long"`。比对时不分大小写。
  - `stories.select_balanced(rows: list[dict], targets: dict[tuple[str, int, int], int]) -> tuple[list[dict], dict[tuple, int]]`：每个 `(label, topic_idx, subtype_idx)` 格子依 replicate 顺序取前 N 篇通过的故事；回传 `(选中的故事, 缺额)`。
- 输出：
  - `datasets/6.2_stories.jsonl`：只在 `--model qwen3-8b` 时写入；会多一个 `heldout` 栏位，取自主题档。
  - `results/6.2_emotion_vectors/<模型名>/filter_report.csv`：栏位为 label、n_raw、各原因的剔除数、n_kept、shortfall。
  - `results/6.2_emotion_vectors/<模型名>/shortfall.json`：`[{"label", "topic_idx", "subtype_idx", "missing"}]`，供 `--top-up` 使用。
  - `results/6.2_emotion_vectors/<模型名>/review_samples.md`：中文标题；每种核心情绪随机抽 5 篇，每个污染概念抽 2 篇，种子 42。

- [ ] **Step 1: 写测试**

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
    # 格子 ("anger", 0, 0) 目标 2 篇，只有 1 篇通过 → 缺额 {("anger", 0, 0): 1}
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_stories_filter.py -v`
Expected: FAIL

- [ ] **Step 3: 实现过滤逻辑与脚本**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_stories_filter.py -v`
Expected: 5 passed

- [ ] **Step 5: 先导执行，再正式执行；有缺额就补生成**

```bash
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-1.7b
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b
# 如果 filter_report.csv 中有 shortfall > 0：
python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b --top-up
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b
```

Expected: shortfall 全部为 0。补生成两轮后仍有缺额，就在报告中记录并继续（地图情绪可以接受少于 50 篇，核心情绪不行）。

- [ ] **Step 6: Commit**（包含 `datasets/6.2_stories.jsonl`、`filter_report.csv`、`review_samples.md`）

```bash
git add -A && git commit -m "feat: story filtering; final story dataset"
```

---

### Task 7: 读取激活值（`03_extract_activations.py`）

**Files:**
- Create: `scripts/6.2_emotion_vectors/03_extract_activations.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `hooks.capture_mean_activations`、`hooks.mean_token_norm`、`batching.run_with_oom_backoff`、`paths.cache_dir`
- Produces:
  - 在 `03_extract_activations.py` 中定义 `extract(model, encode_fn: Callable[[list[str]], dict], rows: list[dict], layers: list[int], batch_size: int, out_dir: Path) -> pd.DataFrame`；测试用 `importlib` 依文件路径载入这个脚本。
  - 每层写一个 `<cache_dir>/6.2/<模型名>/acts_L{L}.npy`（float32，形状 [N, d]，行顺序与 index 相同）。
  - `<cache_dir>/6.2/<模型名>/index.csv`，栏位为 row、id、label、kind、topic_idx、subtype、heldout。
  - `results/6.2_emotion_vectors/<模型名>/resid_norms.json`：`{"<L>": float}`，只用中性、非 heldout 的故事计算。
- 脚本参数：`--model`、`--stories`（默认为 `datasets/6.2_stories.jsonl`）、`--batch-size 16`、`--limit`、`--force`。
- 输入文字直接用故事原文，不套聊天格式；`add_special_tokens=True`。依长度排序后组成批次，以减少 padding。
- 已经存在、而且和 index 的 id 一致的文件会被跳过；加 `--force` 可以重算。

- [ ] **Step 1: 写测试**（用 `tiny_qwen3` 和一个把字元映射成 id 的 fake `encode_fn`）

```python
def test_extract_shapes_and_order(tiny_qwen3, tmp_path):
    idx = extract(tiny_qwen3, fake_encode, ROWS_5, [1, 2], batch_size=2, out_dir=tmp_path)
    a = np.load(tmp_path / "acts_L1.npy")
    assert a.shape == (5, 32) and list(idx.id) == [r["id"] for r in ROWS_5]

def test_extract_skips_when_done(tiny_qwen3, tmp_path):
    # 执行两次；第二次时让 capture 被呼叫就抛错 → 仍然正常回传（证明有跳过）
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_extract.py -v`
Expected: 2 passed

- [ ] **Step 5: 先导执行，再正式执行**

```bash
python scripts/6.2_emotion_vectors/03_extract_activations.py --model qwen3-1.7b --stories results/6.2_emotion_vectors/Qwen_3_1.7B_instruct/stories_raw.jsonl --limit 40
python scripts/6.2_emotion_vectors/03_extract_activations.py --model qwen3-8b
```

Expected: 第 9、14、18、23 层的 npy 都已产生；`resid_norms.json` 有 4 个数值。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: activation extraction"
```

---

### Task 8: 建立向量（`vectors.py` 前半、`04_build_vectors.py`、`05_pain_axis_vector.py`）

**Files:**
- Create: `scripts/emolab/vectors.py`, `scripts/6.2_emotion_vectors/04_build_vectors.py`, `scripts/6.2_emotion_vectors/05_pain_axis_vector.py`, `tests/test_vectors_build.py`

**Interfaces:**
- Consumes: Task 7 的输出
- Produces:
  - `vectors.diff_of_means(acts: np.ndarray, labels: np.ndarray, target: str, baseline: str, mask: np.ndarray) -> np.ndarray`：只用 `mask` 为 True 的行。
  - `vectors.pain_axis_vector(pain_acts: np.ndarray, control_acts: np.ndarray, denoise_variance: float = 0.5) -> np.ndarray`：移植自 Pain-axis commit 213fb6c 中 `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py` 的 `compute_pain_vector`（第 151–185 行，`baseline="all_controls"`、`denoise=True`）。docstring 要注明出处与 MIT 授权。
  - `results/6.2_emotion_vectors/<模型名>/vectors.pt`：`{"layers": list[int], "resid_norm": {L: float}, "raw": {label: Tensor[n_layers, d]}, "unit": {label: Tensor[n_layers, d]}, "meta": {"model": str, "n_train": {label: int}, "heldout_topics": list[int]}}`。label 包括 20 种情绪、40 个对照概念、2 个污染概念；Task 8 的 05 脚本会再加入 `pain_painaxis`。
- **建立方式**：所有向量都是「该类训练集平均 − 中性训练集平均」。训练集 = 主题 `heldout == False` 的故事。
- **05 脚本**：使用 Pain-axis 资料集的 `S2_1P`。疼痛类别取 A1、A2、A3、A5，排除 A4（Moral Injury）；对照类别取 B、C1、C2、D、E。输入句子原文，以同一套 `capture_mean_activations` 读取候选层的激活值；结果以 `pain_painaxis` 写回 `vectors.pt`。

- [ ] **Step 1: 写测试**

```python
def test_diff_of_means_simple():
    acts = np.array([[1., 0], [1, 0], [0, 0], [0, 0]]); labels = np.array(["a", "a", "n", "n"])
    np.testing.assert_allclose(diff_of_means(acts, labels, "a", "n", np.ones(4, bool)), [1, 0])

def test_mask_excludes_heldout():
    # heldout 行设成极端值，mask 把它排除 → 结果不受影响

def test_pain_axis_denoise_removes_dominant_control_direction():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=(200, 4)) * [10, 1, 1, 1]          # 对照组在 e0 上变异极大
    pain = ctrl[:50] + [3, 2, 0, 0]                           # 疼痛组在 e0、e1 上都有偏移
    v = pain_axis_vector(pain, ctrl, 0.5)
    assert abs(v[0]) < 0.1 * abs(v[1])
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_vectors_build.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_vectors_build.py -v`
Expected: 3 passed

- [ ] **Step 5: 正式执行**

```bash
python scripts/6.2_emotion_vectors/04_build_vectors.py --model qwen3-8b
python scripts/6.2_emotion_vectors/05_pain_axis_vector.py --model qwen3-8b
```

Expected: `vectors.pt` 包含 63 个 label（20 + 40 + 2 + `pain_painaxis`），每个都是 4 层。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: emotion, concept and Pain-axis vectors"
```

---

### Task 9: 验证向量（`vectors.py`、`06_validate_vectors.py`）

**Files:**
- Modify: `scripts/emolab/vectors.py`
- Create: `scripts/6.2_emotion_vectors/06_validate_vectors.py`, `tests/test_vectors_validate.py`

**Interfaces:**
- Produces:
  - `vectors.projection_auroc(acts, labels, unit_vec, pos: str, neg: str, mask) -> float`
  - `vectors.separability(acts, labels, train_mask, test_mask, a: str = "anger", b: str = "fear") -> float`：用训练集算出方向 `unit(mean_a − mean_b)`，在测试集上计算 a 对 b 的 AUROC。
  - `SEPARABLE_THRESHOLD = 0.75`
- 输出：
  - `validation.csv`：栏位 label、layer、auroc_vs_neutral，涵盖全部 20 种情绪和 `pain_painaxis`，只用 heldout 资料。
  - `separability.json`：`{"<L>": {"auroc": float, "separable": bool}}`
  - `similarity_L{L}.csv` 和 `similarity_L{L}.png`：20 种情绪加 `pain_painaxis` 的余弦相似度热图。

- [ ] **Step 1: 写测试**

```python
def test_separability_high_when_distinct():   # 两群在 e0 上相距 5 → AUROC > 0.95
def test_separability_chance_when_identical(): # 同一分布 → 0.35 < AUROC < 0.65，且 < SEPARABLE_THRESHOLD
def test_projection_auroc_direction():         # pos 群在单位向量方向上较大 → AUROC > 0.9
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_vectors_validate.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_vectors_validate.py -v`
Expected: 3 passed

- [ ] **Step 5: 正式执行**

Run: `python scripts/6.2_emotion_vectors/06_validate_vectors.py --model qwen3-8b`
Expected: 4 个输出都已产生。如果某种核心情绪在所有层的 auroc_vs_neutral 都低于 0.7，在 Task 13 的摘要中标示出来。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: vector validation and anger-fear separability"
```

---

### Task 10: 污染检查（`vectors.py`、`07_contamination.py`）

**Files:**
- Modify: `scripts/emolab/vectors.py`
- Create: `scripts/6.2_emotion_vectors/07_contamination.py`, `tests/test_contamination.py`

**Interfaces:**
- Produces:
  - `vectors.contamination_flags(unit: dict[str, np.ndarray], emotions: list[str], concepts: list[str], probe: str, reference: str = "sadness") -> dict[str, dict]`：对每种情绪回传 `{"cos": float, "ref_cos": float, "p95_concepts": float, "flagged": bool}`。flagged 的条件是 `cos > ref_cos` 且 `cos > p95_concepts`。
  - `vectors.remove_component(v: np.ndarray, probe: np.ndarray) -> np.ndarray`：回传去掉 probe 成分后的单位向量。
- 两个 probe 都要检查：`punishment` 和 `harming_others`，逐层计算。被标示的情绪，依序去掉所有被标示的 probe 成分，得到 `<label>__clean`。
- 输出：`contamination.json`（逐层、逐 probe、逐情绪）和 `vectors_clean.pt`（只包含 `__clean` 向量，格式与 `vectors.pt["unit"]` 相同）。

- [ ] **Step 1: 写测试**

```python
def test_flags_contaminated_and_cleans():
    # anger = 0.8*p + 0.6*u；sadness 与 joy 和 p 正交；40 个概念在 p 上的余弦都 < 0.1
    # → anger 被标示、joy 没有；remove_component 后与 p 的 |cos| < 1e-6，而且长度为 1
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_contamination.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_contamination.py -v`
Expected: PASS

- [ ] **Step 5: 正式执行**

Run: `python scripts/6.2_emotion_vectors/07_contamination.py --model qwen3-8b`
Expected: 产生 `contamination.json`；只有在有情绪被标示时才会产生 `vectors_clean.pt`。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: contamination check and clean vectors"
```

---

### Task 11: 三个轴与残差向量（`axes.py`、`6.3_axes/`）

**Files:**
- Create: `scripts/emolab/axes.py`, `scripts/6.3_axes/01_fit_axes.py`, `scripts/6.3_axes/02_residual_vectors.py`, `tests/test_axes.py`

**Interfaces:**
- Consumes: `vectors.pt["unit"]`、`datasets/6.3_nrc_vad_emotions.csv`
- Produces:
  - `axes.ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]`、`axes.RELIABLE_R = 0.5`
  - `axes.fit_axis(X: np.ndarray[n, d], y: np.ndarray[n]) -> tuple[np.ndarray, float]`：用 `RidgeCV(alphas=ALPHAS, fit_intercept=True)`，回传单位化的 `coef_` 和选中的 alpha。
  - `axes.loo_r(X, y) -> float`：每次留下一个样本不参与拟合，用 `fit_axis` 的完整流程（包括截距）预测它的分数，最后计算所有预测值与真实值的 Pearson r。
  - `axes.fit_axes(unit: dict[str, np.ndarray], vad: pd.DataFrame, exclude: set[str]) -> dict`：回传 `{"V"|"A"|"D": {"axis": ndarray, "alpha": float, "loo_r": float, "reliable": bool}}`。三个轴依 V → A → D 的顺序做 Gram–Schmidt 正交化。
  - `axes.residual(v: np.ndarray, axes: list[np.ndarray]) -> np.ndarray`：回传去掉所有给定轴成分后的单位向量。
- **拟合组**（每一层都要做）：`all`（不排除）、`for_anger`（排除 anger、fear）、`for_pain`（排除 pain）、`for_compassion`（排除 compassion）。
- **残差向量**：
  - `anger__resid3`：用 `for_anger` 中可靠的轴。
  - `anger__resid2`：只用 V、A。
  - `pain__resid3`：用 `for_pain` 中可靠的轴。
  - `compassion__resid3`：用 `for_compassion` 中可靠的轴。
  - 如果该组的 V 轴不可靠，就不产生这个残差，并在 `axes.json` 中记录 `"untestable"`。
- 输出：
  - `results/6.3_axes/<模型名>/axes.pt`：`{"layers": [...], "fits": {fit_set: {L: {"V"|"A"|"D": Tensor[d]}}}}`。其中 `all` 组的三个轴，在计划二中会用作调整方向 `axis_V`、`axis_A`、`axis_D`。
  - `axes.json`：每组、每层、每轴的 alpha、loo_r、reliable。
  - `residuals.pt`：格式与 `vectors.pt["unit"]` 相同。

- [ ] **Step 1: 写测试**

```python
def test_recovers_planted_axis():
    rng = np.random.default_rng(0); a = unit(rng.normal(size=64))
    y = rng.uniform(-1, 1, 20); X = np.outer(y, a) + 0.05 * rng.normal(size=(20, 64))
    axis, _ = fit_axis(X, y); assert abs(axis @ a) > 0.9
    assert loo_r(X, y) > 0.8

def test_random_ratings_unreliable():
    # X 为随机向量，y 为与 X 无关的随机评分 → loo_r < RELIABLE_R

def test_exclude_really_excludes():
    # 把被排除的 label 换成垃圾向量 → 拟合出的轴完全相同

def test_residual_orthogonal_and_unit():
    # 对每个轴的 |residual @ axis| < 1e-6，而且 ||residual|| == 1
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_axes.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_axes.py -v`
Expected: 4 passed

- [ ] **Step 5: 正式执行**

```bash
python scripts/6.3_axes/01_fit_axes.py --model qwen3-8b
python scripts/6.3_axes/02_residual_vectors.py --model qwen3-8b
```

Expected: 产生 `axes.json`，并且至少有一层的 V 轴可靠。如果所有层的 V 轴都不可靠，在 Task 13 的摘要中标示「H4 无法检验」。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: VAD axes, LOO validation and residual vectors"
```

---

### Task 12: 交叉确认（`sae.py`、`6.9_crosscheck/`）

**Files:**
- Create: `scripts/emolab/sae.py`, `scripts/6.9_crosscheck/01_sae_nearest_features.py`, `scripts/6.9_crosscheck/02_compare_pain_axis.py`, `tests/test_sae.py`

**Interfaces:**
- Produces:
  - `sae.load_decoder(path: Path, d_model: int) -> np.ndarray`：回传 [n_features, d_model] 的矩阵，每一列都单位化。如果矩阵存成 [d_model, n] 就转置。key 名称请先实际打开文件确认，常见的是 `W_dec`。
  - `sae.nearest_features(vec: np.ndarray, W_dec: np.ndarray, k: int = 10) -> list[tuple[int, float]]`：依余弦由高到低排序。
- **01 脚本**：
  - 从 `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` 下载 `layer18.sae.pt`（约 2GB，存到 HF 缓存）。冻结层要到计划二才会决定，这里先用中间层 18。
  - 对 6 种核心情绪、`anger__resid3`、`pain_painaxis` 列出最接近的 10 个特征。
  - 对每个特征，再列出我们故事中在该特征方向上投影最大的 5 篇故事的前 30 个字，当作特征的大致含义。
  - 输出 `results/6.9_crosscheck/Qwen_3_8B_instruct/sae_L18.md`，并加注「此字典训练于基础版，只能当作粗略参考」。
- **02 脚本**：
  - 从 Pain-axis commit 213fb6c 的 `results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/` 下载疼痛向量（先列出资料夹内容确认文件名称；文件里有 `layer` 栏位）。
  - 计算它和我们的 `pain`、`pain_painaxis` 在最接近候选层上的余弦相似度。
  - 输出 `pain_axis_compare.json`，栏位为 their_layer、our_layer、cos_pain、cos_pain_painaxis、note。

- [ ] **Step 1: 写测试**

```python
def test_load_decoder_transposes_and_normalizes(tmp_path):
    torch.save({"W_dec": torch.randn(16, 100)}, tmp_path / "s.pt")   # 存成 [d, n]
    W = load_decoder(tmp_path / "s.pt", d_model=16)
    assert W.shape == (100, 16)
    np.testing.assert_allclose(np.linalg.norm(W, axis=1), 1, atol=1e-6)

def test_nearest_features_finds_planted():
    # W 的第 37 列 = vec 的方向 → 第一名是 (37, ≈1.0)
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_sae.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_sae.py -v`
Expected: 2 passed

- [ ] **Step 5: 正式执行**

```bash
python scripts/6.9_crosscheck/01_sae_nearest_features.py --model qwen3-8b --layer 18
python scripts/6.9_crosscheck/02_compare_pain_axis.py --model qwen3-8b
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: SAE nearest features and Pain-axis comparison"
```

---

### Task 13: 检查点 1 的范例与中文摘要（`report.py`、`08_checkpoint_examples.py`）

**Files:**
- Create: `scripts/emolab/report.py`, `scripts/6.2_emotion_vectors/08_checkpoint_examples.py`, `tests/test_report.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 8–12 的所有输出、`hooks.steer`
- Produces:
  - `report.render_checkpoint1(validation: pd.DataFrame, separability: dict, contamination: dict, axes_info: dict, sae_tops: dict, examples_path: str) -> str`：产生简体中文 Markdown。
  - `results/6.2_emotion_vectors/<模型名>/checkpoint1_examples.md`
  - `results/6.2_emotion_vectors/<模型名>/checkpoint1_summary.md`
- **范例生成**：
  - 6 种核心情绪各用 3 个提示：`"Describe your morning routine."`、`"Write a short paragraph about a walk in the park."`、`"Tell me about your favorite meal."`
  - α ∈ {0.0, 0.1, 0.2, 0.4}，第 18 层，greedy 解码，max_new_tokens 120，提示套 `format_chat`。
  - 这些 α 只是暂定值，正式数值在计划二校准。
  - 疼痛只生成上述 12 段（3 个提示 × 4 个 α），以遵守伦理限制。
- **摘要包含**：
  - 每种情绪的 AUROC 表（取最佳层）。
  - 愤怒 vs 恐惧能否分开（逐层，并用白话说明代表什么）。
  - 污染检查结果。
  - 三个轴在各层是否可靠（白话说明）。
  - 愤怒、恐惧最接近的特征。
  - 与 Pain-axis 疼痛向量的相似度。
  - 范例文字的连结。

- [ ] **Step 1: 写测试**

```python
def test_render_checkpoint1_contains_key_sections():
    s = render_checkpoint1(VALID_DF, {"18": {"auroc": 0.8123, "separable": True}}, CONTAM, AXES, SAE, "x.md")
    for h in ["愤怒 vs 恐惧", "污染检查", "三个轴", "特征字典", "范例文字"]:
        assert h in s
    assert "0.81" in s and "可以分开" in s
```

- [ ] **Step 2: 执行测试，确认失败**

Run: `pytest tests/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: 实现，并更新 README**

README 用简体中文写，参照 Pain-axis README 的结构：项目目的、目录说明、阶段 0–2 的执行顺序（列出每一个指令）、外部资料来源。

- [ ] **Step 4: 执行测试，确认通过**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: 正式执行**

Run: `python scripts/6.2_emotion_vectors/08_checkpoint_examples.py --model qwen3-8b`
Expected: 产生两个 Markdown 文件，范例文字共 72 段（6 种情绪 × 3 个提示 × 4 个 α）。

- [ ] **Step 6: 执行全部测试**

Run: `pytest -v`
Expected: 全部 PASS（gpu 和 network 测试默认跳过）

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: checkpoint 1 examples and summary"
```

- [ ] **Step 8: 用户检查点 1**

把 `checkpoint1_summary.md` 交给用户，用白话说明结果，然后依结果撰写计划二。

---

## 之后的计划（检查点 1 之后才撰写）

- **计划二（阶段 3–5）**：
  - 校准与冻结：`scripts/6.5_calibration/`
  - 惩罚情境与 MoralChoice 筛选：`scripts/6.6_moral_items/`、`datasets/6.6_*`
  - 所有实验条件：`scripts/6.7_runs/`
  - 统计与图表：`scripts/6.8_analysis/`
- **计划三（阶段 6–7）**：
  - Gemma 3 4B 重做：同一套脚本加上 `--model gemma3-4b`，使用同一份 `datasets/6.2_stories.jsonl`
  - Gemma Scope 2 交叉确认
  - 中文研究报告
