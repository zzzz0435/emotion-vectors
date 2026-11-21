# emotional

在开源大模型（Qwen3-8B 等）内部提取情绪方向，用「调整（steering）」改变模型的内部情绪状态，并测量其道德判断的变化。
目前覆盖阶段 0–2：环境、情绪向量的提取与验证、情绪轴的拟合。

## 目录结构

- `datasets/`：提示描述、故事主题、对照概念、生成的故事
- `scripts/emolab/`：共用模块（可测试的逻辑）
- `scripts/6.x_*/`：按编号执行的脚本
- `results/`：各脚本的输出
- `tests/`：pytest 测试（`-m gpu` 需要显卡）

## 执行顺序（阶段 0–2）

所有命令都在项目根目录执行，先启用虚拟环境（放在 OneDrive 之外：`C:\Users\xgame\.venvs\emotional`）。
先用 `qwen3-8b`（正式）；`qwen3-1.7b` 可用来先试跑。需要显卡的步骤是 01、03、05、6.9/02、08。

```
# 0. 检查环境
python scripts/6.0_env/01_check_environment.py --models qwen3-8b

# 1. 词表：先下载 NRC VAD v2.1（见下方外部资料），再提取 20 种情绪的评分
python scripts/6.3_axes/00_extract_vad.py

# 2. 生成故事，然后过滤
python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b
#    补足循环：若 shortfall.json 还有缺口，就补生成再过滤，重复到核心情绪的缺口为 0 为止。
#    每一轮补生成都用新的随机种子（42 + 1000 × 起始 replicate），并按 filter_report.csv 里该情绪
#    的通过率多生成（缺口 ÷ 通过率，通过率下限 0.1，上限 5 倍缺口），所以通常 2–4 轮就够。
python scripts/6.2_emotion_vectors/01_generate_stories.py --model qwen3-8b --top-up
python scripts/6.2_emotion_vectors/02_filter_stories.py --model qwen3-8b

# 3. 提取激活、构建向量
python scripts/6.2_emotion_vectors/03_extract_activations.py --model qwen3-8b
python scripts/6.2_emotion_vectors/04_build_vectors.py --model qwen3-8b

# 4. Pain-axis 疼痛向量、验证、污染检查
python scripts/6.2_emotion_vectors/05_pain_axis_vector.py --model qwen3-8b
#    注意：05 会就地修改 vectors.pt；之后若重跑 04，pain_painaxis 会丢失，需要再跑一次 05。
python scripts/6.2_emotion_vectors/06_validate_vectors.py --model qwen3-8b
python scripts/6.2_emotion_vectors/07_contamination.py --model qwen3-8b

# 5. 情绪轴（V/A/D）与残差向量
python scripts/6.3_axes/01_fit_axes.py --model qwen3-8b
python scripts/6.3_axes/02_residual_vectors.py --model qwen3-8b

# 6. 交叉确认：SAE 最近特征、与 Pain-axis 论文向量比较
#    6.9/01 会下载约 2 GB 的 SAE 权重，需要约 2–3 GB 内存，但不需要显卡。
python scripts/6.9_crosscheck/01_sae_nearest_features.py --model qwen3-8b
python scripts/6.9_crosscheck/02_compare_pain_axis.py --model qwen3-8b

# 7. 检查点 1：范例文字（72 段）与中文摘要
python scripts/6.2_emotion_vectors/08_checkpoint_examples.py --model qwen3-8b
```

最后会得到 `results/6.2_emotion_vectors/<模型名>/checkpoint1_summary.md`（给你读的摘要）和
`checkpoint1_examples.md`（范例文字）。范例里的调整强度 α 只是暂定值。

测试：`python -m pytest -v`（gpu 与 network 测试默认跳过）。

## 外部资料

- NRC VAD Lexicon v2.1：到 https://saifmohammad.com/WebPages/nrc-vad.html 下载，解压到
  `<缓存目录>/nrc_vad/NRC-VAD-Lexicon-v2.1/`（缓存目录默认 `C:\Users\xgame\.cache\emotional`，
  可用环境变量 `EMOTIONAL_CACHE` 改，但不能在项目目录内）。词表不可再散布，不进 git；
  提取出的 `datasets/6.3_nrc_vad_emotions.csv` 同样不进 git。
- Pain-axis（valen-research/Pain-axis，固定在 commit `213fb6c`）：其句子数据放在
  `datasets/6.2_pain_axis_3.1_pain_and_control_datasets.json`，其公开的疼痛向量由 6.9/02 自动下载到缓存目录。
- Qwen-Scope SAE（`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`）：6.9/01 从 Hugging Face 自动下载；
  它训练于基础版模型，只当粗略参考。
- 各数据集的来源与授权说明见 `datasets/ATTRIBUTION.md`。
