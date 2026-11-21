# emotional

在开源大模型（Qwen3-8B 等）内部提取情绪方向，用「调整（steering）」改变模型的内部情绪状态，并测量其道德判断的变化。
目前覆盖阶段 0–2：环境、情绪向量的提取与验证、情绪轴的拟合。

## 目录结构

- `datasets/`：提示描述、故事主题、对照概念、生成的故事
- `scripts/emolab/`：共用模块（可测试的逻辑）
- `scripts/6.x_*/`：按编号执行的脚本
- `results/`：各脚本的输出
- `tests/`：pytest 测试（`-m gpu` 需要显卡）

## 执行顺序

（由后续任务补充）

- 6.3 轴的输入：下载 NRC VAD v2.1 到缓存目录后，运行 `python scripts/6.3_axes/00_extract_vad.py` 生成 `datasets/6.3_nrc_vad_emotions.csv`（词表不可再散布，不进 git）。
