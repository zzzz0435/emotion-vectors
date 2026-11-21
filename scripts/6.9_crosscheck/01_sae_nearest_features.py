"""Look at our emotion vectors through the official Qwen-Scope SAE dictionary.

For the 6 core emotions, `anger__resid3` and `pain_painaxis`: the 10 decoder features with the
highest cosine, and for each feature the 5 of our stories projecting most on it (first 30
characters), as a rough sense of what the feature means. Writes sae_L{layer}.md (Chinese).
The dictionary was trained on the base model, so it is only a rough reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import axes, models, sae  # noqa: E402
from emolab.paths import DATASETS, cache_dir, results_dir  # noqa: E402

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
EXTRA_VECTORS = ("anger__resid3", "pain_painaxis")
TOP_K, TOP_STORIES, SNIPPET = 10, 5, 30


def snippet(text: str, n: int = SNIPPET) -> str:
    return " ".join(text.split())[:n].replace("|", "\\|")


def render(sections: list[dict], layer: int) -> str:
    """sections: {"name", "features": [(idx, cos, [(story_id, label, snippet), ...])]} or
    {"name", "skipped": reason}."""
    lines = [f"# SAE 最近特征（第 {layer} 层）", "",
             "> 注意：此字典训练于基础版（Qwen3-8B-Base），而我们的向量来自指令版，"
             "只能当作粗略参考。", "",
             f"字典：`{SAE_REPO}`，`layer{layer}.sae.pt`。每个向量列出余弦最高的 {TOP_K} 个特征；"
             f"每个特征再列出我们的故事里在该特征方向上投影最大的 {TOP_STORIES} 篇的前 {SNIPPET} 个字。",
             "按余弦从高到低列出；余弦为负的特征（方向相反）不在列表中。", ""]
    for sec in sections:
        lines += [f"## {sec['name']}", ""]
        if "skipped" in sec:
            lines += [f"（略过：{sec['skipped']}）", ""]
            continue
        for rank, (idx, cos, stories) in enumerate(sec["features"], 1):
            lines += [f"### {rank}. 特征 {idx}（余弦 {cos:.3f}）", ""]
            lines += [f"- `{sid}`（{lab}）：{snip}" for sid, lab, snip in stories]
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--d-model", type=int, default=4096)
    args = ap.parse_args()
    name = models.MODELS[args.model].name

    from huggingface_hub import hf_hub_download
    W = sae.load_decoder(Path(hf_hub_download(SAE_REPO, f"layer{args.layer}.sae.pt")), args.d_model)

    vec = torch.load(results_dir("6.2_emotion_vectors", name) / "vectors.pt")
    res = torch.load(results_dir("6.3_axes", name) / "residuals.pt")
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    core = [e["label"] for e in emotions if e["group"] == "core"]
    li = vec["layers"].index(args.layer)

    cdir = cache_dir() / "6.2" / name
    acts = np.load(cdir / f"acts_L{args.layer}.npy")
    index = pd.read_csv(cdir / "index.csv")
    texts = {}
    for line in (DATASETS / "6.2_stories.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            texts[r["id"]] = r["text"]

    sections = []
    for label in [*core, *EXTRA_VECTORS]:
        try:
            if label in res["unit"]:
                v = axes.get_residual(res, label, args.layer)
            else:
                v = vec["unit"][label][li].double().numpy()
        except (ValueError, KeyError) as e:
            sections.append({"name": label, "skipped": str(e)})
            continue
        near = sae.nearest_features(v, W, TOP_K)
        tops = sae.top_stories_per_feature(acts, W, [i for i, _ in near], TOP_STORIES)
        feats = [(i, c, [(index["id"][r], index["label"][r], snippet(texts[index["id"][r]]))
                         for r in tops[i]]) for i, c in near]
        sections.append({"name": label, "features": feats})
        print(label, [i for i, _ in near[:3]], flush=True)

    out = results_dir("6.9_crosscheck", name) / f"sae_L{args.layer}.md"
    out.write_text(render(sections, args.layer), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
