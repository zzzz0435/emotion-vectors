"""Checkpoint 1: steered example generations plus the Chinese summary.

72 examples = 6 core emotions x 3 prompts x 4 alphas, layer 18, greedy decoding. The alpha = 0
text is generated once per prompt and shared by all emotions. Pain is only generated for these
12 passages (ethics limit). Alphas are provisional; real values are calibrated in plan two.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import hooks, models  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402
from emolab.report import render_checkpoint1  # noqa: E402

LAYER = 18
PROMPTS = ["Describe your morning routine.",
           "Write a short paragraph about a walk in the park.",
           "Tell me about your favorite meal."]
ALPHAS = [0.0, 0.1, 0.2, 0.4]
MAX_NEW_TOKENS = 120
NOTE = ("> 注意：这里的 α（调整强度）只是暂定值，用来先看一眼效果；正式数值会在计划二校准。"
        "α = 0 表示不调整（对照），同一个提示的 α = 0 文字只生成一次，所有情绪共用。"
        "解码用贪婪解码（每次结果相同），疼痛只生成了这 12 段。")


def render_examples(core: list[str], gens: dict) -> str:
    """gens[(emotion_or_None, prompt_idx, alpha)] -> text; alpha 0.0 is keyed with None."""
    lines = ["# 检查点 1：调整后的范例文字", "", NOTE, ""]
    for emo in core:
        lines += [f"## {emo}", ""]
        for pi, prompt in enumerate(PROMPTS):
            lines += [f"### 提示：{prompt}", ""]
            for a in ALPHAS:
                text = gens[(None if a == 0 else emo, pi, a)]
                lines += [f"**α = {a}**", "", "> " + text.strip().replace("\n", "\n> "), ""]
    return "\n".join(lines)


def parse_sae_tops(md: str, labels=("anger", "fear")) -> dict:
    """Read (feature, cosine) pairs per label out of sae_L{layer}.md."""
    out: dict = {}
    for sec in re.split(r"^## ", md, flags=re.M)[1:]:
        name = sec.split("\n", 1)[0].strip()
        if name in labels:
            out[name] = [(int(i), float(c)) for i, c in
                         re.findall(r"^### \d+\. 特征 (\d+)（余弦 (-?[\d.]+)）", sec, flags=re.M)]
    return out


def generate(model, tok, prompt: str, direction, alpha: float, resid_norm: float) -> str:
    dev = next(model.parameters()).device
    enc = tok(models.format_chat(tok, prompt), return_tensors="pt", add_special_tokens=False).to(dev)
    with hooks.steer(model, LAYER, direction, alpha, resid_norm), torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    rdir = results_dir("6.2_emotion_vectors", name)

    payload = torch.load(rdir / "vectors.pt")
    li = payload["layers"].index(LAYER)
    resid_norm = float(payload["resid_norm"][LAYER])
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    core = [e["label"] for e in emotions if e["group"] == "core"]

    model, tok = models.load(args.model)
    gens: dict = {}
    for pi, prompt in enumerate(PROMPTS):
        gens[(None, pi, 0.0)] = generate(model, tok, prompt, payload["unit"][core[0]][li], 0.0, resid_norm)
        for emo in core:
            for a in ALPHAS[1:]:
                gens[(emo, pi, a)] = generate(model, tok, prompt, payload["unit"][emo][li], a, resid_norm)
                print(emo, pi, a, flush=True)
    ex_path = rdir / "checkpoint1_examples.md"
    ex_path.write_text(render_examples(core, gens), encoding="utf-8", newline="\n")

    def load_json(p: Path):
        return json.loads(p.read_text(encoding="utf-8"))
    sae_md = results_dir("6.9_crosscheck", name) / f"sae_L{LAYER}.md"
    summary = render_checkpoint1(
        pd.read_csv(rdir / "validation.csv"), load_json(rdir / "separability.json"),
        load_json(rdir / "contamination.json"), load_json(results_dir("6.3_axes", name) / "axes.json"),
        parse_sae_tops(sae_md.read_text(encoding="utf-8")) if sae_md.exists() else {},
        str(ex_path), pain_compare=load_json(results_dir("6.9_crosscheck", name) / "pain_axis_compare.json"),
        core=core)
    (rdir / "checkpoint1_summary.md").write_text(summary, encoding="utf-8", newline="\n")
    print("wrote", ex_path, "and checkpoint1_summary.md")


if __name__ == "__main__":
    main()
