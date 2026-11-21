"""Cosine between the Pain-axis paper's published pain vector and ours.

Theirs: results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/pain_vectors.pt at
valen-research/Pain-axis commit 213fb6c (keys s1_pain_vector, s2_pain_vector, layer,
extraction). Our `pain_painaxis` follows their S2_1P recipe, so s2 is the primary comparison
(s1 is reported too). Compared at our candidate layer nearest to their layer.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models  # noqa: E402
from emolab.paths import cache_dir, results_dir  # noqa: E402

COMMIT = "213fb6c89c96004dbe93d1704c636ad8c9d801f3"
URL = ("https://raw.githubusercontent.com/valen-research/Pain-axis/" + COMMIT +
       "/results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/pain_vectors.pt")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest_layer(layers: list[int], target: int) -> int:
    return min(layers, key=lambda L: (abs(L - target), L))


def fetch(dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name

    theirs = torch.load(fetch(cache_dir() / "pain_axis" / "pain_vectors.pt"),
                        map_location="cpu", weights_only=False)
    their_layer = int(theirs["layer"])
    vec = torch.load(results_dir("6.2_emotion_vectors", name) / "vectors.pt")
    our_layer = nearest_layer(vec["layers"], their_layer)
    i = vec["layers"].index(our_layer)
    pain = vec["unit"]["pain"][i].double().numpy()
    pain_pa = vec["unit"]["pain_painaxis"][i].double().numpy()
    s2, s1 = theirs["s2_pain_vector"].double().numpy(), theirs["s1_pain_vector"].double().numpy()

    note = ("层号相同" if our_layer == their_layer else
            f"它们的层是 {their_layer}，我们的候选层里最接近的是 {our_layer}，两者不同层，"
            "余弦只能当作粗略参考")
    note += "；它们的向量取自 Qwen3-8B-Base，我们的取自指令版；比较对象为 s2_pain_vector（与 S2_1P 做法一致）"
    out = {"their_layer": their_layer, "our_layer": our_layer,
           "cos_pain": cosine(pain, s2), "cos_pain_painaxis": cosine(pain_pa, s2),
           "note": note,
           "cos_pain_s1": cosine(pain, s1), "cos_pain_painaxis_s1": cosine(pain_pa, s1)}
    path = results_dir("6.9_crosscheck", name) / "pain_axis_compare.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
