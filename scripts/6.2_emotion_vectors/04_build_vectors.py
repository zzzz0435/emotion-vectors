"""Build label-minus-neutral difference-of-means vectors from the training stories."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models  # noqa: E402
from emolab.paths import DATASETS, cache_dir, results_dir  # noqa: E402
from emolab.vectors import as_bool, diff_of_means  # noqa: E402


def expected_labels() -> tuple[list[str], list[str], str]:
    """(20 emotion labels, 42 concept + contamination labels, neutral label)."""
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    concepts = json.loads((DATASETS / "6.2_concepts.json").read_text(encoding="utf-8"))
    emo = [e["label"] for e in emotions if e["group"] != "neutral"]
    neutral = next(e["label"] for e in emotions if e["group"] == "neutral")
    return emo, [c["label"] for c in concepts], neutral


def build_vectors(acts: dict[int, np.ndarray], index: pd.DataFrame, labels: list[str],
                  neutral: str, resid_norm: dict[int, float], model_name: str) -> dict:
    """acts: {layer: [N, d]} row-aligned with index. Training rows are heldout == False."""
    layers = sorted(acts)
    row_labels = index["label"].to_numpy()
    held = as_bool(index["heldout"]).to_numpy()
    train = ~held
    n_train = {lab: int(((row_labels == lab) & train).sum()) for lab in [*labels, neutral]}
    for lab, n in n_train.items():
        if n == 0:
            raise ValueError(f"label {lab!r} has no training rows (non-heldout stories)")
    raw: dict[str, torch.Tensor] = {}
    unit: dict[str, torch.Tensor] = {}
    for lab in labels:
        v = np.stack([diff_of_means(acts[L], row_labels, lab, neutral, train) for L in layers])
        t = torch.from_numpy(v.astype(np.float32))
        raw[lab] = t
        unit[lab] = t / t.norm(dim=-1, keepdim=True)
    heldout_topics = sorted(int(t) for t in index.loc[held, "topic_idx"].unique())
    return {"layers": layers, "resid_norm": {L: float(resid_norm[L]) for L in layers},
            "raw": raw, "unit": unit,
            "meta": {"model": model_name, "n_train": n_train, "heldout_topics": heldout_topics}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    cdir = cache_dir() / "6.2" / name
    rdir = results_dir("6.2_emotion_vectors", name)

    index = pd.read_csv(cdir / "index.csv")
    norms = {int(k): v for k, v in
             json.loads((rdir / "resid_norms.json").read_text(encoding="utf-8")).items()}
    acts = {L: np.load(cdir / f"acts_L{L}.npy") for L in sorted(norms)}
    emo, concepts, neutral = expected_labels()
    out = build_vectors(acts, index, emo + concepts, neutral, norms, name)
    torch.save(out, rdir / "vectors.pt")
    print(f"wrote {rdir / 'vectors.pt'}: {len(out['raw'])} labels x layers {out['layers']}")


if __name__ == "__main__":
    main()
