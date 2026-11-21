"""Cosine between the Pain-axis paper's published pain vector and ours, at THEIR layer.

Theirs: results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/pain_vectors.pt at
valen-research/Pain-axis commit 213fb6c (keys s1_pain_vector, s2_pain_vector, layer,
extraction). Our vectors are recomputed (running the model) at their layer, so the layers
match: mean-over-tokens label-minus-neutral vectors from the training stories (pain plus
baseline emotions) and `pain_painaxis` from the S2_1P sentences. Cosines with the baselines
show whether pain's cosine stands out. s2 matches our S2_1P recipe; s1 is reported too.
Remaining mismatches (extraction, base vs chat model) are recorded in the JSON.
"""
import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models  # noqa: E402
from emolab.paths import DATASETS, cache_dir, results_dir  # noqa: E402
from emolab.vectors import diff_of_means, pain_axis_vector  # noqa: E402

COMMIT = "213fb6c89c96004dbe93d1704c636ad8c9d801f3"
URL = ("https://raw.githubusercontent.com/valen-research/Pain-axis/" + COMMIT +
       "/results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/pain_vectors.pt")
BASELINES = ("sadness", "fear", "anger", "joy")


def _load_script(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename, Path(__file__).parents[1] / "6.2_emotion_vectors" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def label_vectors(acts: np.ndarray, labels: np.ndarray, targets: list[str],
                  neutral: str = "neutral") -> dict[str, np.ndarray]:
    """label-minus-neutral difference of means at one layer; acts [N, d] are training rows."""
    mask = np.ones(len(labels), dtype=bool)
    return {t: diff_of_means(acts, labels, t, neutral, mask) for t in targets}


def compare(their_layer: int, theirs: dict[str, np.ndarray], ours: dict[str, np.ndarray],
            our_layer: int | None = None) -> dict:
    """ours: {"pain", "pain_painaxis", <baseline emotions>} at one layer; theirs: {"s2", "s1"}."""
    our_layer = their_layer if our_layer is None else our_layer
    vectors = {}
    for tname, tv in theirs.items():
        cos = {k: cosine(v, tv) for k, v in ours.items()}
        vectors[tname] = {
            "cos_pain": cos["pain"], "cos_pain_painaxis": cos["pain_painaxis"],
            "cos_baselines": {k: cos[k] for k in BASELINES if k in cos},
            "ranking": sorted(cos, key=lambda k: -cos[k]),
        }
    return {"their_layer": their_layer, "our_layer": our_layer,
            "layer_match": their_layer == our_layer,
            "extraction_mismatch": "theirs final_token, ours mean",
            "model_mismatch": "theirs base, ours chat",
            "primary": "s2", "vectors": vectors}


def fetch(dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    name = models.MODELS[args.model].name

    theirs = torch.load(fetch(cache_dir() / "pain_axis" / "pain_vectors.pt"),
                        map_location="cpu", weights_only=False)
    L = int(theirs["layer"])

    heldout = {t["idx"]: t["heldout"] for t in
               json.loads((DATASETS / "6.2_story_topics.json").read_text(encoding="utf-8"))}
    stories = [json.loads(x) for x in
               (DATASETS / "6.2_stories.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    wanted = {"pain", "neutral", *BASELINES}
    rows = [r for r in stories if r["label"] in wanted and not heldout[r["topic_idx"]]]
    data = json.loads((DATASETS / "6.2_pain_axis_3.1_pain_and_control_datasets.json")
                      .read_text(encoding="utf-8"))
    pain_axis = _load_script("05_pain_axis_vector.py")
    pain_t, ctrl_t = pain_axis.select_pain_rows(data)

    model, tok = models.load(args.model)
    dev = next(model.parameters()).device

    def encode(texts: list[str]) -> dict:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
        return {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev)}

    story_acts = pain_axis.sentence_activations(
        model, encode, [r["text"] for r in rows], [L], args.batch_size)[L]
    sent_acts = pain_axis.sentence_activations(
        model, encode, pain_t + ctrl_t, [L], args.batch_size)[L]
    ours = label_vectors(story_acts, np.array([r["label"] for r in rows]), ["pain", *BASELINES])
    n = len(pain_t)
    ours["pain_painaxis"] = pain_axis_vector(sent_acts[:n], sent_acts[n:], 0.5)

    out = compare(L, {"s2": theirs["s2_pain_vector"].double().numpy(),
                      "s1": theirs["s1_pain_vector"].double().numpy()}, ours)
    out["n_training_stories"] = {k: sum(r["label"] == k for r in rows) for k in sorted(wanted)}
    path = results_dir("6.9_crosscheck", name) / "pain_axis_compare.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
