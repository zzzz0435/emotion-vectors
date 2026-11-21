"""Add the Pain-axis-paper pain vector (`pain_painaxis`) to vectors.pt."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import hooks, models  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402
from emolab.vectors import pain_axis_vector  # noqa: E402

PAIN_CATS = ("A1", "A2", "A3", "A5")  # A4 (Moral Injury) excluded
CONTROL_CATS = ("B", "C1", "C2", "D", "E")


def select_pain_rows(data: dict, dataset: str = "S2_1P") -> tuple[list[str], list[str]]:
    """(pain sentences, control sentences) from the Pain-axis dataset JSON, in file order."""
    sents = data["datasets"][dataset]["sentences"]
    return ([s["prompt"] for s in sents if s["category"] in PAIN_CATS],
            [s["prompt"] for s in sents if s["category"] in CONTROL_CATS])


def add_pain_vector(path: Path, vec_by_layer: dict[int, np.ndarray]) -> None:
    """Load vectors.pt, add `pain_painaxis` to raw and unit, re-save (other keys untouched)."""
    payload = torch.load(path)
    raw = torch.from_numpy(np.stack([vec_by_layer[L] for L in payload["layers"]]).astype(np.float32))
    payload["raw"]["pain_painaxis"] = raw
    payload["unit"]["pain_painaxis"] = raw / raw.norm(dim=-1, keepdim=True)
    torch.save(payload, path)


def _extract_script():
    """03_extract_activations.py has a digit-leading name, so load it by path (reuse _batches)."""
    spec = importlib.util.spec_from_file_location(
        "extract_script", Path(__file__).with_name("03_extract_activations.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sentence_activations(model, encode, texts: list[str], layers: list[int],
                         batch_size: int) -> dict[int, np.ndarray]:
    ext = _extract_script()
    rows = [{"text": t} for t in texts]
    acts: dict[int, np.ndarray] = {}
    for batch, res in ext._batches(
            model, encode, rows, batch_size,
            lambda m, i, a: hooks.capture_mean_activations(m, i, a, layers)):
        for L, h in res.items():
            if L not in acts:
                acts[L] = np.zeros((len(rows), h.shape[1]), dtype=np.float32)
            acts[L][batch] = h.numpy()
    return acts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    spec = models.MODELS[args.model]
    path = results_dir("6.2_emotion_vectors", spec.name) / "vectors.pt"
    if not path.exists():
        sys.exit(f"{path} not found; run 04_build_vectors.py first")
    layers = torch.load(path)["layers"]

    data = json.loads((DATASETS / "6.2_pain_axis_3.1_pain_and_control_datasets.json")
                      .read_text(encoding="utf-8"))
    pain_t, ctrl_t = select_pain_rows(data)
    model, tok = models.load(args.model)
    dev = next(model.parameters()).device

    def encode(texts: list[str]) -> dict:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
        return {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev)}

    acts = sentence_activations(model, encode, pain_t + ctrl_t, layers, args.batch_size)
    n = len(pain_t)
    vecs = {L: pain_axis_vector(acts[L][:n], acts[L][n:], 0.5) for L in layers}
    add_pain_vector(path, vecs)
    print(f"added pain_painaxis ({n} pain / {len(ctrl_t)} control sentences) to {path}")


if __name__ == "__main__":
    main()
