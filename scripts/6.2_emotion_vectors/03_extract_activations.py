"""Run every story through the model and save per-story mean residual activations."""
import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import hooks, models  # noqa: E402
from emolab.batching import run_with_oom_backoff  # noqa: E402
from emolab.paths import DATASETS, cache_dir, results_dir  # noqa: E402

INDEX_COLS = ["id", "label", "kind", "topic_idx", "subtype", "heldout"]


def _index_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{c: r.get(c, False if c == "heldout" else None) for c in INDEX_COLS}
                       for r in rows])
    df.insert(0, "row", range(len(df)))
    return df


def _is_done(out_dir: Path, ids: list[str], layers: list[int]) -> bool:
    """True only if the saved index has exactly these ids (in order) and all layer files exist."""
    idx_path = out_dir / "index.csv"
    if not idx_path.exists():
        return False
    if pd.read_csv(idx_path, usecols=["id"], dtype=str)["id"].tolist() != ids:
        return False
    for L in layers:
        p = out_dir / f"acts_L{L}.npy"
        if not p.exists() or np.load(p, mmap_mode="r").shape[0] != len(ids):
            return False
    return True


def _batches(model, encode_fn, rows, batch_size, fn):
    """Yield (row positions, fn(model, ids, mask)) over rows sorted by text length."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))

    def run(batch):
        enc = encode_fn([rows[i]["text"] for i in batch])
        return [fn(model, enc["input_ids"], enc["attention_mask"])]

    done = 0
    for batch, (res,) in run_with_oom_backoff(run, order, batch_size):
        done += len(batch)
        print(f"{done}/{len(rows)} rows", flush=True)
        yield batch, res


def extract(model, encode_fn: Callable[[list[str]], dict], rows: list[dict], layers: list[int],
            batch_size: int, out_dir: Path, force: bool = False) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in rows]
    if not force and _is_done(out_dir, ids, layers):
        print("activations already extracted, skipping", flush=True)
        return pd.read_csv(out_dir / "index.csv")

    index = _index_frame(rows)
    acts: dict[int, np.ndarray] = {}
    for batch, res in _batches(
            model, encode_fn, rows, batch_size,
            lambda m, i, a: hooks.capture_mean_activations(m, i, a, layers)):
        for L, h in res.items():
            if L not in acts:
                acts[L] = np.zeros((len(rows), h.shape[1]), dtype=np.float32)
            acts[L][batch] = h.numpy()
    for L in layers:
        np.save(out_dir / f"acts_L{L}.npy", acts[L])
    index.to_csv(out_dir / "index.csv", index=False, lineterminator="\n")  # last: marks completion
    return index


def resid_norms(model, encode_fn, rows: list[dict], layers: list[int],
                batch_size: int) -> dict[int, float]:
    """Per-token mean L2 norm over all used positions (batch means weighted by token count)."""
    tot = {L: 0.0 for L in layers}
    count = 0
    for _, (norms, n_used) in _batches(
            model, encode_fn, rows, batch_size,
            lambda m, i, a: (hooks.mean_token_norm(m, i, a, layers),
                             int(a.sum()) - a.shape[0])):  # minus first real token per row
        count += n_used
        for L in layers:
            tot[L] += norms[L] * n_used
    return {L: tot[L] / count for L in layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--stories", type=Path, default=DATASETS / "6.2_stories.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = models.MODELS[args.model]
    heldout = {t["idx"]: t["heldout"] for t in
               json.loads((DATASETS / "6.2_story_topics.json").read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in
            args.stories.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    for r in rows:
        r["heldout"] = bool(heldout[r["topic_idx"]])

    model, tok = models.load(args.model)
    layers = models.candidate_layers(len(models.get_layers(model)))
    dev = next(model.parameters()).device

    def encode(texts: list[str]) -> dict:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
        return {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev)}

    out_dir = cache_dir() / "6.2" / spec.name
    extract(model, encode, rows, layers, args.batch_size, out_dir, force=args.force)

    neutral = [r for r in rows if r["kind"] == "neutral" and not r["heldout"]]
    if not neutral:
        neutral = [r for r in rows if r["kind"] == "neutral"]
        if neutral:
            print("WARNING: no non-heldout neutral stories; resid_norms uses all neutral rows",
                  flush=True)
    if not neutral:
        print("no neutral stories in input; resid_norms.json not written", flush=True)
        return
    norms = resid_norms(model, encode, neutral, layers, args.batch_size)
    path = results_dir("6.2_emotion_vectors", spec.name) / "resid_norms.json"
    path.write_text(json.dumps({str(L): v for L, v in norms.items()}, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
