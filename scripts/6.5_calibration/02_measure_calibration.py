"""Rate calibration texts with the unsteered model as judge; write metrics.csv.

Default mode: reads texts.jsonl + coherence.csv from 01, rates each group's texts with the
group's emotion phrase, writes metrics.csv (one row per layer, alpha, label; alpha = 0 rows for
every layer x core emotion).

--controls --layer L --alpha A: generates the 20 prompts at (L, A) for 40 concept directions and
20 random directions, rates every text with all 6 core phrases; appends control_texts.jsonl and
control_ratings.csv (rows of other alphas are kept).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import calibration as cal  # noqa: E402
from emolab import models  # noqa: E402
from emolab.conditions import load_labels, random_directions  # noqa: E402
from emolab.paths import DATASETS, RESULTS, cache_dir, results_dir  # noqa: E402
from emolab.scoring import SteerSpec, judge_ratings  # noqa: E402
from emolab.stories import repair_torn_tail  # noqa: E402

PROMPTS = DATASETS / "6.5_calibration_prompts.json"
PHRASES = DATASETS / "6.5_emotion_phrases.json"


def _require(*paths: Path) -> None:
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"missing required input {p}")


def run_metrics(model, tok, out_dir: Path, emotions: list[str], judge: dict, bs: int) -> None:
    rows = repair_torn_tail(out_dir / "texts.jsonl")
    groups = cal.complete_groups(rows)
    coh = cal.read_coherence(out_dir / "coherence.csv")
    layers = sorted({int(k[0]) for k in groups if k[0] is not None})
    base = groups[(None, 0.0, cal.BASELINE)]
    ratings: dict = {}
    for e in emotions:
        ratings[(None, 0.0, e)] = float(np.nanmean(judge_ratings(model, tok, base, judge[e], bs)))
    measured = {(int(r.layer), float(r.alpha), r.label) for r in coh.itertuples(index=False)
                if not pd.isna(r.layer)}
    todo = [k for k in sorted(groups, key=lambda k: (k[1], k[0] or -1, k[2]))
            if k[0] is not None and k in measured and k[2] in emotions]
    for n, key in enumerate(todo, 1):
        ratings[key] = float(np.nanmean(judge_ratings(model, tok, groups[key], judge[key[2]], bs)))
        print(f"[{n}/{len(todo)}] L{key[0]} a={key[1]} {key[2]}: rating {ratings[key]:.3f}",
              flush=True)
    df = cal.build_metrics(ratings, coh, layers, emotions)
    df.to_csv(out_dir / "metrics.csv", index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows to {out_dir / 'metrics.csv'}", flush=True)


def control_directions(vec, name: str, layer: int, concepts: list[str]) -> dict:
    """control name -> unit direction (concept vectors + 20 random, random file created once)."""
    import torch
    rnd_path = RESULTS / "6.7_runs" / name / f"random_directions_L{layer}.npy"
    if rnd_path.exists():
        rnd = np.load(rnd_path)
    else:
        acts_dir = cache_dir() / "6.2" / name
        index = pd.read_csv(acts_dir / "index.csv")
        train = ~index["heldout"].astype(bool).to_numpy()
        acts = np.load(acts_dir / f"acts_L{layer}.npy", mmap_mode="r")
        rnd = random_directions(np.asarray(acts)[train], cal.N_RANDOM_CONTROLS, seed=42)
        rnd_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rnd_path, rnd)
    i = vec["layers"].index(layer)
    dirs = {c: vec["unit"][c][i] for c in concepts}
    for j in range(cal.N_RANDOM_CONTROLS):
        dirs[f"random{j:02d}"] = torch.tensor(rnd[j], dtype=torch.float32)
    return dirs


def run_controls(args, name: str, emotions: list[str], judge: dict, prompts: list[str]) -> None:
    import torch
    out_dir = results_dir("6.5_calibration", name)
    vec = torch.load(results_dir("6.2_emotion_vectors", name) / "vectors.pt", weights_only=False)
    concepts = load_labels()["concepts"]
    layer, alpha = args.layer, args.alpha
    if layer not in vec["layers"]:
        raise ValueError(f"layer {layer} not in vectors.pt layers {vec['layers']}")
    dirs = control_directions(vec, name, layer, concepts)
    names = cal.control_names(concepts)

    model, tok = models.load(args.model)
    tpath, rpath = out_dir / "control_texts.jsonl", out_dir / "control_ratings.csv"
    rows = repair_torn_tail(tpath)
    key = lambda r: (r["layer"], float(r["alpha"]), r["control"])  # noqa: E731
    rows = cal.drop_partial_groups(tpath, rows, key)
    have: dict = {}
    for r in rows:
        have.setdefault(key(r), {})[int(r["prompt_idx"])] = r["text"]
    existing = pd.read_csv(rpath) if rpath.exists() else None

    with tpath.open("a", encoding="utf-8", newline="\n") as f:
        for n, c in enumerate(names, 1):
            k = (layer, float(alpha), c)
            if k in have:
                texts = [have[k][i] for i in range(cal.N_PROMPTS)]
            else:
                spec = SteerSpec(layer, dirs[c], alpha, vec["resid_norm"][layer])
                texts = cal.generate_batch(model, tok, prompts, spec, cal.MAX_NEW_TOKENS,
                                           args.batch_size)
                for i, t in enumerate(texts):
                    f.write(json.dumps(dict(layer=layer, alpha=alpha, control=c, prompt_idx=i,
                                            text=t), ensure_ascii=False) + "\n")
                f.flush()
            per_emotion = {e: judge_ratings(model, tok, texts, judge[e], args.batch_size)
                           for e in emotions}
            existing = cal.merge_control_ratings(existing,
                                                 cal.control_rating_rows(alpha, c, per_emotion))
            existing.to_csv(rpath, index=False, lineterminator="\n")
            print(f"[{n}/{len(names)}] control {c} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    out_dir = results_dir("6.5_calibration", name)
    emotions = load_labels()["core"]
    judge = json.loads(PHRASES.read_text(encoding="utf-8"))["judge"]

    if args.controls:
        if args.layer is None or args.alpha is None:
            ap.error("--controls needs --layer and --alpha")
        _require(PROMPTS, PHRASES, results_dir("6.2_emotion_vectors", name) / "vectors.pt")
        rnd = RESULTS / "6.7_runs" / name / f"random_directions_L{args.layer}.npy"
        if not rnd.exists():
            _require(cache_dir() / "6.2" / name / "index.csv",
                     cache_dir() / "6.2" / name / f"acts_L{args.layer}.npy")
        prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
        run_controls(args, name, emotions, judge, prompts)
        return

    _require(out_dir / "texts.jsonl", out_dir / "coherence.csv", PHRASES)
    model, tok = models.load(args.model)
    run_metrics(model, tok, out_dir, emotions, judge, args.batch_size)


if __name__ == "__main__":
    main()
