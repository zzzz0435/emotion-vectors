"""Calibration sweep logic shared by 6.5_calibration/01 (generate) and 02 (measure).

Pure logic (plan, early stop, resume state, metrics assembly) is GPU-free and takes injected
functions; the model-calling wrappers (generate_batch, format_valid, ppl_ratio) are thin.
"""
import json
import math
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
PPL_MAX = 1.5
FORMAT_MIN = 0.9
MAX_NEW_TOKENS = 80
N_PROMPTS = 20
BASELINE = "baseline"
COHERENCE_COLS = ["layer", "alpha", "label", "ppl_ratio", "format_valid", "coherent"]
METRICS_COLS = ["layer", "alpha", "label", "rating_mean", "ppl_ratio", "format_valid",
                "coherent", "gain"]
CONTROL_COLS = ["alpha", "control", "emotion", "rating"]
N_RANDOM_CONTROLS = 20


def coherent(ppl_ratio: float, format_valid: float) -> bool:
    return bool(ppl_ratio <= PPL_MAX and format_valid >= FORMAT_MIN)


def sweep_plan(emotions: Iterable[str], layers: Iterable[int],
               grid: Iterable[float]) -> Iterator[tuple[int, float, str]]:
    """(layer, alpha, emotion), alpha ascending, then layer, then emotion."""
    emotions, layers = list(emotions), list(layers)
    for alpha in sorted(grid):
        for layer in layers:
            for emo in emotions:
                yield layer, alpha, emo


def run_sweep(plan, done: set, generate_fn: Callable, measure_fn: Callable,
              on_group: Callable, incoherent: set | None = None,
              stored_texts: dict | None = None) -> None:
    """Early-stop driver.

    done: groups (layer, alpha, label) that are complete. incoherent: groups already recorded
    incoherent (history). stored_texts: groups whose texts exist but were not measured; they are
    measured, not regenerated. Once (layer, label) is incoherent at some alpha, larger alphas are
    never generated. generate_fn(layer, alpha, label) -> texts; measure_fn(layer, alpha, label,
    texts) -> (ppl_ratio, format_valid); on_group(layer, alpha, label, texts, ppl, fv, ok).
    """
    bad: dict[tuple, float] = {}
    for layer, alpha, label in (incoherent or ()):
        bad[(layer, label)] = min(alpha, bad.get((layer, label), math.inf))
    stored_texts = stored_texts or {}
    for layer, alpha, label in plan:
        key = (layer, alpha, label)
        if key in done or alpha > bad.get((layer, label), math.inf):
            continue
        texts = stored_texts.get(key)
        if texts is None:
            texts = generate_fn(layer, alpha, label)
        ppl, fv = measure_fn(layer, alpha, label, texts)
        ok = coherent(ppl, fv)
        if not ok:
            bad[(layer, label)] = min(alpha, bad.get((layer, label), math.inf))
        on_group(layer, alpha, label, texts, ppl, fv, ok)


# ---- resume state -------------------------------------------------------------------------

def _gkey(row: dict) -> tuple:
    layer = row["layer"]
    return (None if layer is None else int(layer), float(row["alpha"]), row["label"])


def complete_groups(rows: list[dict], n: int = N_PROMPTS) -> dict[tuple, list[str]]:
    """Groups with exactly the prompt indices 0..n-1 -> texts ordered by prompt_idx."""
    grouped: dict[tuple, list[tuple[int, str]]] = {}
    for r in rows:
        grouped.setdefault(_gkey(r), []).append((int(r["prompt_idx"]), r["text"]))
    out = {}
    for key, items in grouped.items():
        items.sort()
        if [i for i, _ in items] == list(range(n)):
            out[key] = [t for _, t in items]
    return out


def read_coherence(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COHERENCE_COLS)
    df = pd.read_csv(path, dtype={"layer": "string", "label": "string"})
    df["alpha"] = df["alpha"].astype(float)
    df["coherent"] = df["coherent"].astype(str).str.lower().eq("true")
    return df


def resume_state(text_rows: list[dict], coh: pd.DataFrame, n: int = N_PROMPTS):
    """-> (done, incoherent, stored_texts) for run_sweep. Steered groups only.

    done = 20 texts AND a coherence row; stored = texts without a coherence row."""
    full = complete_groups(text_rows, n)
    measured = {}
    for r in coh.itertuples(index=False):
        if pd.isna(r.layer) or r.label == BASELINE:
            continue
        measured[(int(r.layer), float(r.alpha), r.label)] = bool(r.coherent)
    done = {k for k in measured if k in full}
    incoherent = {k for k, ok in measured.items() if not ok}
    stored = {k: v for k, v in full.items() if k[0] is not None and k not in measured}
    return done, incoherent, stored


# ---- metrics assembly (02) ----------------------------------------------------------------

def build_metrics(ratings: dict, coherence: pd.DataFrame, layers: list[int],
                  emotions: list[str]) -> pd.DataFrame:
    """metrics rows. ratings: {(layer, alpha, label): mean rating}, baseline under
    (None, 0.0, label). Steered rows need both a rating and a coherence row; gain is relative
    to the baseline rating of the same label; alpha = 0 rows exist for every layer x emotion."""
    base_fv = float("nan")
    ref = coherence[coherence["label"] == BASELINE]
    if len(ref):
        base_fv = float(ref["format_valid"].iloc[0])
    rows = []
    for layer in layers:
        for emo in emotions:
            rows.append(dict(layer=layer, alpha=0.0, label=emo, rating_mean=ratings[(None, 0.0, emo)],
                             ppl_ratio=1.0, format_valid=base_fv, coherent=True, gain=0.0))
    for c in coherence.itertuples(index=False):
        if pd.isna(c.layer) or c.label == BASELINE:
            continue
        key = (int(c.layer), float(c.alpha), c.label)
        if key not in ratings or c.label not in emotions:
            continue
        rows.append(dict(layer=key[0], alpha=key[1], label=c.label, rating_mean=ratings[key],
                         ppl_ratio=float(c.ppl_ratio), format_valid=float(c.format_valid),
                         coherent=bool(c.coherent),
                         gain=ratings[key] - ratings[(None, 0.0, c.label)]))
    df = pd.DataFrame(rows, columns=METRICS_COLS)
    order = {e: i for i, e in enumerate(emotions)}
    df["_o"] = df["label"].map(order)
    return df.sort_values(["layer", "alpha", "_o"]).drop(columns="_o").reset_index(drop=True)


# ---- controls (02 --controls) -------------------------------------------------------------

def control_names(concepts: list[str], n_random: int = N_RANDOM_CONTROLS) -> list[str]:
    return list(concepts) + [f"random{i:02d}" for i in range(n_random)]


def control_rating_rows(alpha: float, control: str, ratings_by_emotion: dict) -> list[dict]:
    """ratings_by_emotion: {emotion: per-prompt ratings} -> one row per emotion (nan-mean)."""
    out = []
    for emo, vals in ratings_by_emotion.items():
        v = np.asarray(vals, dtype=float)
        out.append(dict(alpha=alpha, control=control, emotion=emo,
                        rating=float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")))
    return out


def merge_control_ratings(existing: pd.DataFrame | None, new_rows: list[dict]) -> pd.DataFrame:
    """Append new rows; rows of other alphas are kept, same (alpha, control) are replaced."""
    new = pd.DataFrame(new_rows, columns=CONTROL_COLS)
    if existing is None or not len(existing):
        return new
    keys = set(zip(new["alpha"].astype(float), new["control"]))
    keep = [(float(a), c) not in keys for a, c in zip(existing["alpha"], existing["control"])]
    return pd.concat([existing[keep], new], ignore_index=True)


def rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def drop_partial_groups(path: Path, rows: list[dict], key_fn: Callable,
                        n: int = N_PROMPTS) -> list[dict]:
    """Remove groups with != n rows (crash mid-group) from the file; return the kept rows."""
    counts: dict = {}
    for r in rows:
        counts[key_fn(r)] = counts.get(key_fn(r), 0) + 1
    kept = [r for r in rows if counts[key_fn(r)] == n]
    if len(kept) != len(rows):
        rewrite_jsonl(path, kept)
    return kept


# ---- model wrappers -----------------------------------------------------------------------

def generate_batch(model, tok, prompts: list[str], steer_spec, max_new_tokens: int = MAX_NEW_TOKENS,
                   batch_size: int = N_PROMPTS) -> list[str]:
    """Greedy continuations of chat-formatted prompts (left padded), steered if steer_spec."""
    import contextlib

    import torch

    from emolab.batching import run_with_oom_backoff
    from emolab.hooks import steer
    from emolab.models import format_chat

    texts = [format_chat(tok, p) for p in prompts]
    dev = next(model.parameters()).device

    def run(batch: list[str]) -> list[str]:
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False,
                  padding_side="left")
        enc = {k: v.to(dev) for k, v in enc.items()}
        ctx = (steer(model, steer_spec.layer, steer_spec.direction, steer_spec.alpha,
                     steer_spec.resid_norm) if steer_spec is not None else contextlib.nullcontext())
        with ctx, torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        n_in = enc["input_ids"].shape[1]
        return [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]

    res: list[str] = []
    for _, r in run_with_oom_backoff(run, texts, batch_size):
        res.extend(r)
    return res


def format_valid(model, tok, format_items: dict, steer_spec, batch_size: int = 16) -> float:
    """Mean over all format items of (1 - invalid answer mass)."""
    from emolab.scoring import score_prompts
    inv = []
    for key, options in (("ab", ["A", "B"]), ("rating", [str(i) for i in range(1, 8)])):
        msgs = [it["question"] for it in format_items[key]]
        _, invalid = score_prompts(model, tok, msgs, options, batch_size=batch_size,
                                   steer_spec=steer_spec)
        inv.extend(np.asarray(invalid, dtype=float).tolist())
    return float(1.0 - np.mean(inv))


def median_ppl(model, tok, prompts: list[str], texts: list[str]) -> float:
    """Median continuation perplexity; an empty text counts as infinitely bad."""
    from emolab.scoring import continuation_ppl
    ok = [i for i, t in enumerate(texts) if t.strip()]
    vals = np.full(len(texts), np.inf)
    if ok:
        vals[ok] = continuation_ppl(model, tok, [prompts[i] for i in ok], [texts[i] for i in ok])
    return float(np.median(vals))


def ppl_ratio(model, tok, prompts: list[str], texts: list[str], base_median: float) -> float:
    return median_ppl(model, tok, prompts, texts) / base_median


# ---- resumable driver with disk output (01) -----------------------------------------------

def sweep_to_disk(out_dir: Path, plan, generate_fn: Callable, measure_fn: Callable,
                  log: Callable = print) -> None:
    """Run the early-stop sweep, appending texts.jsonl and coherence.csv group by group.

    Resume: a group is done only with 20 texts AND a coherence row; texts without a coherence
    row are re-measured, not regenerated; groups cut off mid-write are dropped and redone.
    """
    from emolab.stories import repair_torn_tail
    texts_path, coh_path = out_dir / "texts.jsonl", out_dir / "coherence.csv"
    rows = repair_torn_tail(texts_path)
    rows = drop_partial_groups(texts_path, rows, lambda r: _gkey(r))
    coh = read_coherence(coh_path)
    done, incoherent, stored = resume_state(rows, coh)
    if not coh_path.exists() or not len(coh):
        coh_path.write_text(",".join(COHERENCE_COLS) + "\n", encoding="utf-8", newline="\n")
    count = 0

    with texts_path.open("a", encoding="utf-8", newline="\n") as tf, \
            coh_path.open("a", encoding="utf-8", newline="\n") as cf:
        def on_group(layer, alpha, label, texts, ppl, fv, ok):
            nonlocal count
            count += 1
            if (layer, alpha, label) not in stored:
                for i, t in enumerate(texts):
                    tf.write(json.dumps(dict(layer=layer, alpha=alpha, label=label, prompt_idx=i,
                                             text=t), ensure_ascii=False) + "\n")
                tf.flush()
            cf.write(f"{layer},{alpha!r},{label},{ppl!r},{fv!r},{ok}\n")
            cf.flush()
            log(f"[{count}] L{layer} a={alpha} {label}: ppl_ratio={ppl:.3f} "
                f"format_valid={fv:.3f} coherent={ok}")

        run_sweep(plan, done, generate_fn, measure_fn, on_group, incoherent, stored)
