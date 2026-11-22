"""Moral-question runner logic: prompts, score rows, resumable atomic per-(task, condition) files."""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from emolab.integrity import check_frozen_hashes
from emolab.moral_items import h1_choice_prompt, h1_rating_prompt, h2_prompt
from emolab.scoring import expected_rating, harsh_high

TASKS = ("h1_rating", "h1_choice", "h2")
OPTIONS = {"h1_rating": [str(i) for i in range(1, 8)], "h1_choice": ["A", "B"], "h2": ["A", "B"]}
COLUMNS = ("item_id", "variant", "p_target", "invalid")


def task_prompts(task: str, items: list[dict]) -> list[tuple]:
    """(item_id, variant, user_msg, target) per prompt; target is the harsh/harmful letter, or the
    rating direction for h1_rating."""
    out = []
    if task == "h1_rating":
        for v in items:
            for d in ("light1", "harsh1"):
                out.append((v["id"], d, h1_rating_prompt(v, d), d))
    elif task in ("h1_choice", "h2"):
        fn = h1_choice_prompt if task == "h1_choice" else h2_prompt
        for it in items:
            for order in (0, 1):
                prompt, letter = fn(it, order)
                out.append((it["id"], str(order), prompt, letter))
    else:
        raise ValueError(f"unknown task {task!r}")
    return out


def rows_from_scores(task: str, meta: list[tuple], probs, invalid) -> list[dict]:
    probs = np.asarray(probs, dtype=np.float64)
    rows = []
    for (item_id, variant, _msg, target), p, inv in zip(meta, probs, invalid):
        if task == "h1_rating":
            pt = harsh_high(expected_rating(p, list(range(1, 8))), target)
        else:
            pt = float("nan") if np.isnan(p).any() else float(p[OPTIONS[task].index(target)])
        rows.append({"item_id": item_id, "variant": variant, "p_target": pt, "invalid": float(inv)})
    return rows


def user_messages(cond, meta: list[tuple]) -> list[str]:
    if cond.kind == "text":
        return [cond.induce + " " + m[2] for m in meta]
    return [m[2] for m in meta]


def score_path(out_dir, task: str, cond_name: str) -> Path:
    return Path(out_dir) / f"{task}__{cond_name}.csv"


def _write_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_all(conditions, tasks: dict, out_dir, score_fn, log=print) -> dict:
    """tasks: {task: items}. score_fn(cond, user_msgs, options) -> (probs, invalid).
    Skips existing final files; writes each file as soon as it is computed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.tmp"):
        stale.unlink()
    prompts = {t: task_prompts(t, items) for t, items in tasks.items()}
    done = skipped = n_prompts = 0
    t0 = time.time()
    total = len(conditions) * len(tasks)
    for cond in conditions:
        for task, meta in prompts.items():
            path = score_path(out_dir, task, cond.name)
            if path.exists():
                skipped += 1
                continue
            ts = time.time()
            probs, invalid = score_fn(cond, user_messages(cond, meta), OPTIONS[task])
            _write_atomic(path, rows_from_scores(task, meta, probs, invalid))
            done += 1
            n_prompts += len(meta)
            log(f"[{done + skipped}/{total}] {cond.name} {task}: {len(meta)} prompts, "
                f"{time.time() - ts:.1f}s")
    return {"done": done, "skipped": skipped, "prompts": n_prompts, "elapsed": time.time() - t0}


def check_pilot_out_dir(pilot: bool, out_dir, real_dir) -> Path:
    """A pilot (--conditions / --limit-items) must write to its own --out-dir."""
    if not pilot:
        return Path(real_dir)
    if out_dir is None:
        raise SystemExit("先導執行（--conditions 或 --limit-items）必須指定 --out-dir，"
                         "避免部分題目的檔案被當成正式結果。")
    if Path(out_dir).resolve() == Path(real_dir).resolve():
        raise SystemExit("先導執行的 --out-dir 不可以和正式的 scores 資料夾相同。")
    return Path(out_dir)


def preflight(frozen_path, hash_files: dict, random_path_fn) -> dict:
    """Refuse (SystemExit) before the model is loaded; returns the frozen config.
    random_path_fn(frozen) -> path of the random directions file."""
    frozen_path = Path(frozen_path)
    if not frozen_path.exists():
        raise SystemExit(f"找不到凍結檔 {frozen_path}，請先完成第 6.5 步的校準並凍結設定。")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    check_frozen_hashes(frozen, hash_files, allow_changed=False)
    rnd = Path(random_path_fn(frozen))
    if not rnd.exists():
        raise SystemExit(f"找不到隨機方向檔 {rnd}；本程式不會重新產生它。")
    return frozen
