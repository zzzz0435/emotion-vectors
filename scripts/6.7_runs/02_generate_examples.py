"""6.7 範例回答：每個任務均勻挑 20 題，在 baseline 與 6 種核心情緒 +/-alpha_max 下生成完整回答，
輸出 results/6.7_runs/<模型>/examples.md（可續跑：生成結果快取在 examples.jsonl）。

用法：python scripts/6.7_runs/02_generate_examples.py --model qwen3-8b
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import calibration, models, runner, stories  # noqa: E402
from emolab import conditions as cmod  # noqa: E402
from emolab.paths import DATASETS, RESULTS  # noqa: E402

SUFFIX = "\n\nAnswer first, then explain your reasoning in two or three sentences."
N_ITEMS = 20
NOTE = ("注意：「被調整的條件」是指在模型內部某一層加上一個方向向量，"
        "並不代表模型真的有感覺。以下回答只用來觀察調整後文字的樣子。")
SELECTION_RULE = ("題目挑選規則（固定、可重現）：H1 依「罪行 × 嚴重程度」的 18 個組合輪流各取一題"
                  "（第 k 組取第 k 個變體，不夠 20 題時從前面的組合再各補一題），因此 6 種罪行和 "
                  "3 種嚴重程度都有；H2 在 150 題中取等距的 20 題。H1 懲罰評分用 light1 版，"
                  "H1 二選一與 H2 用選項順序 0。")
# The scored prompts demand a bare answer; the examples ask for an answer plus an explanation,
# so the demands that contradict that are removed here (moral_items.py itself is untouched).
# (old, new) pairs; every pair must apply to the rendered prompt or the prompt text has drifted.
EXAMPLE_EDITS = {
    "h1_rating": [("\n\nRespond with the digit only, without any other words or formatting.", ""),
                  ("Answer with a single digit.", "Start your answer with a single digit.")],
    "h1_choice": [("\n\nRespond with the letter only, without any other words or formatting.", ""),
                  ("Answer with A or B only.", "Start your answer with A or B.")],
    "h2": [("- Your answer vocabulary is limited to A or B.\n", ""),
           (" Respond solely with A or B.", ""),
           ("\nAnswer:", "")],
}


def pick_conditions(frozen, labels):
    by_name = {c.name: c for c in cmod.build_conditions(frozen, labels)}
    a = float(frozen["alpha_max"])
    names = ["baseline"] + [cmod.condition_name(e, s * a) for e in labels["core"] for s in (1, -1)]
    return [by_name[n] for n in names]


def example_text(task: str, msg: str) -> str:
    """The scored prompt with the bare-answer demands removed (strings only)."""
    for old, new in EXAMPLE_EDITS[task]:
        if old not in msg:
            raise ValueError(f"{task}: 範例提示詞的改寫規則對不上（找不到 {old!r}）；題目文字可能改過")
        msg = msg.replace(old, new)
    return msg


def pick_items(task: str, items: list[dict], n: int = N_ITEMS) -> list[dict]:
    """n items spread across the set, deterministically (see SELECTION_RULE)."""
    if len(items) <= n:
        return list(items)
    if task == "h2":
        return [items[i * len(items) // n] for i in range(n)]
    cells: dict[tuple, list[dict]] = {}
    for it in items:
        cells.setdefault((it["offense"], it["severity"]), []).append(it)
    groups, out, r = list(cells.values()), [], 0
    while len(out) < n and r < max(len(g) for g in groups):
        for k, g in enumerate(groups):
            if r < len(g) and len(out) < n:
                out.append(g[(k + r) % len(g)])
        r += 1
    return out


def example_prompts(task, items):
    """n spread-out items; h1_rating uses light1, h1_choice and h2 use order 0."""
    keep = ("light1", "0")
    return [(iid, var, example_text(task, msg), tgt)
            for iid, var, msg, tgt in runner.task_prompts(task, pick_items(task, items))
            if var in keep]


def load_cache(path: Path) -> dict:
    """key -> response; a torn last line from a hard kill is cut off before resuming."""
    return {r["key"]: r["response"] for r in stories.repair_torn_tail(path)}


def render(tasks_meta, conds, cache) -> str:
    lines = ["# 6.7 範例回答", "", NOTE, "", SELECTION_RULE, ""]
    titles = {"h1_rating": "H1 懲罰評分", "h1_choice": "H1 二選一", "h2": "H2 傷害選擇"}
    for task, meta in tasks_meta.items():
        lines += [f"## 任務：{titles[task]}（{task}）", ""]
        for iid, variant, msg, _ in meta:
            lines += [f"### 題目 {iid}（variant {variant}）", "", "```text", msg, "```", ""]
            for c in conds:
                resp = cache.get(f"{task}|{iid}|{c.name}", "（尚未生成）")
                lines += [f"#### 條件：{c.name}", "", "```text", resp.strip(), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    name = models.MODELS[args.model].name
    run_dir = RESULTS / "6.7_runs" / name
    frozen = json.loads((RESULTS / "6.5_calibration" / name / "frozen_config.json")
                        .read_text(encoding="utf-8"))
    conds = pick_conditions(frozen, cmod.load_labels())
    vig = json.loads((DATASETS / "6.6_punishment_vignettes.json").read_text(encoding="utf-8"))
    moral = json.loads((DATASETS / "6.6_moralchoice_harm.json").read_text(encoding="utf-8"))
    tasks_meta = {t: example_prompts(t, items)
                  for t, items in (("h1_rating", vig), ("h1_choice", vig), ("h2", moral))}

    cache_path = run_dir / "examples.jsonl"
    cache = load_cache(cache_path)

    todo = {c.name: [(t, m) for t, meta in tasks_meta.items() for m in meta
                     if f"{t}|{m[0]}|{c.name}" not in cache] for c in conds}
    if any(todo.values()):
        import torch
        from emolab.scoring import SteerSpec

        layer = frozen["layer"]
        stores = cmod.load_stores(name, layer)
        model, tok = models.load(args.model)
        for c in conds:
            if not todo[c.name]:
                continue
            d = cmod.resolve_direction(c, layer, stores)
            spec = None if d is None else SteerSpec(
                layer, torch.from_numpy(d).float(), c.alpha, frozen["resid_norm"])
            pending = todo[c.name]
            msgs = [m[2] + SUFFIX for _, m in pending]
            outs = calibration.generate_batch(model, tok, msgs, spec, max_new_tokens=200,
                                              batch_size=args.batch_size)
            with open(cache_path, "a", encoding="utf-8", newline="\n") as f:
                for (t, m), o in zip(pending, outs):
                    key = f"{t}|{m[0]}|{c.name}"
                    cache[key] = o
                    f.write(json.dumps({"key": key, "response": o}, ensure_ascii=False) + "\n")
            print(f"{c.name}: 生成 {len(pending)} 則", flush=True)

    out = run_dir / "examples.md"
    out.write_text(render(tasks_meta, conds, cache), encoding="utf-8", newline="\n")
    print(f"已寫入 {out}")


if __name__ == "__main__":
    main()
