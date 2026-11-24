"""6.7 道德題實驗：每個條件 x 每個任務寫一個 scores/<task>__<condition>.csv（可續跑、原子寫入）。

正式執行：
  python scripts/6.7_runs/01_run_conditions.py --model qwen3-8b
先導（3 個條件、10 題，輸出到獨立資料夾）：
  python scripts/6.7_runs/01_run_conditions.py --model qwen3-8b --conditions baseline,anger@+0.6000,text_anger --limit-items 10 --out-dir results/6.7_runs/Qwen_3_8B_instruct/pilot
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import conditions as cmod, models, runner  # noqa: E402
from emolab.paths import DATASETS, RESULTS  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--conditions", help="逗號分隔的條件名稱（先導用）")
    ap.add_argument("--limit-items", type=int, help="每個任務只取前 N 題（先導用）")
    ap.add_argument("--out-dir", help="先導時必填，且不可等於正式 scores 資料夾")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    name = models.MODELS[args.model].name
    run_dir = RESULTS / "6.7_runs" / name
    real_dir = run_dir / "scores"
    pilot = args.conditions is not None or args.limit_items is not None
    out_dir = runner.check_pilot_out_dir(pilot, args.out_dir, real_dir)

    vig_path = DATASETS / "6.6_punishment_vignettes.json"
    mc_path = DATASETS / "6.6_moralchoice_harm.json"
    frozen = runner.preflight(
        RESULTS / "6.5_calibration" / name / "frozen_config.json",
        {"conditions_sha256": Path(cmod.__file__),
         "vignettes_sha256": vig_path, "moralchoice_sha256": mc_path},
        lambda fz: run_dir / f"random_directions_L{fz['layer']}.npy")

    all_conds = cmod.build_conditions(frozen, cmod.load_labels())
    conds = all_conds
    if args.conditions:
        by_name = {c.name: c for c in all_conds}
        wanted = [c for c in args.conditions.split(",") if c]
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"不存在的條件：{missing}")
        conds = [by_name[w] for w in wanted]

    vignettes = json.loads(vig_path.read_text(encoding="utf-8"))
    moral = json.loads(mc_path.read_text(encoding="utf-8"))
    full_per_cond = 2 * len(vignettes) * 2 + 2 * len(moral)
    if args.limit_items:
        vignettes, moral = vignettes[:args.limit_items], moral[:args.limit_items]
    tasks = {"h1_rating": vignettes, "h1_choice": vignettes, "h2": moral}

    import torch
    from emolab.scoring import SteerSpec, score_prompts

    layer = frozen["layer"]
    stores = cmod.load_stores(name, layer)
    model, tok = models.load(args.model)

    def score_fn(cond, msgs, options):
        d = cmod.resolve_direction(cond, layer, stores)
        spec = None if d is None else SteerSpec(
            layer, torch.from_numpy(d).float(), cond.alpha, frozen["resid_norm"])
        return score_prompts(model, tok, msgs, options, batch_size=args.batch_size, steer_spec=spec)

    stats = runner.run_all(conds, tasks, out_dir, score_fn)
    if stats["prompts"] and stats["elapsed"] > 0:
        pps = stats["prompts"] / stats["elapsed"]
        hours = len(all_conds) * full_per_cond / pps / 3600
        print(f"速度：{pps:.2f} prompts/s；全部 {len(all_conds)} 個條件 x {full_per_cond} 題"
              f"預估 {hours:.1f} 小時")
    print(f"完成 {stats['done']} 個檔案，略過 {stats['skipped']} 個，輸出：{out_dir}")


if __name__ == "__main__":
    main()
