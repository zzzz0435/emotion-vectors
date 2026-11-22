"""6.8 結果摘要：讀 6.8/01 的輸出和凍結檔，寫出繁體中文的
results/6.8_analysis/<模型名>/results_summary.md（給使用者讀的主要成果）。
任何輸入缺少就用中文訊息拒絕執行。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import results_report  # noqa: E402
from emolab.paths import RESULTS  # noqa: E402

DEFAULT_MODEL = "Qwen_3_8B_instruct"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--frozen", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir or RESULTS / "6.8_analysis" / args.model_dir
    frozen_path = args.frozen or RESULTS / "6.5_calibration" / args.model_dir / "frozen_config.json"
    if not frozen_path.exists():
        raise SystemExit(f"找不到凍結檔 {frozen_path}：請先完成校準。")
    try:
        primary, secondary, explo, interp = results_report.load_analysis(out)
    except FileNotFoundError as exc:
        raise SystemExit(f"缺少 6.8/01 的輸出（{exc}）：請先執行 6.8_analysis/01_stats.py。")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    text = results_report.render_results(primary, secondary, explo, interp, frozen)
    dest = out / "results_summary.md"
    dest.write_text(text, encoding="utf-8", newline="\n")
    missing = [t for t, p in results_report.FIGURES if not (out / p).exists()]
    if missing:
        print(f"提醒：還沒有這些圖（請先執行 02_figures.py）：{'、'.join(missing)}", flush=True)
    print(f"寫入 {dest}", flush=True)


if __name__ == "__main__":
    main()
