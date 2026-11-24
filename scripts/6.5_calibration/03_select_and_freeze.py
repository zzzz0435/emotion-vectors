"""6.5 選層、α 與凍結設定（設計文件 §6.5，2026-11-22 修訂）。

第一階段：讀 metrics.csv，先確認它是完整、在網格上的校準結果（vectors.pt 的每個候選層 ×
  核心情緒都從最小 α 連續測到最大 α 或第一個不通順的 α；否則拒絕），再依事先寫定的規則
  選層和 α_max，寫出 selection.json。
  若 control_ratings.csv 在（該層, α_max）還沒有完整的 60 個對照向量評分，印出要執行的
  02 --controls 指令並以代碼 3 結束。control_meta.json 的層必須和選出的層相同。
第二階段：判斷每種核心情緒是否成功誘發（需要時往上找備案 α；備案 α 缺對照評分時同樣印出
  指令並以代碼 3 結束），計算等強度 α_eq，組出凍結設定。
最後：stats.py、conditions.py 或三個資料集有尚未提交的修改就拒絕凍結；用 build_conditions
  檢查設定可用，再寫出 frozen_config.json（已存在則拒絕，除非 --force：舊檔改名保留）。

用法：python scripts/6.5_calibration/03_select_and_freeze.py --model qwen3-8b [--force]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import calibration, conditions, freeze, models  # noqa: E402
from emolab.paths import REPO_ROOT, RESULTS  # noqa: E402

NEED_CONTROLS = 3


def controls_command(model_key: str, layer: int, alpha: float) -> str:
    return (f"python scripts/6.5_calibration/02_measure_calibration.py --model {model_key} "
            f"--controls --layer {layer} --alpha {alpha:g}")


def _need_controls(model_key: str, layer: int, alpha: float, why: str) -> None:
    print(f"{why}\n請先執行下面的指令產生對照向量評分，完成後再執行一次本腳本：\n"
          f"{controls_command(model_key, layer, alpha)}", flush=True)
    raise SystemExit(NEED_CONTROLS)


def _json(path: Path):
    if not path.exists():
        raise SystemExit(f"找不到必要的輸入檔 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _torch(path: Path):
    import torch
    if not path.exists():
        raise SystemExit(f"找不到必要的輸入檔 {path}")
    return torch.load(path, weights_only=False)


def _check_control_layer(cal_dir: Path, layer: int) -> None:
    meta, ratings = cal_dir / "control_meta.json", cal_dir / "control_ratings.csv"
    if meta.exists():
        old = json.loads(meta.read_text(encoding="utf-8")).get("layer")
        if old != layer:
            raise SystemExit(
                f"{cal_dir} 裡的對照向量結果屬於第 {old} 層，但選出的是第 {layer} 層。請先把 "
                "control_texts.jsonl、control_ratings.csv、control_meta.json 移到別處，再重跑對照。")
    elif ratings.exists():
        raise SystemExit(f"{ratings} 存在但沒有 control_meta.json，無法確認它屬於哪一層；請先移走。")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--force", action="store_true",
                    help="凍結檔已存在時仍重新凍結（舊檔改名保留）")
    ap.add_argument("--results-dir", type=Path, default=RESULTS, help="results 根目錄（測試用）")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                    help="計算 sha256 和檢查 git 狀態的 repo 根目錄（測試用）")
    args = ap.parse_args(argv)
    name = models.MODELS[args.model].name
    cal_dir = args.results_dir / "6.5_calibration" / name
    vec_dir = args.results_dir / "6.2_emotion_vectors" / name
    ax_dir = args.results_dir / "6.3_axes" / name
    frozen_path = cal_dir / freeze.FROZEN_NAME
    if frozen_path.exists() and not args.force:
        raise SystemExit(f"凍結檔已存在（{frozen_path}），不會覆寫。若確定要重新凍結，"
                         "請加上 --force（舊檔會改名保留）。")
    labels = conditions.load_labels()
    core = labels["core"]

    # ---- phase 1: layer and alpha_max ----
    if not (cal_dir / "metrics.csv").exists():
        raise SystemExit(f"找不到 {cal_dir / 'metrics.csv'}：請先跑完 01 和 02。")
    vec = _torch(vec_dir / "vectors.pt")
    try:
        metrics = freeze.load_metrics(cal_dir / "metrics.csv")
        freeze.validate_metrics(metrics, vec["layers"], core)
    except (ValueError, KeyError) as exc:
        raise SystemExit(
            f"{cal_dir / 'metrics.csv'} 不是完整、在網格上的校準結果，拒絕選層"
            f"（候選層 {list(vec['layers'])}，α 網格 {calibration.ALPHA_GRID}）。"
            "請確認 01、02 都已跑完；每個層 × 核心情緒都要從最小的 α 連續測到最大的 α，"
            "或停在第一個不通順的 α，而且不能有重複、缺漏、不在網格上的 α 或空白的通順欄位。\n"
            f"{exc}") from exc
    contamination = _json(vec_dir / "contamination.json")
    table = freeze.layer_table(metrics, contamination, core)
    layer = freeze.choose_layer(metrics, contamination, core)
    amax = freeze.alpha_max(metrics, layer, core)
    cmax = {e: freeze.coherent_max(metrics, layer, e) for e in core}
    selection = {"model": name, "layer": layer, "alpha_max": amax, "coherent_max": cmax,
                 "layer_table": table, "created_at": _now()}
    (cal_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        newline="\n")
    for r in table:
        mg = "—" if r["mean_gain"] is None else f"{r['mean_gain']:.3f}"
        print(f"L{r['layer']}: alpha_max={r['alpha_max']:g} mean_gain={mg} "
              f"flagged={r['n_flagged']}", flush=True)
    print(f"選出第 {layer} 層，α_max = {amax:g}（已寫入 selection.json）", flush=True)

    _check_control_layer(cal_dir, layer)
    controls = freeze.load_controls(cal_dir / "control_ratings.csv")
    if not freeze.controls_complete(controls, amax, core):
        _need_controls(args.model, layer, amax,
                       f"第 {layer} 層 α = {amax:g} 還沒有完整的 {freeze.N_CONTROLS} 個對照向量評分。")

    # ---- phase 2: induction, alpha_eq, payload ----
    try:
        ind = freeze.induction(metrics, controls, layer, amax, cmax, core)
    except freeze.NeedControls as exc:
        _need_controls(args.model, layer, exc.alpha,
                       f"有情緒在 α_max 下沒過誘發標準，備案強度 α = {exc.alpha:g} 需要對照向量評分。")
    for e, d in ind.items():
        print(f"{e}: {d['status']} alpha={d['alpha']} gain={d['gain']:.3f} "
              f"rating={d['rating']:.3f} p95={d['p95']:.3f}", flush=True)
    target, eq = freeze.alpha_eq(metrics, layer, cmax, ind, core)
    residuals = _torch(ax_dir / "residuals.pt")
    payload = freeze.build_payload(
        model=name, layer=layer, alpha_max=amax, coherent=cmax, induction=ind,
        eq_target=target, alpha_eq=eq, layer_table=table, contamination=contamination,
        separability=_json(vec_dir / "separability.json"),
        residual_status=residuals["status"], axes_meta=_json(ax_dir / "axes.json"),
        resid_norm=vec["resid_norm"], hashes=freeze.file_hashes(args.repo_root),
        git_commit=freeze.git_commit(args.repo_root), created_at=_now())

    dirty = freeze.uncommitted(args.repo_root)
    if dirty:
        raise SystemExit("拒絕凍結：下列被記錄 sha256 的檔案有尚未提交（commit）的修改，"
                         "請先提交再凍結：\n" + "\n".join(dirty))
    conds = conditions.build_conditions(payload, labels)
    print(f"依凍結設定共有 {len(conds)} 個實驗條件。", flush=True)
    try:
        freeze.write_frozen(frozen_path, payload, force=args.force)
    except FileExistsError as exc:
        raise SystemExit(f"拒絕覆寫凍結檔：{exc}") from exc
    note = f"（取代 {payload['supersedes']}）" if "supersedes" in payload else ""
    print(f"已寫入 {frozen_path}{note}", flush=True)


if __name__ == "__main__":
    main()
