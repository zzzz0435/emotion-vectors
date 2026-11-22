"""Layer / alpha selection, induction check, equal-intensity alpha and the frozen configuration
(design spec §6.5, amended 2026-11-22), plus the checkpoint-2 summary text.

Tables (as written by 6.5_calibration/02):
- metrics: calibration.METRICS_COLS, one row per (layer, alpha, label); alpha = 0 rows have
  gain 0 and coherent True. Larger alphas may be missing (early stop after incoherence).
- control ratings: calibration.CONTROL_COLS, one row per (alpha, control, emotion); all rows
  belong to the layer in control_meta.json.

Rules (binding, see task-8 brief):
- coherent_max: largest alpha > 0 before the first incoherent one (ascending), else 0.0.
- alpha_max: min of coherent_max over the 6 core emotions.
- choose_layer: largest mean core gain at the layer's own alpha_max; ties (isclose, abs 1e-9)
  -> fewer flagged emotions -> smaller layer.
- induction (R13): ok = gain >= 1 and rating > p95 of exactly 60 controls (strict); else
  walk grid alphas in (alpha_max, coherent_max[e]] for the first that passes ("fallback");
  else "failed". Never above an emotion's own coherent_max (pain: ethics constraint).
- alpha_eq (R17): T = min(2, min over non-failed of best gain within coherent_max); first
  crossing of gain = T from alpha 0, linear interpolation, 4 decimals.
"""
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from emolab import axes as axes_mod
from emolab import calibration as cal
from emolab import integrity, stats

INDUCTION_GAIN = 1.0
CONTROL_PERCENTILE = 95
N_CONTROLS = 60
EQ_TARGET_MAX = 2.0
TIE_TOL = 1e-9
ALPHA_TOL = 1e-9
# 72 punishment vignettes x (2 rating directions + 2 choice orders) + 150 MoralChoice x 2 orders
PASSES_PER_CONDITION = 588
DEFAULT_RATE = 6.0
FROZEN_NAME = "frozen_config.json"
HASHED = {"stats_sha256": "scripts/emolab/stats.py",
          "conditions_sha256": "scripts/emolab/conditions.py",
          "concepts_sha256": "datasets/6.2_concepts.json",
          "vignettes_sha256": "datasets/6.6_punishment_vignettes.json",
          "moralchoice_sha256": "datasets/6.6_moralchoice_harm.json"}
THRESHOLDS = {"format_min": cal.FORMAT_MIN, "fluency_drop": cal.FLUENCY_DROP,
              "induction_gain": INDUCTION_GAIN, "control_percentile": CONTROL_PERCENTILE,
              "eq_target_max": EQ_TARGET_MAX, "holm_alpha": stats.SIG,
              "gate_need": stats.GATE_NEED, "bootstrap_n": 10000, "seed": 42}
COHERENCE_RULE = (f"通順＝格式題有效答案機率總和 ≥ {cal.FORMAT_MIN}，且未調整的主模型給文字打的"
                  f"通順程度（1–5 分）平均不低於未調整時減 {cal.FLUENCY_DROP} 分；困惑度比只記錄、"
                  "不作門檻（2026-11-22 修訂，在任何道德題資料產生之前）。")


class NeedControls(ValueError):
    """Control ratings are missing or incomplete at `alpha`: run 02 --controls there."""

    def __init__(self, alpha: float, msg: str | None = None):
        self.alpha = float(alpha)
        super().__init__(msg or f"control ratings needed at alpha {self.alpha:g}")


# ---------- loading ----------

def _truthy(v) -> bool:
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)


def load_metrics(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["layer"] = df["layer"].astype(int)
    df["alpha"] = df["alpha"].astype(float)
    df["label"] = df["label"].astype(str)
    df["coherent"] = df["coherent"].map(_truthy)
    return df


def load_controls(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=cal.CONTROL_COLS)
    df = pd.read_csv(path, dtype={"control": str, "emotion": str})
    df["alpha"] = df["alpha"].astype(float)
    df["rating"] = df["rating"].astype(float)
    return df


def _core(emotions) -> list[str]:
    if emotions is not None:
        return list(emotions)
    from emolab.conditions import load_labels
    return load_labels()["core"]


def _layer_get(d: dict, layer: int):
    if layer in d:
        return d[layer]
    if str(layer) in d:
        return d[str(layer)]
    raise ValueError(f"no entry for layer {layer} (keys: {sorted(map(str, d))})")


# ---------- metrics lookups ----------

def _steps(metrics: pd.DataFrame, layer: int, emotion: str) -> pd.DataFrame:
    """alpha > 0 rows of (layer, emotion), alpha ascending."""
    m = metrics[(metrics["layer"].astype(int) == int(layer)) & (metrics["label"] == emotion)
                & (metrics["alpha"].astype(float) > 0)]
    return m.sort_values("alpha", kind="stable")


def _at(metrics: pd.DataFrame, layer: int, emotion: str, alpha: float) -> pd.Series:
    s = _steps(metrics, layer, emotion)
    hit = s[np.isclose(s["alpha"].astype(float), alpha, rtol=0, atol=ALPHA_TOL)]
    if len(hit) != 1:
        raise ValueError(f"metrics has {len(hit)} rows for layer {layer}, {emotion}, "
                         f"alpha {alpha:g} (need exactly 1)")
    return hit.iloc[0]


def coherent_max(metrics: pd.DataFrame, layer: int, emotion: str) -> float:
    best = 0.0
    s = _steps(metrics, layer, emotion)
    for a, ok in zip(s["alpha"], s["coherent"]):
        if not _truthy(ok):
            break
        best = float(a)
    return best


def alpha_max(metrics: pd.DataFrame, layer: int, emotions=None) -> float:
    return min(coherent_max(metrics, layer, e) for e in _core(emotions))


def flagged_at(contamination: dict, layer: int) -> list[str]:
    """Emotions flagged for any probe at `layer` (all labels in contamination.json)."""
    if str(layer) not in contamination:
        raise ValueError(f"contamination.json has no layer {layer}")
    return sorted({e for flags in contamination[str(layer)].values()
                   for e, f in flags.items() if f.get("flagged")})


def layer_table(metrics: pd.DataFrame, contamination: dict, emotions=None) -> list[dict]:
    """Per layer: alpha_max, mean core gain at it (None when alpha_max is 0), flagged count."""
    emotions = _core(emotions)
    rows = []
    for layer in sorted({int(x) for x in metrics["layer"]}):
        am = alpha_max(metrics, layer, emotions)
        mg = None
        if am > 0:
            gains = [float(_at(metrics, layer, e, am)["gain"]) for e in emotions]
            if not all(np.isfinite(gains)):
                raise ValueError(f"layer {layer}: missing gain at alpha_max {am:g}")
            mg = float(np.mean(gains))
        rows.append({"layer": layer, "alpha_max": am, "mean_gain": mg,
                     "n_flagged": len(flagged_at(contamination, layer))})
    return rows


def choose_layer(metrics: pd.DataFrame, contamination: dict, emotions=None) -> int:
    cands = [r for r in layer_table(metrics, contamination, emotions) if r["alpha_max"] > 0]
    if not cands:
        raise ValueError("no candidate layer keeps all core emotions coherent at any alpha")
    best = max(r["mean_gain"] for r in cands)
    tied = [r for r in cands if math.isclose(r["mean_gain"], best, abs_tol=TIE_TOL)]
    return int(min(tied, key=lambda r: (r["n_flagged"], r["layer"]))["layer"])


# ---------- controls and induction (R13) ----------

def control_values(controls: pd.DataFrame, alpha: float, emotion: str) -> np.ndarray:
    """Ratings of the 60 controls for `emotion` at `alpha`. Fewer than 60 -> NeedControls
    (a ValueError: rerun 02 --controls, which resumes); more, duplicates or NaN -> ValueError."""
    sel = controls[np.isclose(controls["alpha"].astype(float), alpha, rtol=0, atol=ALPHA_TOL)
                   & (controls["emotion"] == emotion)]
    if sel["control"].duplicated().any():
        raise ValueError(f"duplicate control rows at alpha {alpha:g} for {emotion}")
    if len(sel) < N_CONTROLS:
        raise NeedControls(alpha, f"only {len(sel)} of {N_CONTROLS} control ratings at alpha "
                                  f"{alpha:g} for {emotion}")
    if len(sel) > N_CONTROLS:
        raise ValueError(f"{len(sel)} control ratings at alpha {alpha:g} for {emotion}; "
                         f"exactly {N_CONTROLS} required")
    vals = sel["rating"].astype(float).to_numpy()
    if not np.isfinite(vals).all():
        raise ValueError(f"NaN control rating at alpha {alpha:g} for {emotion}")
    return vals


def control_p95(controls: pd.DataFrame, alpha: float, emotion: str) -> float:
    return float(np.percentile(control_values(controls, alpha, emotion), CONTROL_PERCENTILE))


def controls_complete(controls: pd.DataFrame, alpha: float, emotions) -> bool:
    try:
        for e in emotions:
            control_values(controls, alpha, e)
    except NeedControls:
        return False
    return True


def _detail(status: str, alpha, row: pd.Series, p95: float) -> dict:
    return {"status": status, "alpha": None if alpha is None else float(alpha),
            "gain": float(row["gain"]), "rating": float(row["rating_mean"]), "p95": float(p95)}


def _passes(row: pd.Series, p95: float) -> bool:
    return bool(float(row["gain"]) >= INDUCTION_GAIN and float(row["rating_mean"]) > p95)


def induction(metrics: pd.DataFrame, control_ratings: pd.DataFrame, layer: int,
              alpha_max: float, coherent_max: dict, emotions=None) -> dict[str, dict]:
    """{emotion: {status, alpha, gain, rating, p95}}; a failed emotion reports its alpha_max
    numbers. Raises NeedControls(alpha) when a fallback alpha has no (complete) controls."""
    emotions = list(coherent_max) if emotions is None else list(emotions)
    out = {}
    for e in emotions:
        row = _at(metrics, layer, e, alpha_max)
        p95 = control_p95(control_ratings, alpha_max, e)
        if _passes(row, p95):
            out[e] = _detail("ok", alpha_max, row, p95)
            continue
        out[e] = _detail("failed", None, row, p95)
        for a in sorted(cal.ALPHA_GRID):
            if a <= alpha_max + ALPHA_TOL or a > coherent_max[e] + ALPHA_TOL:
                continue
            row_a = _at(metrics, layer, e, a)
            if float(row_a["gain"]) < INDUCTION_GAIN:
                continue
            p95_a = control_p95(control_ratings, a, e)
            if _passes(row_a, p95_a):
                out[e] = _detail("fallback", a, row_a, p95_a)
                break
    return out


# ---------- equal-intensity alpha (R17) ----------

def _status(v) -> str:
    return v["status"] if isinstance(v, dict) else v


def _curve(metrics: pd.DataFrame, layer: int, emotion: str, cm: float) -> list[tuple]:
    s = _steps(metrics, layer, emotion)
    s = s[s["alpha"].astype(float) <= cm + ALPHA_TOL]
    return [(0.0, 0.0)] + [(float(a), float(g)) for a, g in zip(s["alpha"], s["gain"])]


def _crossing(curve: list[tuple], target: float) -> float | None:
    for (a0, g0), (a1, g1) in zip(curve, curve[1:]):
        if g1 >= target:
            if g0 >= target:
                return round(a0, 4)
            return round(a0 + (target - g0) * (a1 - a0) / (g1 - g0), 4)
    return None


def alpha_eq(metrics: pd.DataFrame, layer: int, coherent_max: dict, induction: dict,
             emotions=None) -> tuple[float, dict[str, float | None]]:
    emotions = list(coherent_max) if emotions is None else list(emotions)
    live = [e for e in emotions if _status(induction[e]) != "failed"]
    out: dict[str, float | None] = {e: None for e in emotions}
    if not live:
        return float("nan"), out
    curves = {e: _curve(metrics, layer, e, coherent_max[e]) for e in live}
    target = min(EQ_TARGET_MAX, min(max(g for _, g in c) for c in curves.values()))
    for e in live:
        out[e] = _crossing(curves[e], target)
    return float(target), out


# ---------- frozen payload ----------

def removed_axes_at(axes_meta: dict, residual_status: dict, layer: int) -> dict[str, list]:
    """Residual name -> reliable axes of its `which`, in order; [] when not "ok"."""
    out = {}
    for name, (_, fit_set, which) in axes_mod.RESIDUALS.items():
        if _layer_get(residual_status[name], layer) != "ok":
            out[name] = []
        else:
            out[name] = axes_mod.removed_axes(axes_meta["fits"][fit_set][str(layer)], which)
    return out


def file_hashes(root) -> dict[str, str]:
    return {k: integrity.file_sha256(Path(root) / rel) for k, rel in HASHED.items()}


def build_payload(*, model: str, layer: int, alpha_max: float, coherent: dict,
                  induction: dict, eq_target: float, alpha_eq: dict, layer_table: list,
                  contamination: dict, separability: dict, residual_status: dict,
                  axes_meta: dict, resid_norm: dict, hashes: dict, git_commit: str,
                  created_at: str) -> dict:
    """The frozen configuration (pure). Emotion order follows `induction`."""
    layer = int(layer)
    emotions = list(induction)
    sep = separability[str(layer)]
    target = None if eq_target is None or not math.isfinite(eq_target) else float(eq_target)
    payload = {
        "model": model,
        "layer": layer,
        "resid_norm": float(_layer_get(resid_norm, layer)),
        "alpha_max": float(alpha_max),
        "alpha_eq": {e: None if alpha_eq[e] is None else float(alpha_eq[e]) for e in emotions},
        "alpha_eq_target": target,
        "alpha_individual": {e: float(d["alpha"]) for e, d in induction.items()
                             if d["status"] == "fallback"},
        "induction": {e: d["status"] for e, d in induction.items()},
        "induction_detail": {e: dict(d) for e, d in induction.items()},
        "coherent_max": {e: float(coherent[e]) for e in emotions},
        "thresholds": dict(THRESHOLDS),
        "grid": list(cal.ALPHA_GRID),
        "contamination_at_layer": flagged_at(contamination, layer),
        "separability_at_layer": {"auroc": float(sep["auroc"]), "passed": bool(sep["separable"])},
        "residual_status_at_layer": {n: _layer_get(residual_status[n], layer)
                                     for n in axes_mod.RESIDUALS},
        "residual_removed_axes": removed_axes_at(axes_meta, residual_status, layer),
        "layer_table": [dict(r) for r in layer_table],
        **{k: hashes[k] for k in HASHED},
        "git_commit": git_commit,
        "created_at": created_at,
        "coherence_rule": COHERENCE_RULE,
    }
    need = (*stats.KEYS_PRIMARY, *stats.KEYS_SECONDARY, *stats.KEYS_INTERPRET)
    missing = [k for k in need if k not in payload]
    if missing:
        raise AssertionError(f"frozen payload lacks keys required by stats.py: {missing}")
    return payload


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def write_frozen(path, payload: dict, force: bool = False) -> None:
    """Write JSON (indent 2, UTF-8, LF, trailing newline). An existing file is never
    overwritten; with force it is renamed to <stem>.superseded-<UTC>.json, and the new file
    records that name under "supersedes"."""
    path = Path(path)
    old = None
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; the frozen configuration is not "
                                  "overwritten (use --force to supersede it)")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        old = path.with_name(f"{path.stem}.superseded-{stamp}{path.suffix}")
        if old.exists():
            raise FileExistsError(f"{old} already exists; retry in a second")
        payload["supersedes"] = old.name
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2, allow_nan=False)
                   + "\n", encoding="utf-8", newline="\n")
    if old is not None:
        path.rename(old)
    tmp.replace(path)


# ---------- git ----------

def _git(root, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {r.stderr.strip()}")
    return r.stdout


def git_commit(root) -> str:
    return _git(root, "rev-parse", "HEAD").strip()


def uncommitted(root, rel_paths=None) -> list[str]:
    """`git status --porcelain` lines for the hashed files (modified, staged or untracked)."""
    rel = list(HASHED.values()) if rel_paths is None else list(rel_paths)
    return [ln for ln in _git(root, "status", "--porcelain", "--", *rel).splitlines()
            if ln.strip()]


# ---------- checkpoint-2 summary ----------

STATUS_TEXT = {"ok": "成功", "fallback": "強度不同", "failed": "未能誘發"}
EMOTION_ZH = {"anger": "憤怒", "fear": "恐懼", "sadness": "悲傷", "joy": "喜悅",
              "compassion": "同情", "pain": "疼痛"}
# which hypotheses use each core emotion (design spec §2, stats._primary_specs/_secondary_specs)
HYP_FAILED = {"anger": "H1、H3、H4 會寫成「無法判斷」（次要比較「憤怒 vs 悲傷」也是）",
              "fear": "H3 會寫成「無法判斷」",
              "pain": "H2 會寫成「無法判斷」（疼痛殘差等次要檢驗也是）",
              "compassion": "不影響 H1–H4；次要假設 H5 和同情殘差檢驗會寫成「無法判斷」",
              "joy": "不影響 H1–H4；次要假設 H5 會寫成「無法判斷」",
              "sadness": "不影響 H1–H4；次要比較「憤怒 vs 悲傷」會寫成「無法判斷」"}
HYP_FALLBACK = {"anger": "H1 改用這個強度，結果標示「強度不同」；H3 仍用 α_eq，不受影響",
                "pain": "H2 改用這個強度，結果標示「強度不同」",
                "fear": "不影響 H1–H4（H3 用 α_eq）", "sadness": "不影響 H1–H4",
                "joy": "不影響 H1–H4", "compassion": "不影響 H1–H4"}
AXIS_ZH = {"V": "正負面（V）", "A": "激動程度（A）", "D": "掌控感（D）"}
RESID_ZH = {"anger__resid3": "憤怒殘差（扣 V、A、D）", "anger__resid2": "憤怒殘差（只扣 V、A）",
            "pain__resid3": "疼痛殘差（扣 V、A、D）",
            "compassion__resid3": "同情殘差（扣 V、A、D）"}


def _zh(e: str) -> str:
    return f"{EMOTION_ZH[e]}（{e}）" if e in EMOTION_ZH else e


def _a(x) -> str:
    return "—" if x is None else f"{float(x):.4g}"


def _n(x) -> str:
    return "—" if x is None else f"{float(x):.2f}"


def _quote(text: str) -> list[str]:
    lines = [ln.rstrip() for ln in str(text).strip().splitlines()] or ["（空白）"]
    return [f"> {ln}" if ln else ">" for ln in lines]


def _pick(texts: list[dict], layer, alpha: float, label: str, k: int) -> list[str]:
    rows = [r for r in texts if r.get("label") == label
            and (r.get("layer") is None if layer is None
                 else r.get("layer") is not None and int(r["layer"]) == int(layer))
            and math.isclose(float(r.get("alpha", 0.0)), alpha, abs_tol=ALPHA_TOL)]
    rows.sort(key=lambda r: int(r.get("prompt_idx", 0)))
    return [r["text"] for r in rows[:k]]


def _sec_layer(fz: dict) -> list[str]:
    L, chosen = fz["layer"], next((r for r in fz["layer_table"] if r["layer"] == fz["layer"]), {})
    out = ["## 一、選哪一層、為什麼", "",
           "規則（事先寫定）：每個候選層先找出 6 種核心情緒都還保持通順的最大強度（α_max），"
           "再看在這個強度下 6 種情緒的評分平均比未調整時高出多少（平均增幅）；"
           "選平均增幅最大的一層。平手時先選被標示「汙染」的情緒較少的一層，再選層號較小的一層。", "",
           f"**結果：選第 {L} 層**（α_max = {_a(fz['alpha_max'])}，"
           f"平均增幅 {_n(chosen.get('mean_gain'))} 分）。", "",
           "| 層 | α_max | 平均增幅（分） | 被標示汙染的情緒數 | 選中 |", "|---|---|---|---|---|"]
    for r in fz["layer_table"]:
        am = "0（最小強度就不通順，不列入比較）" if not r["alpha_max"] else _a(r["alpha_max"])
        out.append(f"| {r['layer']} | {am} | {_n(r['mean_gain'])} | {r['n_flagged']} | "
                   f"{'✔' if r['layer'] == L else ''} |")
    return out + [""]


def _sec_coherence(fz: dict, metrics: pd.DataFrame, emotions: list[str]) -> list[str]:
    out = ["## 二、「通順」的判斷規則（2026-11-22 修訂）", "",
           "原本的規則用「困惑度」（文字對模型來說有多出乎意料）當門檻。先導測試發現它分不出"
           "「通順但帶情緒」和「真正壞掉」的文字：未調整時文字的困惑度本來就極低，只要內容稍有改變，"
           "比值就會超過門檻。因此在任何道德題資料產生之前，根據先導資料把規則改成：", "",
           f"- 格式題的有效答案機率總和 ≥ {cal.FORMAT_MIN}（模型還能按格式回答），而且",
           f"- 未調整的模型給文字打的「通順程度」（1–5 分）平均不低於未調整時減 {cal.FLUENCY_DROP} 分。",
           "- 困惑度比仍會記錄，但只作參考，不再當門檻。", "",
           "下表是每一層、每種情緒還保持通順的最大 α（通順上限）。0 表示最小的強度就已經不通順。", "",
           "| 層 | " + " | ".join(_zh(e) for e in emotions) + " |",
           "|---|" + "---|" * len(emotions)]
    for L in sorted({int(x) for x in metrics["layer"]}):
        out.append(f"| {L} | " + " | ".join(_a(coherent_max(metrics, L, e)) for e in emotions)
                   + " |")
    return out + [""]


def _sec_alpha(fz: dict, emotions: list[str]) -> list[str]:
    T = fz.get("alpha_eq_target")
    out = ["## 三、強度：α_max 與等強度 α_eq", "",
           f"α 的單位是「第 {fz['layer']} 層殘差流的平均長度」（{fz['resid_norm']:.1f}）："
           "加進去的量 = α × 平均長度 × 情緒方向。α 越大，調整越強。", "",
           f"- **α_max = {_a(fz['alpha_max'])}**：6 種核心情緒在這一層都還保持通順的最大強度。"
           "大部分實驗條件都用 +α_max、−α_max 和一半強度。",
           "- **α_eq（等強度 α）**：為了公平比較（例如憤怒 vs 恐懼），替每種情緒各找一個 α，"
           "讓評分都比未調整時高出同樣的分數，避免「只是某個向量比較強」的解釋。"]
    if T is None:
        out += ["", "所有核心情緒都未能誘發，因此無法計算等強度 α。"]
    else:
        out += [f"  目標增幅 = {T:.2f} 分（所有成功誘發的情緒在各自通順上限內都能達到的最高共同增幅，"
                f"最多 {EQ_TARGET_MAX:g} 分）。評分只是粗略的強度指標。"]
    out += ["", "| 情緒 | 通順上限 | α_eq |", "|---|---|---|"]
    for e in emotions:
        eq = fz["alpha_eq"].get(e)
        out.append(f"| {_zh(e)} | {_a(fz['coherent_max'][e])} | "
                   f"{'無（未能誘發）' if eq is None else _a(eq)} |")
    return out + [""]


def _sec_induction(fz: dict, emotions: list[str]) -> list[str]:
    out = ["## 四、每種情緒是否成功誘發", "",
           f"標準：在 α_max 時，該情緒的評分比未調整時至少高 {INDUCTION_GAIN:g} 分，"
           f"而且高於 {N_CONTROLS} 個對照向量（40 個非情緒概念 + 20 個隨機方向）"
           f"在同一題評分的第 {CONTROL_PERCENTILE} 百分位數。沒過標準的情緒只替它自己把 α 往上加，"
           "但不超過它自己的通順上限（疼痛也一樣，這是倫理限制）。", "",
           "| 情緒 | 結果 | 使用的 α | 增幅（分） | 評分 | 對照第 95 百分位數 |",
           "|---|---|---|---|---|---|"]
    for e in emotions:
        d = fz["induction_detail"][e]
        out.append(f"| {_zh(e)} | {STATUS_TEXT[d['status']]} | {_a(d['alpha'])} | "
                   f"{_n(d['gain'])} | {_n(d['rating'])} | {_n(d['p95'])} |")
    out += ["", "「成功」和「未能誘發」列的數值是在 α_max 時量到的；「強度不同」列是在它自己的備案 α 量到的。",
            ""]
    notes = []
    for e in emotions:
        d = fz["induction_detail"][e]
        if d["status"] == "fallback":
            notes.append(f"- **{_zh(e)}：強度不同。** 在 α_max 下沒過標準，加到 α = {_a(d['alpha'])}"
                         f" 才通過（仍在它自己的通順上限 {_a(fz['coherent_max'][e])} 之內），"
                         f"並在這個 α 另外跑了一組對照向量。影響：{HYP_FALLBACK.get(e, '不影響 H1–H4')}。")
        elif d["status"] == "failed":
            notes.append(f"- **{_zh(e)}：未能誘發。** 加到它自己的通順上限 "
                         f"{_a(fz['coherent_max'][e])} 仍沒過標準。影響："
                         f"{HYP_FAILED.get(e, '不影響 H1–H4')}。")
    out += notes or ["六種核心情緒都在 α_max 下成功誘發，H1–H4 都照原計畫檢驗。"]
    return out + [""]


def _sec_examples(fz: dict, texts: list[dict], emotions: list[str]) -> list[str]:
    L, a = fz["layer"], float(fz["alpha_max"])
    out = [f"## 五、範例文字（第 {L} 層，α_max = {_a(a)}）", "",
           "下面是同一組中性寫作提示下的輸出。「模型被調整成某種情緒狀態」只表示我們在模型內部"
           "加上了該情緒的方向，讓輸出文字帶有那種情緒的色彩；這不代表模型有任何感受。"
           "未調整的範例和每種情緒的「範例 1」用的是同一個寫作提示，可以直接比較。", "",
           "### 對照：未調整（α = 0）", ""]
    base = _pick(texts, None, 0.0, cal.BASELINE, 1)
    out += (_quote(base[0]) if base else ["（找不到未調整的文字）"]) + [""]
    for e in emotions:
        out += [f"### 模型被調整成「{EMOTION_ZH.get(e, e)}」的狀態（{e}）", ""]
        got = _pick(texts, L, a, e, 2)
        if not got:
            out += ["（找不到這組文字）", ""]
        for i, t in enumerate(got, 1):
            out += [f"範例 {i}：", "", *_quote(t), ""]
    return out


def _sec_contamination(fz: dict) -> list[str]:
    flagged = fz["contamination_at_layer"]
    out = [f"## 六、汙染的處理（第 {fz['layer']} 層）", "",
           "「汙染」是指某個情緒向量本身就和「懲罰」或「傷害他人」的概念方向重疊得比較多"
           "（比悲傷向量多，也比 40 個非情緒概念向量的第 95 百分位數多）。"
           "這樣即使結果顯著，也可能只是向量裡本來就有「懲罰」的成分。", ""]
    if flagged:
        out += [f"這一層被標示的情緒：{'、'.join(_zh(e) for e in flagged)}。",
                "處理方式：對它們另外跑「乾淨版」向量（把和懲罰／傷害方向重疊的成分扣掉）；"
                "涉及這些情緒的檢驗，必須乾淨版也通過才算支持假設。"]
    else:
        out.append("這一層沒有情緒被標示汙染，不需要乾淨版。")
    return out + [""]


def _sec_axes(fz: dict) -> list[str]:
    out = ["## 七、殘差向量扣掉了哪些軸", "",
           "殘差向量 = 情緒向量扣掉通用的正負面（V）、激動程度（A）、掌控感（D）三個軸之後剩下的部分，"
           "用來檢驗「情緒的效果是否超出這些通用面向」。只扣掉驗證可靠的軸；"
           "沒扣到的軸，結論只能寫成「超出已扣掉的這幾個軸」。", "",
           "| 殘差向量 | 狀態 | 扣掉的軸 |", "|---|---|---|"]
    for name, removed in fz["residual_removed_axes"].items():
        st = fz["residual_status_at_layer"].get(name)
        axes_txt = ("、".join(AXIS_ZH[k] for k in removed) if removed
                    else "無法建立（正負面軸不可靠）" if st != "ok" else "（無）")
        out.append(f"| {RESID_ZH.get(name, name)} | {'可用' if st == 'ok' else '無法檢驗'} | "
                   f"{axes_txt} |")
    return out + [""]


def _sec_workload(n_conditions: int, rate: float, passes: int) -> list[str]:
    total = n_conditions * passes
    hours = total / rate / 3600
    return ["## 八、接下來的工作量", "",
            f"依凍結設定共有 **{n_conditions} 個實驗條件**；每個條件要計分 {passes} 次"
            "（72 個懲罰情境 × 4 次：評分正反各一次、二選一 A/B 順序各一次；"
            "150 題 MoralChoice × 2 次：A/B 順序各一次）。", "",
            f"合計 {n_conditions} × {passes} = **{total:,} 次前向計算**；以每秒 {rate:g} 次估計，"
            f"約 **{hours:.1f} 小時**。正式開始前會先用 3 個條件、10 題做先導，重新量測速度。", ""]


def _sec_items() -> list[str]:
    return ["## 九、題目樣本", "",
            "題目樣本請看 [results/6.6_moral_items/item_samples.md]"
            "(../../6.6_moral_items/item_samples.md)。", "",
            "注意：MoralChoice 的「造成痛苦」規則也包含心理上的痛苦（例如讓人難過或受辱），"
            "不只身體上的疼痛。因此已預先登記一個探索性分析：只看涉及身體傷害"
            "（致死、致殘）的題目子集，檢查結果是否一致。", ""]


def _sec_confirm(fz: dict) -> list[str]:
    return ["## 十、請你確認的事項", "",
            f"1. 選第 {fz['layer']} 層、α_max = {_a(fz['alpha_max'])} 是否合理（第一、三節）。",
            "2. 第五節的範例文字是否看得出模型被調整成對應的情緒狀態，而且文字仍然通順。",
            "3. 第四節的誘發結果：若有「強度不同」或「未能誘發」，是否接受對應的處理方式。",
            "4. 第六、七節的汙染處理和扣掉的軸是否清楚。",
            "5. 第九節的題目樣本是否可以接受。",
            "6. 第八節的預估時間是否可以接受。", "",
            "確認之後才會開始正式的道德題實驗。凍結設定之後不再修改；若一定要改，"
            "必須用 --force 重新凍結，舊檔會改名保留，並記錄在新檔的 supersedes 欄位。", ""]


def render_checkpoint2(frozen: dict, metrics: pd.DataFrame, texts: list[dict],
                       n_conditions: int, rate: float = DEFAULT_RATE,
                       passes: int = PASSES_PER_CONDITION) -> str:
    """Plain Traditional-Chinese checkpoint-2 summary (pure)."""
    emotions = list(frozen["induction"])
    lines = ["# 檢查點 2 摘要：選層、強度與誘發結果", "",
             f"模型：{frozen['model']}。這份摘要說明正式道德題實驗之前的校準結果。"
             f"所有設定已寫入 frozen_config.json（凍結時間 {frozen['created_at']}，"
             f"程式版本 {str(frozen['git_commit'])[:12]}），之後不得再改。"
             "請閱讀後回答最後一節的確認事項。", ""]
    lines += _sec_layer(frozen)
    lines += _sec_coherence(frozen, metrics, emotions)
    lines += _sec_alpha(frozen, emotions)
    lines += _sec_induction(frozen, emotions)
    lines += _sec_examples(frozen, texts, emotions)
    lines += _sec_contamination(frozen)
    lines += _sec_axes(frozen)
    lines += _sec_workload(n_conditions, rate, passes)
    lines += _sec_items()
    lines += _sec_confirm(frozen)
    return "\n".join(lines).rstrip("\n") + "\n"
