import json

import pandas as pd
import pytest
import torch

from emolab import calibration as cal
from emolab.scoring import SteerSpec

EMOS = ["anger", "fear"]
LAYERS = [9, 14]
GRID = cal.ALPHA_GRID


def test_coherent_boundaries():
    assert cal.coherent(1.5, 1.0)
    assert not cal.coherent(1.51, 1.0)
    assert cal.coherent(1.0, 0.9)
    assert not cal.coherent(1.0, 0.89)


def test_constants():
    assert cal.ALPHA_GRID == [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    assert (cal.PPL_MAX, cal.FORMAT_MIN, cal.MAX_NEW_TOKENS) == (1.5, 0.9, 80)


def test_sweep_plan_order():
    plan = list(cal.sweep_plan(EMOS, LAYERS, [0.4, 0.2]))
    assert plan == [(9, 0.2, "anger"), (9, 0.2, "fear"), (14, 0.2, "anger"), (14, 0.2, "fear"),
                    (9, 0.4, "anger"), (9, 0.4, "fear"), (14, 0.4, "anger"), (14, 0.4, "fear")]


def _fake(bad_pair=(14, "anger"), bad_alpha=0.8):
    gen = []

    def generate_fn(layer, alpha, label):
        gen.append((layer, alpha, label))
        return [f"{layer}-{alpha}-{label}-{i}" for i in range(20)]

    def measure_fn(layer, alpha, label, texts):
        if (layer, label) == bad_pair and alpha >= bad_alpha:
            return 9.0, 1.0
        return 1.0, 1.0
    return gen, generate_fn, measure_fn


def test_early_stop_alpha_08():
    gen, g, m = _fake()
    rec = []
    cal.run_sweep(cal.sweep_plan(EMOS, LAYERS, GRID), set(), g, m,
                  lambda *a: rec.append(a[:3] + (a[6],)))
    pair = [a for (L, a, e) in gen if (L, e) == (14, "anger")]
    assert pair == [0.2, 0.4, 0.6, 0.8]
    assert (14, 0.8, "anger", False) in rec
    assert not any(r[:3] == (14, 1.0, "anger") for r in rec)
    for other in [(9, "anger"), (9, "fear"), (14, "fear")]:
        assert [a for (L, a, e) in gen if (L, e) == other] == GRID


def test_early_stop_uses_history_and_done():
    gen, g, m = _fake(bad_pair=(None, None))
    done = {(9, 0.2, "anger")}
    cal.run_sweep(cal.sweep_plan(["anger"], [9], GRID), done, g, m, lambda *a: None,
                  incoherent={(9, 0.4, "anger")})
    assert gen == [(9, 0.4, "anger")]  # 0.2 done, 0.6+ stopped by history


def _read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def test_sweep_to_disk_resume(tmp_path):
    gen, g, m = _fake(bad_pair=(None, None))
    plan = list(cal.sweep_plan(["anger"], [9], [0.2, 0.4]))
    # first run: only 0.2 done and measured; 0.4 has texts but no coherence row; torn tail
    (tmp_path / "texts.jsonl").write_text(
        "".join(json.dumps(dict(layer=9, alpha=a, label="anger", prompt_idx=i, text="t"))
                + "\n" for a in (0.2, 0.4) for i in range(20)) + '{"layer": 9, "al',
        encoding="utf-8")
    (tmp_path / "coherence.csv").write_text(
        "layer,alpha,label,ppl_ratio,format_valid,coherent\n9,0.2,anger,1.0,1.0,True\n")
    measured = []
    cal.sweep_to_disk(tmp_path, plan, g, lambda L, a, e, t: (measured.append(a), (1.0, 1.0))[1],
                      log=lambda s: None)
    assert gen == []  # nothing regenerated
    assert measured == [0.4]  # re-measured only the group with no coherence row
    coh = pd.read_csv(tmp_path / "coherence.csv")
    assert sorted(coh["alpha"]) == [0.2, 0.4]
    assert len(_read(tmp_path / "texts.jsonl")) == 40  # no duplicates, torn tail cut


def test_sweep_to_disk_partial_group_regenerated(tmp_path):
    gen, g, m = _fake(bad_pair=(None, None))
    (tmp_path / "texts.jsonl").write_text(
        "".join(json.dumps(dict(layer=9, alpha=0.2, label="anger", prompt_idx=i, text="t"))
                + "\n" for i in range(7)), encoding="utf-8")
    cal.sweep_to_disk(tmp_path, [(9, 0.2, "anger")], g, m, log=lambda s: None)
    assert gen == [(9, 0.2, "anger")]
    assert len(_read(tmp_path / "texts.jsonl")) == 20


def test_sweep_to_disk_incoherent_history_stops(tmp_path):
    gen, g, m = _fake(bad_pair=(9, "anger"), bad_alpha=0.4)
    plan = list(cal.sweep_plan(["anger"], [9], GRID))
    cal.sweep_to_disk(tmp_path, plan, g, m, log=lambda s: None)
    n = len(gen)
    assert [a for _, a, _ in gen] == [0.2, 0.4]
    cal.sweep_to_disk(tmp_path, list(cal.sweep_plan(["anger"], [9], GRID)), g, m,
                      log=lambda s: None)
    assert len(gen) == n  # rerun generates nothing more


def test_build_metrics_gain_and_alpha0_rows():
    coh = pd.DataFrame([
        dict(layer=None, alpha=0.0, label="baseline", ppl_ratio=1.0, format_valid=0.97, coherent=True),
        dict(layer="9", alpha=0.2, label="anger", ppl_ratio=1.1, format_valid=0.95, coherent=True),
        dict(layer="14", alpha=0.2, label="fear", ppl_ratio=2.0, format_valid=0.95, coherent=False),
    ])
    ratings = {(None, 0.0, "anger"): 1.5, (None, 0.0, "fear"): 2.0,
               (9, 0.2, "anger"): 3.0, (14, 0.2, "fear"): 2.5}
    df = cal.build_metrics(ratings, coh, [9, 14], ["anger", "fear"])
    assert list(df.columns) == cal.METRICS_COLS
    zero = df[df["alpha"] == 0]
    assert len(zero) == 4 and set(zip(zero["layer"], zero["label"])) == {
        (L, e) for L in (9, 14) for e in ("anger", "fear")}
    assert (zero["gain"] == 0).all() and (zero["ppl_ratio"] == 1.0).all()
    assert (zero["format_valid"] == 0.97).all() and zero["coherent"].all()
    a = df[(df["layer"] == 9) & (df["alpha"] == 0.2) & (df["label"] == "anger")].iloc[0]
    assert a["gain"] == pytest.approx(1.5) and a["ppl_ratio"] == 1.1
    f = df[(df["layer"] == 14) & (df["alpha"] == 0.2) & (df["label"] == "fear")].iloc[0]
    assert f["gain"] == pytest.approx(0.5) and not f["coherent"]
    assert len(df) == 6  # absent groups are absent


def test_control_names_and_aggregation_and_append():
    names = cal.control_names(["c1", "c2"], 3)
    assert names == ["c1", "c2", "random00", "random01", "random02"]
    rows = cal.control_rating_rows(1.0, "c1", {"anger": [1, 3], "fear": [2, float("nan")]})
    assert {r["emotion"]: r["rating"] for r in rows} == {"anger": 2.0, "fear": 2.0}
    first = cal.merge_control_ratings(None, rows)
    second = cal.merge_control_ratings(first, cal.control_rating_rows(1.6, "c1", {"anger": [5]}))
    assert list(second.columns) == cal.CONTROL_COLS
    assert sorted(set(second["alpha"])) == [1.0, 1.6]
    assert len(second) == 3
    redo = cal.merge_control_ratings(second, cal.control_rating_rows(1.6, "c1", {"anger": [4]}))
    assert len(redo) == 3 and redo[redo["alpha"] == 1.6]["rating"].iloc[0] == 4.0


@pytest.mark.parametrize("steered", [False, True])
def test_generate_batch_equals_single(tiny_qwen3, fake_tok, steered):
    fake_tok.padding_side = "left"
    fake_tok.decode = lambda ids, skip_special_tokens=True: " ".join(str(int(i)) for i in ids)
    prompts = ["hi", "a longer prompt here", "mid size"]
    spec = SteerSpec(1, torch.randn(32), 1.0, 5.0) if steered else None
    batch = cal.generate_batch(tiny_qwen3, fake_tok, prompts, spec, max_new_tokens=6)
    single = [cal.generate_batch(tiny_qwen3, fake_tok, [p], spec, max_new_tokens=6)[0]
              for p in prompts]
    assert batch == single
    assert all(isinstance(t, str) for t in batch)
