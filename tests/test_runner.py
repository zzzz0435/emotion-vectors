import json

import numpy as np
import pytest

from emolab import runner, stats
from emolab.conditions import Condition
from emolab.integrity import file_sha256

VIG = [{"id": f"v{i}", "text": f"T{i}", "lenient": f"len{i}", "harsh": f"har{i}"} for i in range(3)]
MC = [{"id": f"m{i}", "context": f"C{i}", "action1": f"x{i}", "action2": f"y{i}",
       "harmful": 1 + i % 2} for i in range(3)]
TASKS = {"h1_rating": VIG, "h1_choice": VIG, "h2": MC}


def conds(n=5):
    return [Condition("baseline", "baseline", None, 0.0)] + [
        Condition(f"c{i}", "emotion", "x", 0.5) for i in range(n - 1)]


def fake_score(cond, msgs, options):
    k = len(options)
    p = np.full((len(msgs), k), 1.0 / k)
    return p, np.full(len(msgs), 0.01)


def test_runner_resume_skips_done_and_never_leaves_partial(tmp_path):
    calls = {"n": 0}

    def flaky(cond, msgs, options):
        if cond.name == "c2":
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
        return fake_score(cond, msgs, options)

    cs = conds()
    with pytest.raises(RuntimeError):
        runner.run_all(cs, TASKS, tmp_path, flaky, log=lambda s: None)
    (tmp_path / "h2__c3.csv.tmp").write_text("partial")   # stale leftover
    seen = []
    stats_ = runner.run_all(cs, TASKS, tmp_path,
                            lambda c, m, o: (seen.append(c.name), fake_score(c, m, o))[1],
                            log=lambda s: None)
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.glob("*.csv"))) == len(cs) * 3
    assert "baseline" not in seen and "c0" not in seen and "c1" not in seen
    assert stats_["skipped"] >= 3 * 3


@pytest.mark.parametrize("task,items,fn", [
    ("h1_choice", VIG, None), ("h2", MC, None)])
def test_order_swap_maps_back(task, items, fn):
    for harmful_force in (1, 2):
        its = [dict(it, harmful=harmful_force) for it in items] if task == "h2" else items
        meta = runner.task_prompts(task, its)
        # probs: A=0.8, B=0.2 for every prompt
        probs = np.tile([0.8, 0.2], (len(meta), 1))
        rows = runner.rows_from_scores(task, meta, probs, np.zeros(len(meta)))
        for (iid, variant, _msg, letter), r in zip(meta, rows):
            assert r["variant"] == variant
            assert r["p_target"] == pytest.approx(0.8 if letter == "A" else 0.2)
        # the target letter really swaps between orders
        by_item = {}
        for m in meta:
            by_item.setdefault(m[0], {})[m[1]] = m[3]
        for d in by_item.values():
            assert d["0"] != d["1"]
        if task == "h2":
            # harmful=1 -> action1 is harmful: order 0 -> A, order 1 -> B
            exp0 = "A" if harmful_force == 1 else "B"
            assert by_item["m0"]["0"] == exp0
        else:
            assert by_item["v0"] == {"0": "B", "1": "A"}


def _setup_freeze(tmp_path, **override):
    cond = tmp_path / "conditions.py"
    cond.write_text("x=1")
    vig = tmp_path / "vig.json"
    vig.write_text("[1]")
    mc = tmp_path / "mc.json"
    mc.write_text("[2]")
    rnd = tmp_path / "rnd.npy"
    rnd.write_bytes(b"x")
    frozen = {"conditions_sha256": file_sha256(cond), "vignettes_sha256": file_sha256(vig),
              "moralchoice_sha256": file_sha256(mc), "layer": 23}
    frozen.update(override)
    fp = tmp_path / "frozen_config.json"
    fp.write_text(json.dumps(frozen))
    files = {"conditions_sha256": cond, "vignettes_sha256": vig, "moralchoice_sha256": mc}
    return fp, files, rnd, cond, vig, mc


def test_refuses_without_frozen_or_changed_conditions(tmp_path):
    fp, files, rnd, cond, vig, mc = _setup_freeze(tmp_path)
    assert runner.preflight(fp, files, lambda f: rnd)["layer"] == 23
    with pytest.raises(SystemExit):
        runner.preflight(tmp_path / "missing.json", files, lambda f: rnd)
    with pytest.raises(SystemExit):
        runner.preflight(fp, files, lambda f: tmp_path / "nope.npy")
    cond.write_text("x=2")
    with pytest.raises(SystemExit):
        runner.preflight(fp, files, lambda f: rnd)
    cond.write_text("x=1")
    vig.write_text("[1, 1]")
    with pytest.raises(SystemExit):
        runner.preflight(fp, files, lambda f: rnd)
    vig.write_text("[1]")
    mc.write_text("[3]")
    with pytest.raises(SystemExit):
        runner.preflight(fp, files, lambda f: rnd)


def test_text_condition_prefixes_induce(tmp_path):
    seen = {}

    def spy(cond, msgs, options):
        seen[cond.name] = msgs
        return fake_score(cond, msgs, options)

    text = Condition("text_anger", "text", None, 0.0, "You feel furious.")
    base = Condition("baseline", "baseline", None, 0.0)
    runner.run_all([base, text], {"h1_choice": VIG}, tmp_path, spy, log=lambda s: None)
    for b, t in zip(seen["baseline"], seen["text_anger"]):
        assert t == "You feel furious. " + b


def test_h1_rating_reverse_scale():
    meta = runner.task_prompts("h1_rating", VIG[:1])
    assert [m[1] for m in meta] == ["light1", "harsh1"]
    p = np.zeros((2, 7))
    p[:, 6] = 1.0   # model says 7 for both prompts
    rows = runner.rows_from_scores("h1_rating", meta, p, np.zeros(2))
    assert rows[0]["p_target"] == pytest.approx(7.0)   # light1: 7 = harsh
    assert rows[1]["p_target"] == pytest.approx(1.0)   # harsh1: 7 = light -> 1
    nan = runner.rows_from_scores("h1_rating", meta, np.full((2, 7), np.nan), np.ones(2))
    assert np.isnan(nan[0]["p_target"]) and nan[0]["invalid"] == 1.0


def test_score_file_round_trips_through_stats_load_score_table(tmp_path):
    def score(cond, msgs, options):
        k = len(options)
        rng = np.random.default_rng(len(cond.name) + len(msgs))
        p = rng.dirichlet(np.ones(k), size=len(msgs))
        return p, rng.random(len(msgs)) * 0.1

    cs = conds(3)
    runner.run_all(cs, TASKS, tmp_path, score, log=lambda s: None)
    table = stats.load_score_table(tmp_path)
    for c in cs:
        for task, items in TASKS.items():
            meta = runner.task_prompts(task, items)
            probs, inv = score(c, [m[2] for m in meta], runner.OPTIONS[task])
            rows = runner.rows_from_scores(task, meta, probs, inv)
            for it in items:
                vals = [r["p_target"] for r in rows if r["item_id"] == it["id"]]
                assert table[task].loc[it["id"], c.name] == pytest.approx(np.mean(vals))
                invs = [r["invalid"] for r in rows if r["item_id"] == it["id"]]
                assert table[f"invalid_{task}"].loc[it["id"], c.name] == pytest.approx(np.mean(invs))


def test_pilot_requires_separate_out_dir(tmp_path):
    real = tmp_path / "scores"
    assert runner.check_pilot_out_dir(False, None, real) == real
    with pytest.raises(SystemExit):
        runner.check_pilot_out_dir(True, None, real)
    with pytest.raises(SystemExit):
        runner.check_pilot_out_dir(True, real, real)
    assert runner.check_pilot_out_dir(True, tmp_path / "pilot", real) == tmp_path / "pilot"
