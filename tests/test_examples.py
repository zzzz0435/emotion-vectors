import importlib.util
import json
from pathlib import Path

import pytest

from emolab import runner

REPO = Path(__file__).resolve().parents[1]
VIG = json.loads((REPO / "datasets/6.6_punishment_vignettes.json").read_text(encoding="utf-8"))
MC = json.loads((REPO / "datasets/6.6_moralchoice_harm.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location(
        "examples02", REPO / "scripts/6.7_runs/02_generate_examples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("task,items", [("h1_rating", VIG), ("h1_choice", VIG), ("h2", MC)])
def test_example_prompts_have_no_bare_answer_demand(ex, task, items):
    prompts = ex.example_prompts(task, items)
    assert len(prompts) == ex.N_ITEMS
    for _, _, msg, _ in prompts:
        for bad in ("digit only", "letter only", "without any other words", "solely with A or B",
                    "vocabulary is limited", "Answer:", "Answer with a single digit", "A or B only"):
            assert bad not in msg, (task, bad)
        full = msg + ex.SUFFIX
        assert full.endswith("two or three sentences.")


def test_example_text_changes_only_the_demands(ex):
    raw = runner.task_prompts("h2", MC[:1])[0][2]
    out = ex.example_text("h2", raw)
    assert raw.endswith("\nAnswer:") and not out.endswith("Answer:")
    assert "Respond solely with A or B." in raw and "Respond solely" not in out
    assert MC[0]["context"] in out and MC[0]["action1"] in out
    with pytest.raises(ValueError):
        ex.example_text("h2", "some unrelated prompt")


def test_pick_items_spread_and_deterministic(ex):
    h1 = ex.pick_items("h1_rating", VIG)
    assert len(h1) == 20 and len({v["id"] for v in h1}) == 20
    assert len({v["offense"] for v in h1}) == 6 and len({v["severity"] for v in h1}) == 3
    assert h1 == ex.pick_items("h1_rating", VIG)
    h2 = ex.pick_items("h2", MC)
    ids = [m["id"] for m in h2]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert ids[0] == MC[0]["id"] and ids[-1] == MC[(19 * 150) // 20]["id"]
    assert ex.pick_items("h2", MC[:5]) == MC[:5]


def test_render_states_the_selection_rule(ex):
    md = ex.render({}, [], {})
    assert "題目挑選規則" in md and "等距" in md


def test_cache_drops_torn_tail_and_resumes(ex, tmp_path):
    p = tmp_path / "examples.jsonl"
    good = [{"key": "h2|m0|baseline", "response": "好"}, {"key": "h2|m1|baseline", "response": "中"}]
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good).encode("utf-8")
    p.write_bytes(raw + '{"key": "h2|m2|base'.encode("utf-8"))      # hard kill mid-write
    with pytest.warns(UserWarning):
        cache = ex.load_cache(p)
    assert cache == {r["key"]: r["response"] for r in good}
    assert p.read_bytes() == raw                                     # tail removed on disk
    with p.open("a", encoding="utf-8", newline="\n") as f:           # a resumed append is clean
        f.write(json.dumps({"key": "h2|m2|baseline", "response": "新"}) + "\n")
    assert len(ex.load_cache(p)) == 3
    assert ex.load_cache(tmp_path / "missing.jsonl") == {}
