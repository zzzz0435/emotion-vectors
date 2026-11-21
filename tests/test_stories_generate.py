import json
from collections import Counter

import pytest
import torch

from emolab.batching import run_with_oom_backoff
from emolab.paths import DATASETS
from emolab.stories import (GENERAL_RESTRICTION, build_jobs, build_topup_jobs,
                            run_generation)


def _load(name):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


EMOTIONS, TOPICS, CONCEPTS = (_load("6.2_emotions.json"), _load("6.2_story_topics.json"),
                              _load("6.2_concepts.json"))
PAIN_SUBTYPES = next(e for e in EMOTIONS if e["label"] == "pain")["subtypes"]


def _fake(prompts):
    return [f"story: {p[:20]}" for p in prompts]


def _ids(path):
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_jobs_counts():
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)
    c = Counter(j.label for j in jobs)
    assert c["anger"] == 100 and c["pain"] == 100 and c["neutral"] == 100
    assert c["jealousy"] == 50 and c["weather"] == 40 and c["punishment"] == 40
    assert set(Counter(j.subtype for j in jobs if j.label == "pain").values()) == {25}
    assert len(jobs) == 3080


def test_topup_keeps_cell_subtype():
    jobs = build_topup_jobs({("pain", 3, 2): 2}, EMOTIONS, TOPICS, CONCEPTS, replicate_offset=5)
    assert len(jobs) == 2 and all(j.subtype == PAIN_SUBTYPES[2] and j.topic_idx == 3 for j in jobs)
    assert [j.replicate for j in jobs] == [5, 6]
    assert all(j.subtype_idx == 2 for j in jobs)


def test_ids_unique_and_deterministic():
    a, b = build_jobs(EMOTIONS, TOPICS, CONCEPTS), build_jobs(EMOTIONS, TOPICS, CONCEPTS)
    assert [j.id for j in a] == [j.id for j in b] and len({j.id for j in a}) == len(a)
    assert a[0].id == "anger__t00__s0__r0"


def test_contamination_prompt_has_no_restriction():
    j = next(j for j in build_jobs(EMOTIONS, TOPICS, CONCEPTS) if j.label == "punishment")
    assert "must not involve" not in j.prompt
    assert j.kind == "contamination"


def test_own_restriction_replaces_general():
    j = next(j for j in build_jobs(EMOTIONS, TOPICS, CONCEPTS) if j.label == "shame")
    own = next(e for e in EMOTIONS if e["label"] == "shame")["restriction"]
    assert own in j.prompt and GENERAL_RESTRICTION not in j.prompt


def test_resume_skips_existing(tmp_path):
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)[:23]
    out = tmp_path / "s.jsonl"
    calls = {"n": 0}

    def flaky(prompts):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return _fake(prompts)

    with pytest.raises(KeyboardInterrupt):
        run_generation(jobs, out, flaky, batch_size=4, model_name="m")
    assert len(_ids(out)) == 16
    n = run_generation(jobs, out, _fake, batch_size=4, model_name="m")
    ids = _ids(out)
    assert n == 7 and len(ids) == len(set(ids)) == len(jobs)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {"id", "label", "kind", "topic_idx", "topic", "subtype", "subtype_idx",
                        "replicate", "text", "model"}


def test_resume_repairs_torn_last_line(tmp_path):
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)[:6]
    out = tmp_path / "s.jsonl"
    run_generation(jobs[:3], out, _fake, batch_size=3, model_name="m")
    with out.open("a", encoding="utf-8") as f:
        f.write('{"id": "torn')
    run_generation(jobs, out, _fake, batch_size=3, model_name="m")
    assert _ids(out) == [j.id for j in jobs]


def test_oom_halves_batch(tmp_path):
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)[:20]
    out = tmp_path / "s.jsonl"
    sizes = []

    def oomy(prompts):
        sizes.append(len(prompts))
        if len(prompts) > 2:
            raise torch.cuda.OutOfMemoryError("fake oom")
        return _fake(prompts)

    run_generation(jobs, out, oomy, batch_size=8, model_name="m")
    assert _ids(out) == [j.id for j in jobs]
    assert sizes[:3] == [8, 4, 2]


def test_oom_at_batch_one_propagates():
    def always(items):
        raise torch.cuda.OutOfMemoryError("fake oom")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(run_with_oom_backoff(always, [1, 2, 3], 4))
