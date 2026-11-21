import json
from collections import Counter

import pytest
import torch

from emolab.batching import run_with_oom_backoff
from emolab.paths import DATASETS
from emolab.stories import (GENERAL_RESTRICTION, build_jobs, build_topup_jobs,
                            round_robin_limit, run_generation)


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


def test_resume_repairs_line_torn_inside_multibyte_char(tmp_path):
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)[:6]
    out = tmp_path / "s.jsonl"
    run_generation(jobs[:3], out, lambda ps: ["it’s fine"] * len(ps), batch_size=3,
                   model_name="m")
    with out.open("ab") as f:  # first byte of a 3-byte char, cut mid-character
        f.write('{"id": "x", "text": "’'.encode("utf-8")[:-2])
    run_generation(jobs, out, _fake, batch_size=3, model_name="m")
    assert _ids(out) == [j.id for j in jobs]


def test_round_robin_limit_covers_labels_deterministically():
    jobs = build_jobs(EMOTIONS, TOPICS, CONCEPTS)
    labels = list(dict.fromkeys(j.label for j in jobs))
    sel = round_robin_limit(jobs, len(labels) + 3)
    assert [j.label for j in sel[:len(labels)]] == labels
    assert [j.label for j in sel[len(labels):]] == labels[:3]
    assert sel[0].replicate == 0 and sel[len(labels)].id != sel[0].id
    assert sel == round_robin_limit(jobs, len(labels) + 3)
    assert len(round_robin_limit(jobs, 10**6)) == len(jobs)


# ---- I1: top-up rounds draw new samples and oversample short cells ----

def test_topup_job_count_formula():
    from emolab.stories import topup_job_count
    assert topup_job_count(3, 1.0) == 3
    assert topup_job_count(3, 0.5) == 6
    assert topup_job_count(3, 0.01) == 15      # acceptance floored at 0.1 -> 30, capped at 5x = 15
    assert topup_job_count(10, 0.1) == 50      # 100 capped at 5x missing
    assert topup_job_count(2, 0.3) == 7        # ceil(2/0.3)


def test_topup_oversamples_with_acceptance_and_keeps_numbering():
    jobs = build_topup_jobs({("pain", 3, 2): 2, ("anger", 0, 0): 1}, EMOTIONS, TOPICS, CONCEPTS,
                            replicate_offset=5, acceptance={"pain": 0.5, "anger": 0.25})
    pain = [j for j in jobs if j.label == "pain"]
    anger = [j for j in jobs if j.label == "anger"]
    assert len(pain) == 4 and len(anger) == 4
    assert [j.replicate for j in pain] == [5, 6, 7, 8]
    assert all(j.subtype == PAIN_SUBTYPES[2] for j in pain)
    # a label absent from the acceptance map is treated as 1.0
    j2 = build_topup_jobs({("pain", 3, 2): 2}, EMOTIONS, TOPICS, CONCEPTS, 0, acceptance={})
    assert len(j2) == 2


def test_acceptance_rates_from_report_rows():
    from emolab.stories import acceptance_rates
    rows = [{"label": "a", "n_raw": "100", "n_kept": "40"},
            {"label": "b", "n_raw": "0", "n_kept": "0"}]
    assert acceptance_rates(rows) == {"a": 0.4}


def test_generate_seed_base_differs_between_rounds(monkeypatch):
    from emolab import models
    from emolab.stories import make_generate_fn, topup_seed_base

    class Enc(dict):
        def to(self, _d):
            return self

    class Tok:
        pad_token_id = 0

        def __call__(self, texts, **_k):
            return Enc(input_ids=torch.zeros(len(texts), 2, dtype=torch.long))

        def batch_decode(self, new, **_k):
            return ["x"] * new.shape[0]

    seeds = []

    class Model:
        device = "cpu"

        def generate(self, input_ids, **_k):
            seeds.append(torch.initial_seed())
            return torch.cat([input_ids, input_ids], dim=1)

    monkeypatch.setattr(models, "format_chat", lambda tok, p: p)
    gen0 = make_generate_fn(Model(), Tok())
    gen0(["p"]); gen0(["p"])
    assert seeds == [42, 43]
    seeds.clear()
    genk = make_generate_fn(Model(), Tok(), seed_base=topup_seed_base(5))
    genk(["p"])
    assert seeds == [42 + 5000]
    assert topup_seed_base(5) != topup_seed_base(9) and topup_seed_base(0) == 42


# ---- I2: reader is pure; only run_generation repairs a torn final line ----

def _rows_file(tmp_path, lines):
    p = tmp_path / "r.jsonl"
    p.write_bytes(b"".join(lines))
    return p


def test_read_rows_mid_file_bad_line_raises_and_leaves_file(tmp_path):
    from emolab.stories import read_rows
    good = lambda i: (json.dumps({"id": f"r{i}"}) + "\n").encode()
    p = _rows_file(tmp_path, [good(0), good(1), b'{"id": "to\n', good(3), good(4)])
    before = p.read_bytes()
    with pytest.raises(ValueError, match="line 3"):
        read_rows(p)
    assert p.read_bytes() == before
    with pytest.raises(ValueError):
        run_generation(build_jobs(EMOTIONS, TOPICS, CONCEPTS)[:2], p, _fake, 2, "m")
    assert p.read_bytes() == before


def test_read_rows_bad_last_line_skipped_without_writing(tmp_path):
    from emolab.stories import read_rows
    p = _rows_file(tmp_path, [b'{"id": "r0"}\n', b'{"id": "torn'])
    before = p.read_bytes()
    with pytest.warns(UserWarning, match="torn"):
        rows = read_rows(p)
    assert rows == [{"id": "r0"}] and p.read_bytes() == before
    assert read_rows(tmp_path / "missing.jsonl") == []
