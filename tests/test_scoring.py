import numpy as np
import pytest
import torch

from emolab.scoring import (SteerSpec, answer_token_ids, continuation_ppl,
                            expected_rating, harsh_high, judge_ratings,
                            option_probs, score_prompts)


def test_option_probs_normalises_and_reports_invalid():
    logits = torch.full((10,), -1e9); logits[[2, 3, 7]] = torch.log(torch.tensor([0.3, 0.1, 0.6]))
    p, invalid = option_probs(logits, [[2, 3], [7]])
    np.testing.assert_allclose(p, [0.4, 0.6]); assert invalid == pytest.approx(0.0, abs=1e-6)


def test_option_probs_all_invalid_gives_nan():
    logits = torch.zeros(10); logits[0] = 50.0
    p, invalid = option_probs(logits, [[2], [3]])
    assert np.isnan(p).all() and invalid == pytest.approx(1.0)


def test_expected_rating_and_reverse_scale_converted():
    assert expected_rating(np.array([0, 0, 0, 0, 0, 0, 1.0]), list(range(1, 8))) == 7
    assert harsh_high(2.0, "harsh1") == 6.0 and harsh_high(2.0, "light1") == 2.0
    assert np.isnan(expected_rating(np.full(3, np.nan), [1, 2, 3]))


def test_score_prompts_steering_changes_scores(tiny_qwen3, fake_tok):
    msgs = ["hello there", "a much longer message than the first", "x"]
    opts = ["A", "B", "C"]
    p1, inv1 = score_prompts(tiny_qwen3, fake_tok, msgs, opts, batch_size=4)
    p2, inv2 = score_prompts(tiny_qwen3, fake_tok, msgs, opts, batch_size=4)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(inv1, inv2)
    assert p1.shape == (3, 3) and inv1.shape == (3,)

    torch.manual_seed(0)
    spec = SteerSpec(layer=1, direction=torch.randn(32), alpha=5.0, resid_norm=10.0)
    ps, invs = score_prompts(tiny_qwen3, fake_tok, msgs, opts, batch_size=4, steer_spec=spec)
    assert not np.allclose(ps, p1, atol=1e-6) or not np.allclose(invs, inv1, atol=1e-6)

    pb, ib = score_prompts(tiny_qwen3, fake_tok, msgs, opts, batch_size=1)
    np.testing.assert_allclose(pb, p1, atol=1e-5)
    np.testing.assert_allclose(ib, inv1, atol=1e-5)
    pb, ib = score_prompts(tiny_qwen3, fake_tok, msgs, opts, batch_size=1, steer_spec=spec)
    np.testing.assert_allclose(pb, ps, atol=1e-5)
    np.testing.assert_allclose(ib, invs, atol=1e-5)


def test_answer_token_ids_dedupes_and_raises(fake_tok):
    assert len(answer_token_ids(fake_tok, "A")) == 1  # " A" is two chars -> dropped
    with pytest.raises(ValueError):
        answer_token_ids(fake_tok, "AB")


def test_continuation_ppl_ignores_padding_and_prompt(tiny_qwen3, fake_tok):
    prompts = ["hi", "a longer prompt here"]
    conts = ["there friend", "ok"]
    a = continuation_ppl(tiny_qwen3, fake_tok, prompts, conts, batch_size=2)
    b = continuation_ppl(tiny_qwen3, fake_tok, prompts, conts, batch_size=1)
    assert a.shape == (2,) and np.all(a > 0) and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_judge_ratings_range(tiny_qwen3, fake_tok):
    r = judge_ratings(tiny_qwen3, fake_tok, ["I am so happy", "meh"], "joy", batch_size=2)
    assert r.shape == (2,)
    assert ((r >= 1) & (r <= 5)).all() or np.isnan(r).all()


@pytest.mark.network
def test_answer_token_ids_cover_variants():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    assert len(answer_token_ids(tok, "A")) >= 1
    for d in "1234567":
        assert answer_token_ids(tok, d)
