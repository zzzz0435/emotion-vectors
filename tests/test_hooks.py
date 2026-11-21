import pytest
import torch

from emolab.hooks import capture_mean_activations, mean_token_norm, steer
from emolab.models import get_layers


def _layer1(model, ids, mask):
    return model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[2]


def _steered_layer1(model, ids, mask, **steer_kw):
    """Layer-1 output seen by a probe registered after steer's hook.

    transformers' own hidden_states capture hook runs before ours, so
    output_hidden_states does not show the steering; a later hook does.
    """
    got = {}
    with steer(model, 1, **steer_kw):
        h = get_layers(model)[1].register_forward_hook(
            lambda _m, _i, out: got.setdefault("o", (out[0] if isinstance(out, tuple) else out).clone()))
        try:
            with torch.no_grad():
                model(ids, attention_mask=mask)
        finally:
            h.remove()
    return got["o"]


def _e0():
    e = torch.zeros(32)
    e[0] = 1.0
    return e


def test_capture_matches_hidden_states(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 7)); mask = torch.ones_like(ids)
    got = capture_mean_activations(tiny_qwen3, ids, mask, [1])[1]
    hs = _layer1(tiny_qwen3, ids, mask)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got[0], hs[0, 1:].mean(0), atol=1e-5, rtol=1e-4)


def test_capture_respects_left_padding(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 7))
    solo = capture_mean_activations(tiny_qwen3, ids, torch.ones_like(ids), [1])[1]
    pad_ids = torch.cat([torch.zeros(1, 3, dtype=torch.long), ids], 1)
    pad_mask = torch.cat([torch.zeros(1, 3, dtype=torch.long), torch.ones_like(ids)], 1)
    padded = capture_mean_activations(tiny_qwen3, pad_ids, pad_mask, [1])[1]
    torch.testing.assert_close(padded, solo, atol=1e-4, rtol=1e-4)
    n_solo = mean_token_norm(tiny_qwen3, ids, torch.ones_like(ids), [1])[1]
    n_pad = mean_token_norm(tiny_qwen3, pad_ids, pad_mask, [1])[1]
    assert n_pad == pytest.approx(n_solo, rel=1e-4)


def test_mean_token_norm(tiny_qwen3):
    ids = torch.randint(0, 128, (2, 6)); mask = torch.ones_like(ids)
    hs = _layer1(tiny_qwen3, ids, mask)
    want = hs[:, 1:].norm(dim=-1).mean().item()
    assert mean_token_norm(tiny_qwen3, ids, mask, [1])[1] == pytest.approx(want, rel=1e-5)


def test_steer_adds_scaled_unit_vector(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 6)); mask = torch.ones_like(ids)
    base = _layer1(tiny_qwen3, ids, mask)
    d = torch.zeros(32); d[0] = 3.0
    out = _steered_layer1(tiny_qwen3, ids, mask, direction=d, alpha=0.5, resid_norm=2.0)
    torch.testing.assert_close(out[0, 0], base[0, 0])
    torch.testing.assert_close(out[0, 1] - base[0, 1], _e0(), atol=1e-6, rtol=0)


def test_steer_left_padded_skips_first_real_token(tiny_qwen3):
    ids = torch.randint(0, 128, (2, 6))
    mask = torch.ones_like(ids); mask[0, :2] = 0  # row 0 left-padded by 2
    base = _layer1(tiny_qwen3, ids, mask)
    d = torch.zeros(32); d[0] = 3.0
    out = _steered_layer1(tiny_qwen3, ids, mask, direction=d, alpha=0.5, resid_norm=2.0)
    # row 0: first real token is position 2, unchanged; later real tokens shifted
    torch.testing.assert_close(out[0, 2], base[0, 2])
    torch.testing.assert_close(out[0, 3] - base[0, 3], _e0(), atol=1e-6, rtol=0)
    # row 1 (no padding): position 0 untouched, position 1 shifted
    torch.testing.assert_close(out[1, 0], base[1, 0])
    torch.testing.assert_close(out[1, 1] - base[1, 1], _e0(), atol=1e-6, rtol=0)


def test_steer_no_mask_treats_position_zero_as_first(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 5))
    base = tiny_qwen3(ids, output_hidden_states=True).hidden_states[2]
    out = _steered_layer1(tiny_qwen3, ids, None, direction=torch.ones(32),
                          alpha=1.0, resid_norm=1.0)
    torch.testing.assert_close(out[0, 0], base[0, 0])
    assert not torch.allclose(out[0, 1], base[0, 1])


def test_steer_decode_step_always_adds(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 5))
    nxt = torch.randint(0, 128, (1, 1))
    with torch.no_grad():
        pre = tiny_qwen3(ids, use_cache=True)
        base = tiny_qwen3(nxt, past_key_values=pre.past_key_values,
                          output_hidden_states=True).hidden_states[2]
        pre2 = tiny_qwen3(ids, use_cache=True)
        got = {}
        with steer(tiny_qwen3, 1, _e0(), alpha=1.0, resid_norm=1.0):
            h = get_layers(tiny_qwen3)[1].register_forward_hook(
                lambda _m, _i, o: got.setdefault("o", (o[0] if isinstance(o, tuple) else o).clone()))
            try:
                tiny_qwen3(nxt, past_key_values=pre2.past_key_values)
            finally:
                h.remove()
        out = got["o"]
    torch.testing.assert_close(out[0, 0] - base[0, 0], _e0(), atol=1e-6, rtol=0)


def test_steer_zero_alpha_registers_nothing(tiny_qwen3):
    layer = get_layers(tiny_qwen3)[1]
    n_fwd, n_pre = len(layer._forward_hooks), len(tiny_qwen3._forward_pre_hooks)
    with steer(tiny_qwen3, 1, torch.ones(32), alpha=0.0, resid_norm=5.0):
        assert len(layer._forward_hooks) == n_fwd
        assert len(tiny_qwen3._forward_pre_hooks) == n_pre


def test_steer_hook_removed_after_exit(tiny_qwen3):
    ids = torch.randint(0, 128, (1, 6)); mask = torch.ones_like(ids)
    base = _layer1(tiny_qwen3, ids, mask)
    layer = get_layers(tiny_qwen3)[1]
    n_fwd, n_pre = len(layer._forward_hooks), len(tiny_qwen3._forward_pre_hooks)
    with steer(tiny_qwen3, 1, torch.ones(32), alpha=1.0, resid_norm=1.0):
        assert len(layer._forward_hooks) == n_fwd + 1
    assert len(layer._forward_hooks) == n_fwd
    assert len(tiny_qwen3._forward_pre_hooks) == n_pre
    torch.testing.assert_close(_layer1(tiny_qwen3, ids, mask), base, atol=0, rtol=0)


def test_steer_hook_removed_on_exception(tiny_qwen3):
    layer = get_layers(tiny_qwen3)[1]
    n_fwd, n_pre = len(layer._forward_hooks), len(tiny_qwen3._forward_pre_hooks)
    with pytest.raises(RuntimeError):
        with steer(tiny_qwen3, 1, torch.ones(32), alpha=1.0, resid_norm=1.0):
            raise RuntimeError("boom")
    assert len(layer._forward_hooks) == n_fwd
    assert len(tiny_qwen3._forward_pre_hooks) == n_pre


def test_nan_raises(tiny_qwen3):
    def poison(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h * float("nan")
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = get_layers(tiny_qwen3)[2].register_forward_hook(poison)
    try:
        ids = torch.randint(0, 128, (1, 5))
        with pytest.raises(FloatingPointError, match="layer 2"):
            capture_mean_activations(tiny_qwen3, ids, torch.ones_like(ids), [2])
    finally:
        handle.remove()
