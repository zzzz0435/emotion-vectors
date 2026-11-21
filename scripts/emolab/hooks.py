"""Activation capture and steering hooks.

Layer L means the output of decoder block L (= output_hidden_states[L + 1]).
Both averaging and steering exclude the first non-padding position of each row.
"""
from contextlib import contextmanager

import torch

from emolab.models import get_layers


def _first_real(mask: torch.Tensor) -> torch.Tensor:
    """Index of the first position with mask==1 in each row, shape [B]."""
    return (mask != 0).long().argmax(dim=1)


def _used_positions(mask: torch.Tensor) -> torch.Tensor:
    """Bool [B,T]: real tokens minus the first real token of each row."""
    used = mask != 0
    used[torch.arange(mask.shape[0]), _first_real(mask)] = False
    if (used.sum(dim=1) == 0).any():
        raise ValueError("every row needs at least 2 non-padding tokens")
    return used


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def _capture(model, input_ids, attention_mask, layers):
    """Hidden states per layer, float32 on CPU, shape [B,T,d]; NaN/Inf checked."""
    got: dict[int, torch.Tensor] = {}
    handles = []
    dev = next(model.parameters()).device
    try:
        mod_layers = get_layers(model)
        for L in layers:
            def hook(_m, _i, out, L=L):
                got[L] = _out_tensor(out).detach().float().cpu()
            handles.append(mod_layers[L].register_forward_hook(hook))
        model(input_ids=input_ids.to(dev), attention_mask=attention_mask.to(dev),
              logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    for L in layers:
        if not torch.isfinite(got[L]).all():
            raise FloatingPointError(f"non-finite activations at layer {L}")
    return got


def capture_mean_activations(model, input_ids, attention_mask, layers):
    """Per-row mean of layer outputs over real tokens, excluding the first one."""
    used = _used_positions(attention_mask.cpu()).unsqueeze(-1)
    cnt = used.sum(dim=1)
    return {L: (h * used).sum(dim=1) / cnt
            for L, h in _capture(model, input_ids, attention_mask, layers).items()}


def mean_token_norm(model, input_ids, attention_mask, layers):
    """Mean per-token L2 norm over the same positions as capture_mean_activations."""
    used = _used_positions(attention_mask.cpu())
    return {L: float(h.norm(dim=-1)[used].mean())
            for L, h in _capture(model, input_ids, attention_mask, layers).items()}


@contextmanager
def steer(model, layer: int, direction: torch.Tensor, alpha: float, resid_norm: float):
    """Add alpha * resid_norm * unit(direction) to the output of block `layer`.

    Prefill (seq len > 1) skips the first non-padding position of each row;
    decode steps (seq len 1) always get the vector. No hook when alpha == 0.
    The attention mask is learned via a pre-hook on the top-level model (keyword or second
    positional argument). During prefill a mask that is present but unusable (not 2-D, or
    length != T) raises ValueError; only a missing mask (None) means "position 0 is first".
    `direction` must be 1-D of hidden size, finite, with nonzero norm (else ValueError).
    """
    if alpha == 0:
        yield
        return
    unit = direction.detach().float()
    hidden = model.get_input_embeddings().weight.shape[1]
    if unit.ndim != 1 or unit.shape[0] != hidden:
        raise ValueError(f"direction must have shape [{hidden}], got {tuple(unit.shape)}")
    if not torch.isfinite(unit).all() or unit.norm() == 0:
        raise ValueError("direction must be finite and nonzero")
    vec = alpha * resid_norm * unit / unit.norm()
    state: dict = {"mask": None}
    handles = []

    def stash(_m, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None and len(args) > 1:
            mask = args[1]
        state["mask"] = mask

    def hook(_m, _i, out):
        h = _out_tensor(out)
        add = vec.to(device=h.device, dtype=h.dtype).expand_as(h)
        T = h.shape[1]
        if T > 1:
            mask = state["mask"]
            if mask is None:  # no mask at all: position 0 is the first real token
                first = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            elif mask.dim() == 2 and mask.shape[1] == T:
                first = _first_real(mask.to(h.device))
            else:
                raise ValueError(f"steer: unusable attention_mask of shape {tuple(mask.shape)} "
                                 f"for prefill length {T}; need a 2-D [B, T] mask")
            keep = torch.arange(T, device=h.device)[None, :] > first[:, None]
            add = add * keep.unsqueeze(-1).to(h.dtype)
        new = h + add
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    try:
        handles.append(model.register_forward_pre_hook(stash, with_kwargs=True))
        handles.append(get_layers(model)[layer].register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()
