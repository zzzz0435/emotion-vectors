"""Answer probabilities, ratings, perplexity and judge scoring."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from emolab.batching import run_with_oom_backoff
from emolab.hooks import steer
from emolab.models import format_chat

JUDGE = ('Here is a short text:\n\n"""{text}"""\n\nOn a scale from 1 (not at all) to 5 (extremely), '
         "how much does this text express {phrase}? Answer with a single digit.")


@dataclass(frozen=True)
class SteerSpec:
    layer: int
    direction: torch.Tensor
    alpha: float
    resid_norm: float


def answer_token_ids(tokenizer, option: str) -> list[int]:
    """Ids of every single-token spelling of `option` (bare and with one leading space)."""
    ids: list[int] = []
    for s in (option, " " + option):
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(int(enc[0]))
    if not ids:
        raise ValueError(f"no single-token spelling for option {option!r}")
    return ids


def option_probs(last_logits: torch.Tensor,
                 option_ids: list[list[int]]) -> tuple[np.ndarray, float]:
    """Normalised option probabilities and the invalid mass (1 - valid)."""
    p = torch.softmax(last_logits.detach().float().cpu(), dim=-1)
    raw = np.array([float(p[ids].sum()) for ids in option_ids], dtype=np.float64)
    all_ids = sorted({i for ids in option_ids for i in ids})
    valid = float(p[all_ids].sum())
    invalid = 1.0 - valid
    if valid < 1e-6:
        return np.full(len(option_ids), np.nan), invalid
    return raw / raw.sum(), invalid


def expected_rating(probs: np.ndarray, values: list[int]) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    if np.isnan(probs).any():
        return float("nan")
    return float((probs * np.asarray(values, dtype=np.float64)).sum())


def harsh_high(rating: float, direction: Literal["light1", "harsh1"], scale_max: int = 7) -> float:
    return rating if direction == "light1" else scale_max + 1 - rating


def score_prompts(model, tokenizer, user_msgs: list[str], options: list[str],
                  batch_size: int = 16, steer_spec: SteerSpec | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Option probabilities [N, k] and invalid mass [N] from the last-position logits."""
    option_ids = [answer_token_ids(tokenizer, o) for o in options]
    texts = [format_chat(tokenizer, m) for m in user_msgs]
    dev = model.device

    def run(batch: list[str]):
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False, padding_side="left")
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        with torch.no_grad():
            if steer_spec is None:
                out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
            else:
                with steer(model, steer_spec.layer, steer_spec.direction,
                           steer_spec.alpha, steer_spec.resid_norm):
                    out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        last = out.logits[:, -1, :]
        return [option_probs(last[i], option_ids) for i in range(last.shape[0])]

    probs, invalid = [], []
    for _, res in run_with_oom_backoff(run, texts, batch_size):
        for p, inv in res:
            probs.append(p)
            invalid.append(inv)
    return np.array(probs).reshape(len(texts), len(options)), np.array(invalid)


def continuation_ppl(model, tokenizer, prompts: list[str], continuations: list[str],
                     batch_size: int = 8) -> np.ndarray:
    """exp(mean NLL over continuation tokens only), unsteered."""
    items = [(format_chat(tokenizer, p), c) for p, c in zip(prompts, continuations)]
    dev = model.device

    def run(batch):
        plen = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p, _ in batch]
        enc = tokenizer([p + c for p, c in batch], return_tensors="pt", padding=True,
                        add_special_tokens=False, padding_side="right")
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # predicts ids[:, 1:]
        pos = torch.arange(1, ids.shape[1], device=dev)[None, :]
        use = (pos >= torch.tensor(plen, device=dev)[:, None]) & (mask[:, 1:] != 0)
        n = use.sum(dim=1)
        if (n == 0).any():
            raise ValueError("empty continuation after tokenisation")
        return torch.exp((nll * use).sum(dim=1) / n).cpu().tolist()

    out: list[float] = []
    for _, res in run_with_oom_backoff(run, items, batch_size):
        out.extend(res)
    return np.array(out)


def judge_ratings(model, tokenizer, texts: list[str], phrase: str,
                  batch_size: int = 16) -> np.ndarray:
    msgs = [JUDGE.format(text=t, phrase=phrase) for t in texts]
    probs, _ = score_prompts(model, tokenizer, msgs, ["1", "2", "3", "4", "5"],
                             batch_size=batch_size)
    return np.array([expected_rating(p, [1, 2, 3, 4, 5]) for p in probs])
