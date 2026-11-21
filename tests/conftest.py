import pytest
import torch
from transformers import (Gemma3ForCausalLM, Gemma3TextConfig, Qwen3Config,
                          Qwen3ForCausalLM)


@pytest.fixture
def tiny_qwen3():
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8)
    return Qwen3ForCausalLM(cfg).float().eval()


@pytest.fixture
def tiny_gemma3():
    torch.manual_seed(42)
    cfg = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=4, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8)
    return Gemma3ForCausalLM(cfg).float().eval()


class _FakeTok:
    """Char-level tokenizer stub: id = ord(c) % 120 + 4 (< 128); pad id 0."""
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def _ids(self, s):
        return [ord(c) % 120 + 4 for c in s]

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kw):
        s = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        return s + ("<assistant>\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=True):
        return self._ids(text)

    def convert_tokens_to_ids(self, tok):
        return self._ids(tok)[0]

    def __call__(self, texts, padding=False, return_tensors=None,
                 add_special_tokens=True, padding_side=None, **kw):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        rows = [self._ids(t) for t in texts]
        side = padding_side or self.padding_side
        T = max(len(r) for r in rows) if padding else None
        ids, mask = [], []
        for r in rows:
            pad = (T - len(r)) if padding else 0
            p = [self.pad_token_id] * pad
            m0, m1 = [0] * pad, [1] * len(r)
            ids.append(p + r if side == "left" else r + p)
            mask.append(m0 + m1 if side == "left" else m1 + m0)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids[0] if single else ids,
                "attention_mask": mask[0] if single else mask}


@pytest.fixture
def fake_tok():
    return _FakeTok()
