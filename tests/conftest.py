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
