import pytest
from transformers import AutoTokenizer

from emolab.models import MODELS, candidate_layers, format_chat, get_layers
from emolab.paths import REPO_ROOT, cache_dir, results_dir


def test_candidate_layers():
    assert candidate_layers(36) == [9, 14, 18, 23]
    assert candidate_layers(28) == [7, 11, 14, 18]
    assert candidate_layers(34) == [9, 14, 17, 22]
    assert candidate_layers(26) == [7, 10, 13, 17]


def test_get_layers_qwen_and_gemma(tiny_qwen3, tiny_gemma3):
    assert len(get_layers(tiny_qwen3)) == 4
    assert len(get_layers(tiny_gemma3)) == 4


def test_registry():
    assert set(MODELS) == {"qwen3-8b", "qwen3-1.7b", "gemma3-4b", "gemma3-1b"}
    assert MODELS["qwen3-8b"].quant == "4bit"
    assert MODELS["gemma3-4b"].quant == "8bit"
    assert MODELS["qwen3-1.7b"].name == "Qwen_3_1.7B_instruct"


def test_cache_dir_must_be_outside_repo(monkeypatch):
    monkeypatch.setenv("EMOTIONAL_CACHE", str(REPO_ROOT / "cache"))
    with pytest.raises(ValueError):
        cache_dir()


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("EMOTIONAL_CACHE", str(tmp_path / "c"))
    assert cache_dir() == tmp_path / "c" and cache_dir().is_dir()


def test_results_dir(monkeypatch, tmp_path):
    import emolab.paths as p
    monkeypatch.setattr(p, "RESULTS", tmp_path)
    assert p.results_dir("6.2_x", "M") == tmp_path / "6.2_x" / "M"
    assert (tmp_path / "6.2_x" / "M").is_dir()


@pytest.mark.network
def test_format_chat_qwen_disables_thinking():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    s = format_chat(tok, "Hello there")
    assert "Hello there" in s
    assert s.endswith("<think>\n\n</think>\n\n")
