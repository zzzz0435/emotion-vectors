import importlib.util
from pathlib import Path

import pytest
from huggingface_hub.errors import GatedRepoError

import emolab.models as models

_SPEC = importlib.util.spec_from_file_location(
    "check_env", Path(__file__).resolve().parents[1] / "scripts" / "6.0_env"
    / "01_check_environment.py")
check_env = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_env)


class _Resp:
    def __init__(self, code):
        self.status_code = code
        self.headers = {}
        self.request = None


def _raising(exc):
    def fake(key):
        raise exc
    return fake


def test_gemma_gated_gives_login_message(monkeypatch):
    monkeypatch.setattr(models, "load", _raising(GatedRepoError("gated", response=_Resp(403))))
    r = check_env.check_model("gemma3-4b")
    assert r["loaded"] is False and r["error"] == check_env.GEMMA_LOGIN_ERROR
    assert r["peak_vram_mb"] == 0.0 and r["tokens_per_s"] == 0.0


def test_gemma_other_error_is_not_mislabelled(monkeypatch):
    monkeypatch.setattr(models, "load", _raising(RuntimeError("bnb 8-bit failed")))
    r = check_env.check_model("gemma3-4b")
    assert r["loaded"] is False
    assert r["error"] == repr(RuntimeError("bnb 8-bit failed"))
    assert r["error"] != check_env.GEMMA_LOGIN_ERROR
