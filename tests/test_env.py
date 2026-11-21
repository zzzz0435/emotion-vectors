import pytest, torch

@pytest.mark.gpu
def test_cuda_supports_this_gpu():
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()

@pytest.mark.gpu
def test_bnb_4bit_forward():
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
    out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (2, 64) and torch.isfinite(out).all()
