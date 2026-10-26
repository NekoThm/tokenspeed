# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Requires Ascend NPU"
)


@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gemma_norm_and_residual(rows, dtype):
    torch.manual_seed(42)
    layer = GemmaRMSNorm(1024).to(device="npu", dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight) * 0.1)
    x = torch.randn(rows, 1024, device="npu", dtype=dtype)
    residual = torch.randn_like(x)
    weight = (layer.weight.detach().cpu() + 1).float()

    def reference(value):
        value = value.float()
        scale = torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return (value * scale * weight).to(dtype)

    # BF16 allows one relative ulp; FP32 checks the reduction to 2e-6.
    atol = 0.03125 if dtype == torch.bfloat16 else 2e-6
    rtol = 0.0078125 if dtype == torch.bfloat16 else 2e-6
    with torch.no_grad():
        plain = layer(x.clone())
        expected_sum = x.cpu() + residual.cpu()
        normed, updated = layer(x.clone(), residual.clone())
        fused, fp8, updated_fused = layer.add_norm_with_fp8(
            x.clone(), residual.clone(), None
        )
    torch.testing.assert_close(
        plain.cpu(), reference(x.cpu()), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(updated.cpu(), expected_sum, atol=0, rtol=0)
    torch.testing.assert_close(
        normed.cpu(), reference(expected_sum), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(fused, normed, atol=0, rtol=0)
    torch.testing.assert_close(updated_fused, updated, atol=0, rtol=0)
    assert fp8 is None
