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
import torch.nn.functional as F

pytest.importorskip("torch_npu")
from tokenspeed.runtime.layers.attention.linear.layernorm_gated import rmsnorm_fn

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires Ascend")


@pytest.mark.parametrize("rows", [1, 17, 65])
@pytest.mark.parametrize("strided_gate", [False, True])
@pytest.mark.parametrize("norm_before_gate", [True, False])
def test_gated_norm(rows, norm_before_gate, strided_gate):
    torch.manual_seed(19)
    x = torch.randn(rows, 2048).bfloat16()
    z = torch.randn_like(x)
    weight = torch.randn(2048).bfloat16()
    gate = F.silu(z.float())
    value = x.float() if norm_before_gate else x.float() * gate
    grouped = value.reshape(rows, 16, 128)
    expected = (
        grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    ).reshape_as(value) * weight.float()
    if norm_before_gate:
        expected *= gate
    # Projection outputs can be views with gaps between token rows.
    z_device = (
        torch.cat([z, z], dim=1).to("npu")[:, :2048] if strided_gate else z.to("npu")
    )
    actual = rmsnorm_fn(
        x.to("npu"),
        weight.to("npu"),
        z=z_device,
        eps=1e-6,
        group_size=128,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=False,
        weights_independent=True,
    )
    torch.testing.assert_close(
        actual.cpu(), expected.bfloat16(), atol=0.0078125, rtol=0.0078125
    )
