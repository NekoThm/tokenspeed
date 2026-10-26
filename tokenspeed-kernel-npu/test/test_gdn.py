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

"""Ascend GDN output, recurrent state, and padding checks."""

import json
import math
import unittest

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn import gdn_decode_step
from tokenspeed_kernel_npu.ops.gdn import gdn_decode, gdn_prefill

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires Ascend")


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "normalize,scale,value_heads,inplace",
    [
        (True, None, 16, False),
        (False, 0.03, 32, True),
    ],
)
def test_decode_state_updates(
    batch, state_dtype, normalize, scale, value_heads, inplace, monkeypatch
):
    import torch_npu

    calls = []
    native = torch_npu.npu_recurrent_gated_delta_rule

    def tracked(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_recurrent_gated_delta_rule", tracked)
    torch.manual_seed(13)
    heads, width = 16, 128
    # Page-interleaved state storage must preserve the gaps between rows.
    backing = (
        (torch.randn((batch * 2 + 1) * 2, value_heads, width, width) * 0.01)
        .to(state_dtype)
        .npu()
    )
    untouched_gaps = backing[1::2].cpu().clone()
    actual_state = backing[::2]
    reads = torch.arange(batch, dtype=torch.int32)
    writes = reads.clone() if inplace else reads + batch
    if batch > 1:
        reads[-1] = -1
        # A padding read must not write even with a nonnegative destination.
        if not inplace:
            writes[-2] = -1
        else:
            writes[-1] = -1
    A = torch.randn(value_heads) * 0.1
    bias = torch.randn(value_heads) * 0.1
    for step in range(3):
        state = actual_state.cpu().float()
        before = state.clone()
        q = (torch.randn(batch, 1, heads, width) * 0.1).bfloat16()
        k = torch.randn_like(q) * 0.1
        v = torch.randn(batch, 1, value_heads, width).bfloat16()
        a = torch.randn(batch, 1, value_heads).bfloat16()
        b = torch.randn_like(a)
        expected = []
        for row in range(batch):
            if reads[row] < 0:
                expected.append(torch.zeros(value_heads, width))
                continue
            qr, kr = q[row, 0].float(), k[row, 0].float()
            if normalize:
                qr = qr / torch.sqrt(qr.square().sum(-1, keepdim=True) + 1e-6)
                kr = kr / torch.sqrt(kr.square().sum(-1, keepdim=True) + 1e-6)
            qr = qr.bfloat16().float().repeat_interleave(value_heads // heads, dim=0)
            kr = kr.bfloat16().float().repeat_interleave(value_heads // heads, dim=0)
            h = before[reads[row]].bfloat16().float()
            decay = torch.exp(-A.exp() * F.softplus(a[row, 0].float() + bias))
            h *= decay[:, None, None]
            beta = b[row, 0].float().sigmoid().bfloat16().float()
            delta = (v[row, 0].float() - torch.einsum("hvk,hk->hv", h, kr)) * beta[
                :, None
            ]
            h += delta[:, :, None] * kr[:, None, :]
            if writes[row] >= 0:
                state[writes[row]] = h.bfloat16().float()
            expected.append(
                torch.einsum("hvk,hk->hv", h, qr)
                * (width**-0.5 if scale is None else scale)
            )
        out = gdn_decode_step(
            q.to("npu"),
            k.to("npu"),
            v.to("npu"),
            A_log=A.to("npu"),
            a=a.to("npu"),
            dt_bias=bias.to("npu"),
            b=b.to("npu"),
            initial_state=actual_state,
            initial_state_indices=reads.to("npu"),
            output_state_indices=None if inplace else writes.to("npu"),
            scale=scale,
            use_qk_l2norm=normalize,
        )
        # Native Q/K, beta and state are BF16; compare to FP32 recurrence
        # with the same input rounding, allowing one BF16 rounding unit.
        torch.testing.assert_close(
            out.cpu()[:, 0].float(), torch.stack(expected), atol=2e-4, rtol=0.008
        )
        torch.testing.assert_close(
            actual_state.cpu().float(), state, atol=2e-3, rtol=0.008
        )
        changed = writes[(reads >= 0) & (writes >= 0)].long()
        untouched = torch.ones(state.shape[0], dtype=torch.bool)
        untouched[changed] = False
        torch.testing.assert_close(
            actual_state.cpu().float()[untouched], before[untouched], atol=0, rtol=0
        )
        torch.testing.assert_close(backing[1::2].cpu(), untouched_gaps, atol=0, rtol=0)
        reads = writes.clone()
    assert len(calls) == 3


@pytest.mark.parametrize("lengths", [(1, 3), (0, 65), (2, 0, 3, 0), (0, 0)])
@pytest.mark.parametrize(
    "qk_l2norm,scale,value_heads,output_final_state",
    [
        (True, None, 16, True),
        (False, 0.03, 32, False),
    ],
)
def test_prefill_outputs_and_final_state(
    lengths, qk_l2norm, scale, value_heads, output_final_state
):
    from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill

    torch.manual_seed(17)
    heads, width = 16, 128
    total = sum(lengths)
    q = torch.randn(1, total, heads, width).bfloat16()
    k = torch.randn_like(q)
    if not qk_l2norm:
        q *= width**-0.5
        k *= width**-0.5
    v = torch.randn(1, total, value_heads, width).bfloat16()
    g = -torch.rand(1, total, value_heads)
    beta = torch.rand(1, total, value_heads)
    initial = torch.randn(len(lengths), value_heads, width, width) * 0.01
    expected_state = initial.clone()
    expected = torch.empty_like(v)
    bounds = [0]
    for seq, length in enumerate(lengths):
        for token in range(bounds[-1], bounds[-1] + length):
            qr, kr = q[0, token].float(), k[0, token].float()
            if qk_l2norm:
                qr /= torch.sqrt(qr.square().sum(-1, keepdim=True) + 1e-6)
                kr /= torch.sqrt(kr.square().sum(-1, keepdim=True) + 1e-6)
            qr = qr.repeat_interleave(value_heads // heads, dim=0)
            kr = kr.repeat_interleave(value_heads // heads, dim=0)
            h = expected_state[seq]
            h *= g[0, token].exp()[:, None, None]
            delta = (v[0, token].float() - torch.einsum("hvk,hk->hv", h, kr)) * beta[
                0, token, :, None
            ]
            h += delta[:, :, None] * kr[:, None, :]
            expected[0, token] = torch.einsum("hvk,hk->hv", h, qr) * (
                width**-0.5 if scale is None else scale
            )
        bounds.append(bounds[-1] + length)
    device_initial = initial.to("npu")
    actual = gdn_chunk_prefill(
        q.to("npu"),
        k.to("npu"),
        v.to("npu"),
        g.to("npu"),
        beta.to("npu"),
        scale=scale,
        initial_state=device_initial,
        cu_seqlens=torch.tensor(bounds, device="npu", dtype=torch.int32),
        qk_l2norm=qk_l2norm,
        output_final_state=output_final_state,
        output_h=False,
    )
    torch.testing.assert_close(actual.out.cpu(), expected, atol=2e-4, rtol=0.008)
    torch.testing.assert_close(device_initial.cpu(), initial, atol=0, rtol=0)
    if output_final_state:
        torch.testing.assert_close(
            actual.final_state.cpu(), expected_state, atol=2e-6, rtol=2e-4
        )
    else:
        assert actual.final_state is None
