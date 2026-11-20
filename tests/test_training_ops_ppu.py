import os
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import flag_gems


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    if torch.allclose(af, bf, atol=1e-6, rtol=1e-5):
        return 1.0
    denom = af.norm() * bf.norm()
    if denom == 0:
        return 1.0 if af.norm() == 0 and bf.norm() == 0 else 0.0
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def _hgbin_count(cache_dir: str, substr: str = "") -> int:
    cnt = 0
    for p in Path(cache_dir).rglob("*.hgbin"):
        if not substr or substr in p.name:
            cnt += 1
    return cnt


def test_layernorm_3d_4d_no_oob():
    print("[1] Testing LayerNorm 3D & 4D forward/backward + canary OOB check...")
    for dtype, cos_thr in ((torch.bfloat16, 0.999), (torch.float32, 0.99999)):
        for shape, norm_shape in (((2, 288, 768), (768,)), ((2, 4, 32, 64), (32, 64))):
            for affine in (True, False):
                torch.manual_seed(42)
                x0 = torch.randn(shape, device="cuda", dtype=dtype)
                dy = torch.randn(shape, device="cuda", dtype=dtype)
                w0 = torch.randn(norm_shape, device="cuda", dtype=dtype) if affine else None
                b0 = torch.randn(norm_shape, device="cuda", dtype=dtype) if affine else None

                xa = x0.clone().requires_grad_(True)
                wa = w0.clone().requires_grad_(True) if affine else None
                ba = b0.clone().requires_grad_(True) if affine else None
                ya = F.layer_norm(xa, norm_shape, wa, ba, 1e-5)
                ya.backward(dy)

                with flag_gems.use_gems():
                    can_before = torch.full((4096,), 0x123456789ABCDEF, dtype=torch.int64, device="cuda")
                    xg = x0.clone().requires_grad_(True)
                    wg = w0.clone().requires_grad_(True) if affine else None
                    bg = b0.clone().requires_grad_(True) if affine else None
                    can_after = torch.full((4096,), 0x123456789ABCDEF, dtype=torch.int64, device="cuda")
                    yg = F.layer_norm(xg, norm_shape, wg, bg, 1e-5)
                    yg.backward(dy)
                    torch.cuda.synchronize()
                    assert bool((can_before == 0x123456789ABCDEF).all())
                    assert bool((can_after == 0x123456789ABCDEF).all())

                assert cos_sim(yg, ya) >= cos_thr, f"LN fwd mismatch {dtype} {shape}"
                assert cos_sim(xg.grad, xa.grad) >= cos_thr, f"LN dx mismatch {dtype} {shape}"
                if affine:
                    assert cos_sim(wg.grad, wa.grad) >= cos_thr, f"LN dw mismatch {dtype} {shape}: {cos_sim(wg.grad, wa.grad)}"
                    assert cos_sim(bg.grad, ba.grad) >= cos_thr, f"LN db mismatch {dtype} {shape}: {cos_sim(bg.grad, ba.grad)}"
    print("  LayerNorm OK")


def test_sdpa_backward_bhsd_and_gqa():
    print("[2] Testing SDPA forward/backward BHSD & GQA causal...")
    cases = [
        ("mha_noncausal", (4, 12, 72, 64), (4, 12, 72, 64), False, False),
        ("mha_causal", (2, 8, 128, 64), (2, 8, 128, 64), True, False),
        ("gqa_causal", (1, 16, 264, 128), (1, 8, 264, 128), True, True),
    ]
    for name, q_shape, kv_shape, is_causal, enable_gqa in cases:
        torch.manual_seed(101)
        q0 = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
        k0 = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
        v0 = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)

        qa = q0.clone().requires_grad_(True)
        ka = k0.clone().requires_grad_(True)
        va = v0.clone().requires_grad_(True)
        oa = F.scaled_dot_product_attention(qa, ka, va, is_causal=is_causal, enable_gqa=enable_gqa)
        oa.backward(dout)

        with flag_gems.use_gems():
            qg = q0.clone().requires_grad_(True)
            kg = k0.clone().requires_grad_(True)
            vg = v0.clone().requires_grad_(True)
            og = F.scaled_dot_product_attention(qg, kg, vg, is_causal=is_causal, enable_gqa=enable_gqa)
            og.backward(dout)
            torch.cuda.synchronize()

        co = cos_sim(og, oa)
        cdq = cos_sim(qg.grad, qa.grad)
        cdk = cos_sim(kg.grad, ka.grad)
        cdv = cos_sim(vg.grad, va.grad)
        print(f"  {name}: out={co:.6f} dQ={cdq:.6f} dK={cdk:.6f} dV={cdv:.6f}")
        assert min(co, cdq, cdk, cdv) >= 0.998, f"SDPA mismatch on {name}"
    print("  SDPA OK")


def test_nll_loss_and_cross_entropy_ignore_index():
    print("[3] Testing NLLLoss & CrossEntropy forward/backward with ignore_index & multi-CTA...")
    for dtype, cos_thr, max_diff_thr in ((torch.float32, 0.99999, 1e-4), (torch.bfloat16, 0.999, 2e-2)):
        for N, C in ((32, 1024), (263, 151936), (2048, 256)):
            for reduction in ("mean", "sum", "none"):
                torch.manual_seed(77)
                inp0 = torch.randn(N, C, device="cuda", dtype=dtype)
                tgt = torch.randint(0, C, (N,), device="cuda", dtype=torch.long)
                tgt[0] = -100
                tgt[::5] = -100

                xa = inp0.clone().requires_grad_(True)
                la = F.cross_entropy(xa, tgt, reduction=reduction, ignore_index=-100)
                if reduction == "none":
                    grad_w = torch.randn_like(la)
                    la.backward(grad_w)
                else:
                    la.backward()

                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    with flag_gems.use_gems():
                        can_pre = torch.full((2048,), 0x76543210FEDCBA98 & ((1 << 63) - 1), dtype=torch.int64, device="cuda")
                        xg = inp0.clone().requires_grad_(True)
                        can_post = torch.full((2048,), 0x76543210FEDCBA98 & ((1 << 63) - 1), dtype=torch.int64, device="cuda")
                        lg = F.cross_entropy(xg, tgt, reduction=reduction, ignore_index=-100)
                        if reduction == "none":
                            lg.backward(grad_w)
                        else:
                            lg.backward()
                        torch.cuda.synchronize()
                        assert bool((can_pre == can_pre[0]).all())
                        assert bool((can_post == can_post[0]).all())
                        # Check ignored row gradient is exactly 0
                        assert bool((xg.grad[0] == 0).all()) and bool((xg.grad[::5] == 0).all())
                    for w in rec:
                        msg = str(w.message)
                        assert "Logical operators 'and' and 'or' are deprecated" not in msg, msg

                lg_f, la_f = lg.detach().float(), la.detach().float()
                loss_diff = float((lg_f - la_f).abs().max().item())
                rtol_thr = 1e-4 if dtype == torch.float32 else 1e-2
                assert torch.allclose(lg_f, la_f, rtol=rtol_thr, atol=max_diff_thr), (
                    f"CE loss diff {loss_diff} on {(N, C, dtype, reduction)}"
                )
                grad_cos = cos_sim(xg.grad, xa.grad)
                assert grad_cos >= cos_thr, f"CE grad cos {grad_cos} on {(N, C, dtype, reduction)}"
    print("  NLLLoss / CrossEntropy OK")


def test_fused_adam_fp32_and_bf16_zero_recompile():
    print("[4] Testing Adam(fused=True) accuracy & zero recompilation...")
    shapes = [(1,), (768,), (768, 768), (4096, 768)]
    lrs = [1e-4, 9e-5, 8e-5, 6e-5, 5e-5, 3e-5, 1e-5, 5e-6]
    orig_cache = os.environ.get("TRITON_CACHE_DIR")
    for dtype, cos_thr in ((torch.float32, 0.99999), (torch.bfloat16, 0.999)):
        for amsgrad in (False, True):
            torch.manual_seed(33)
            init_ps = [torch.randn(s, device="cuda", dtype=dtype) for s in shapes]
            step_gs = [[torch.randn(s, device="cuda", dtype=dtype) for s in shapes] for _ in lrs]

            # ATen reference (compute reference in float32 if bf16 ATen fused_adam is unavailable)
            ref_ps = [torch.nn.Parameter(p.clone()) for p in init_ps]
            try:
                opt_ref = torch.optim.Adam(ref_ps, lr=lrs[0], weight_decay=0.01, amsgrad=amsgrad, fused=True)
                for lr_v, gs in zip(lrs, step_gs):
                    for pg in opt_ref.param_groups:
                        pg["lr"] = lr_v
                    for p, g in zip(ref_ps, gs):
                        p.grad = g.clone()
                    opt_ref.step()
            except RuntimeError:
                ref_ps = [torch.nn.Parameter(p.clone()) for p in init_ps]
                opt_ref = torch.optim.Adam(ref_ps, lr=lrs[0], weight_decay=0.01, amsgrad=amsgrad, fused=False)
                for lr_v, gs in zip(lrs, step_gs):
                    for pg in opt_ref.param_groups:
                        pg["lr"] = lr_v
                    for p, g in zip(ref_ps, gs):
                        p.grad = g.clone()
                    opt_ref.step()

            with tempfile.TemporaryDirectory() as tdir:
                os.environ["TRITON_CACHE_DIR"] = tdir
                # Reset triton jit/libentry caches cleanly per test config? Check differential hgbin count in tdir!
                gems_ps = [torch.nn.Parameter(p.clone()) for p in init_ps]
                with flag_gems.use_gems():
                    opt_g = torch.optim.Adam(gems_ps, lr=lrs[0], weight_decay=0.01, amsgrad=amsgrad, fused=True)
                    step1_cnt = None
                    for idx, (lr_v, gs) in enumerate(zip(lrs, step_gs)):
                        for pg in opt_g.param_groups:
                            pg["lr"] = lr_v
                        for p, g in zip(gems_ps, gs):
                            p.grad = g.clone()
                        opt_g.step()
                        torch.cuda.synchronize()
                        cur_cnt = _hgbin_count(tdir, "_fused_adam_kernel")
                        if idx == 0:
                            step1_cnt = cur_cnt
                        else:
                            assert cur_cnt == step1_cnt, f"Recompiled in Adam step {idx+1}: {cur_cnt} != {step1_cnt}"
                    assert step1_cnt <= 1, f"Expected <=1 _fused_adam_kernel.hgbin on step 1, got {step1_cnt}"

                for gp, rp in zip(gems_ps, ref_ps):
                    c = cos_sim(gp, rp)
                    assert c >= cos_thr, f"Adam mismatch dtype={dtype} amsgrad={amsgrad} shape={tuple(gp.shape)} cos={c}"
    if orig_cache is not None:
        os.environ["TRITON_CACHE_DIR"] = orig_cache
    print("  Adam(fused=True) OK")


def test_embedding_noncontiguous():
    print("[5] Testing Embedding backward with non-contiguous inputs...")
    torch.manual_seed(9)
    w0 = torch.randn(1000, 768, device="cuda", dtype=torch.bfloat16)
    idx_base = torch.randint(0, 1000, (4, 64), device="cuda", dtype=torch.long)
    idx_nc = idx_base[:, ::2]  # non-contiguous (4, 32)
    go_base = torch.randn(32, 4, 768, device="cuda", dtype=torch.bfloat16)
    go_nc = go_base.transpose(0, 1)  # non-contiguous (4, 32, 768)

    wa = w0.clone().requires_grad_(True)
    ya = F.embedding(idx_nc, wa, padding_idx=1)
    ya.backward(go_nc)

    with flag_gems.use_gems():
        wg = w0.clone().requires_grad_(True)
        yg = F.embedding(idx_nc, wg, padding_idx=1)
        yg.backward(go_nc)
        torch.cuda.synchronize()

    assert cos_sim(yg, ya) >= 0.9999
    assert cos_sim(wg.grad, wa.grad) >= 0.9999
    print("  Embedding OK")


def test_thead_sdpa_overrideable_bucketing_no_recompile():
    print("[6] Testing _thead _scaled_dot_product_fused_attention_overrideable bucketing...")
    with tempfile.TemporaryDirectory() as tdir:
        old_cache = os.environ.get("TRITON_CACHE_DIR")
        old_bkt = triton.knobs.autotuning.shape_bucketing
        old_mode = triton.knobs.autotuning.shape_bucketing_mode
        try:
            os.environ["TRITON_CACHE_DIR"] = tdir
            triton.knobs.autotuning.shape_bucketing = True
            triton.knobs.autotuning.shape_bucketing_mode = "training"
            with flag_gems.use_gems():
                counts = []
                for L in (64, 128, 264, 276, 384, 524):
                    q = torch.randn(2, 8, L, 64, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn(2, 8, L, 64, device="cuda", dtype=torch.bfloat16)
                    v = torch.randn(2, 8, L, 64, device="cuda", dtype=torch.bfloat16)
                    _ = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(q, k, v)
                    torch.cuda.synchronize()
                    counts.append((L, _hgbin_count(tdir, "_sdpa_fwd"), _hgbin_count(tdir, "_sdpa_fwd_c")))
                print("  _sdpa_fwd counts across L:", counts)
                assert all(c_c == 0 for _, _, c_c in counts), f"_sdpa_fwd_c should not compile under bucketing: {counts}"
                # One %16==0 variant (64,128,384) and at most one %16!=0 variant (264,276,524).
                assert counts[-1][1] <= 2 and counts[3][1] == counts[2][1] and counts[5][1] == counts[4][1], (
                    f"_sdpa_fwd recompiled beyond alignment classes: {counts}"
                )
        finally:
            triton.knobs.autotuning.shape_bucketing = old_bkt
            triton.knobs.autotuning.shape_bucketing_mode = old_mode
            if old_cache is not None:
                os.environ["TRITON_CACHE_DIR"] = old_cache
    print("  _thead SDPA overrideable bucketing OK")


def test_mse_loss_backward_all_reductions():
    print("[7] Testing MSELoss forward/backward across mean/sum/none in fp32 & bf16...")
    for dtype, cos_thr, diff_thr in ((torch.float32, 0.99999, 1e-5), (torch.bfloat16, 0.999, 2e-2)):
        for shape in ((4, 768), (2, 1), (2, 288, 768)):
            for reduction in ("mean", "sum", "none"):
                torch.manual_seed(55)
                x0 = torch.randn(shape, device="cuda", dtype=dtype)
                tgt = torch.randn(shape, device="cuda", dtype=dtype)
                xa = x0.clone().requires_grad_(True)
                la = F.mse_loss(xa, tgt, reduction=reduction)
                g_out = torch.randn_like(la) if reduction == "none" else torch.tensor(1.75, device="cuda", dtype=dtype)
                la.backward(g_out)

                with flag_gems.use_gems():
                    xg = x0.clone().requires_grad_(True)
                    lg = F.mse_loss(xg, tgt, reduction=reduction)
                    lg.backward(g_out)
                    torch.cuda.synchronize()

                assert torch.allclose(lg.detach().float(), la.detach().float(), rtol=(1e-4 if dtype == torch.float32 else 2e-2), atol=diff_thr), f'MSE loss diff {shape} {dtype} {reduction}: {lg.item()} vs {la.item()}'
                c = cos_sim(xg.grad, xa.grad)
                assert c >= cos_thr, f"MSE backward mismatch {dtype} {shape} {reduction}: cos={c}"
    print("  MSELoss backward OK")


def test_sdpa_backward_with_dropout():
    print("[8] Testing SDPA backward with dropout_p=0.1 (philox parse)...")
    torch.manual_seed(202)
    q = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn_like(q)
    with flag_gems.use_gems():
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=False)
        out.backward(dout)
        torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()
    print("  SDPA dropout backward OK")


def test_linear_backward_noncontiguous():
    print("[9] Testing _thead linear_backward with non-contiguous input & grad_output...")
    torch.manual_seed(303)
    w0 = torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)
    x_c = torch.randn(4, 72, 768, device="cuda", dtype=torch.bfloat16)
    x_nc = torch.randn(72, 4, 768, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    go_nc = torch.randn(72, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(0, 1)

    xa = x_c.clone().requires_grad_(True)
    wa = w0.clone().requires_grad_(True)
    F.linear(xa, wa).backward(go_nc)

    with flag_gems.use_gems():
        xg = x_c.clone().requires_grad_(True)
        wg = w0.clone().requires_grad_(True)
        F.linear(xg, wg).backward(go_nc)
        gi_direct, gw_direct, _ = flag_gems.linear_backward(x_nc, go_nc, w0, (True, True, False))
        torch.cuda.synchronize()

    assert cos_sim(xg.grad, xa.grad) >= 0.999
    assert cos_sim(wg.grad, wa.grad) >= 0.999

    x_nc_a = x_nc.detach().clone().requires_grad_(True)
    w_nc_a = w0.clone().requires_grad_(True)
    F.linear(x_nc_a, w_nc_a).backward(go_nc)
    assert cos_sim(gi_direct, x_nc_a.grad) >= 0.999
    assert cos_sim(gw_direct, w_nc_a.grad) >= 0.999
    print("  linear_backward non-contiguous OK")


def test_select_backward_and_pad_zero_recompile():
    print("[10] Testing select_backward & F.pad zero recompilation across unseen seq_lens...")
    with tempfile.TemporaryDirectory() as tdir:
        old_cache = os.environ.get("TRITON_CACHE_DIR")
        old_bkt = triton.knobs.autotuning.shape_bucketing
        old_mode = triton.knobs.autotuning.shape_bucketing_mode
        try:
            os.environ["TRITON_CACHE_DIR"] = tdir
            triton.knobs.autotuning.shape_bucketing = True
            triton.knobs.autotuning.shape_bucketing_mode = "training"

            sel_counts = []
            pad_counts = []
            for L in (72, 92, 116, 140, 264, 384):
                torch.manual_seed(L)
                x0 = torch.randn(2, L, 768, device="cuda", dtype=torch.bfloat16)
                g0 = torch.randn(2, 768, device="cuda", dtype=torch.bfloat16)
                xa = x0.clone().requires_grad_(True)
                ya = xa[:, 0, :]
                ya.backward(g0)

                x2d = torch.randn(1, 1, L, L, device="cuda", dtype=torch.float32)
                x3d = torch.randn(2, 16, L, device="cuda", dtype=torch.bfloat16)
                p2_a = F.pad(x2d, (0, 8), value=-1.0)
                p3_a = F.pad(x3d, (0, 8), value=0.0)

                with flag_gems.use_gems():
                    xg = x0.clone().requires_grad_(True)
                    yg = xg[:, 0, :]
                    yg.backward(g0)
                    p2_g = F.pad(x2d, (0, 8), value=-1.0)
                    p3_g = F.pad(x3d, (0, 8), value=0.0)
                    torch.cuda.synchronize()

                assert cos_sim(xg.grad, xa.grad) >= 0.99999
                assert torch.allclose(p2_g, p2_a)
                assert torch.allclose(p3_g, p3_a)
                sel_counts.append((L, _hgbin_count(tdir, "_select_backward_kernel")))
                pad_counts.append((L, _hgbin_count(tdir, "_pad_jit_function")))

            print("  select_backward counts:", sel_counts)
            print("  pad counts:", pad_counts)
            assert sel_counts[0][1] == 1 and sel_counts[-1][1] == 1, f"select_backward recompiled: {sel_counts}"
            assert pad_counts[0][1] <= 2 and pad_counts[-1][1] == pad_counts[0][1], f"pad recompiled: {pad_counts}"
        finally:
            triton.knobs.autotuning.shape_bucketing = old_bkt
            triton.knobs.autotuning.shape_bucketing_mode = old_mode
            if old_cache is not None:
                os.environ["TRITON_CACHE_DIR"] = old_cache
    print("  select_backward & F.pad zero-recompile OK")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    test_layernorm_3d_4d_no_oob()
    test_sdpa_backward_bhsd_and_gqa()
    test_nll_loss_and_cross_entropy_ignore_index()
    test_fused_adam_fp32_and_bf16_zero_recompile()
    test_embedding_noncontiguous()
    test_thead_sdpa_overrideable_bucketing_no_recompile()
    test_mse_loss_backward_all_reductions()
    test_sdpa_backward_with_dropout()
    test_linear_backward_noncontiguous()
    test_select_backward_and_pad_zero_recompile()
    print("ALL TRAINING OP UNIT TESTS PASSED!")
