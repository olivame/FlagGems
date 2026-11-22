#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable, self-contained F09 dynamic-shape bucketing acceptance runner.

Supports both local HuggingFace checkpoints (--model-dir) and a zero-dependency
synthetic Transformer model (--synthetic-fallback) across inference, training, and
incremental +1 decode modes. Isolates ATen, FlagGems OFF, and FlagGems ON in clean
subprocesses with independent TRITON_CACHE_DIR and HOME directories so neither disk
nor in-memory caches bleed across variants. Enforces USE_FLAGTUNE_COST_MODEL=0 by
default to prevent remote cost-model download stalls in offline environments.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlagOS F09 portable dynamic-shape bucketing acceptance benchmark runner"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional path to a local HuggingFace model directory (uses AutoModel).",
    )
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Run a deterministic self-contained PyTorch Transformer (zero external weights).",
    )
    parser.add_argument(
        "--mode",
        choices=("infer", "train", "decode"),
        default="infer",
        help="Benchmark workload mode (default: infer).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device string (e.g. cuda, cuda:0, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model compute dtype on accelerator (cpu automatically uses fp32).",
    )
    parser.add_argument(
        "--bucketing-mode",
        choices=("auto", "inference", "training"),
        default="auto",
        help="FlagTree shape_bucketing_mode (default: auto from --mode).",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2).")
    parser.add_argument("--hidden-size", type=int, default=256, help="Synthetic hidden dim.")
    parser.add_argument("--num-layers", type=int, default=2, help="Synthetic Transformer layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Synthetic attention heads.")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Synthetic vocab size.")
    parser.add_argument(
        "--warmup-lengths",
        type=str,
        default="64,128,256,384",
        help="Comma-separated warmup sequence lengths.",
    )
    parser.add_argument(
        "--test-lengths",
        type=str,
        default="45,72,92,140,188,250,276,340",
        help="Comma-separated unseen dynamic sequence lengths for evaluation.",
    )
    parser.add_argument("--decode-base-len", type=int, default=64, help="Decode start length.")
    parser.add_argument("--decode-steps", type=int, default=8, help="Decode +1 steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--allow-cost-model",
        action="store_true",
        help="Do not force USE_FLAGTUNE_COST_MODEL=0 (not recommended offline).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path for structured JSON report.",
    )
    parser.add_argument("--_worker-variant", choices=("aten", "off", "on"), default=None)
    parser.add_argument("--_worker-result", type=Path, default=None)
    parser.add_argument("--_worker-tensors", type=Path, default=None)
    return parser.parse_args(argv)


def _parse_int_list(spec: str) -> List[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty integer list, got: {spec!r}")
    return vals


def _sync_device(torch_mod: Any, device: str) -> None:
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def _count_cache_artifacts(cache_dir: str) -> int:
    root = Path(cache_dir)
    if not root.exists():
        return 0
    binary_exts = (".hgbin", ".cubin", ".hsaco", ".spv")
    bins = [p for p in root.rglob("*") if p.is_file() and p.suffix in binary_exts]
    if bins:
        return len(bins)
    fallback_exts = (".ptx", ".so", ".json")
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix in fallback_exts)


def _parity_metrics(torch_mod: Any, a: Any, b: Any) -> Dict[str, Any]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = (af - bf).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mae = float(diff.mean().item()) if diff.numel() else 0.0
    if torch_mod.allclose(af, bf, atol=1e-6, rtol=1e-5):
        cos = 1.0
    else:
        denom = af.norm() * bf.norm()
        if denom.item() == 0.0:
            cos = 1.0 if af.norm().item() == 0.0 and bf.norm().item() == 0.0 else 0.0
        else:
            cos = float(
                torch_mod.nn.functional.cosine_similarity(
                    af.unsqueeze(0), bf.unsqueeze(0)
                ).item()
            )
    return {
        "cos": cos,
        "max_abs_err": max_abs,
        "mae": mae,
        "all_finite": bool(torch_mod.isfinite(af).all().item()),
    }


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _build_synthetic_model(
    torch_mod: Any,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    is_causal: bool,
):
    nn = torch_mod.nn

    class _SyntheticTransformerBlock(nn.Module):
        def __init__(self, d_model: int, nhead: int, causal: bool) -> None:
            super().__init__()
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.causal = causal
            self.norm1 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn1 = nn.Linear(d_model, 2 * d_model)
            self.ffn2 = nn.Linear(2 * d_model, d_model)

        def forward(self, x):
            bsz, seqlen, dim = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(bsz, seqlen, 3, self.nhead, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            attn = torch_mod.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal
            )
            attn = attn.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            x = x + self.proj(attn)
            h2 = self.norm2(x)
            x = x + self.ffn2(torch_mod.nn.functional.silu(self.ffn1(h2)))
            return x

    class _SyntheticTransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.blocks = nn.ModuleList(
                [
                    _SyntheticTransformerBlock(hidden_size, num_heads, is_causal)
                    for _ in range(num_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.final_norm(x))

    return _SyntheticTransformerModel()


def _load_model(
    args: argparse.Namespace, torch_mod: Any, device: str, dtype: Any
) -> Tuple[Any, int, str]:
    torch_mod.manual_seed(args.seed)
    if args.model_dir:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True)
        model = model.to(device=device, dtype=dtype)
        vocab = int(getattr(model.config, "vocab_size", args.vocab_size))
        return model, vocab, f"hf_local:{Path(args.model_dir).name}"

    model = _build_synthetic_model(
        torch_mod,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        is_causal=(args.mode == "decode"),
    ).to(device=device, dtype=dtype)
    return model, args.vocab_size, (
        f"synthetic_transformer(L={args.num_layers},D={args.hidden_size},H={args.num_heads})"
    )


def _extract_tensor(out: Any) -> Any:
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _run_infer_or_decode_pass(
    torch_mod: Any,
    model: Any,
    inputs: Dict[int, Any],
    lengths: List[int],
    device: str,
) -> Tuple[Dict[int, Any], List[float]]:
    outputs: Dict[int, Any] = {}
    lat_ms: List[float] = []
    with torch_mod.no_grad():
        for seq_len in lengths:
            _sync_device(torch_mod, device)
            t0 = time.perf_counter()
            out = _extract_tensor(model(inputs[seq_len]))
            _sync_device(torch_mod, device)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            outputs[seq_len] = out.detach().cpu()
    return outputs, lat_ms


def _run_train_pass(
    torch_mod: Any,
    base_state: Dict[str, Any],
    model: Any,
    inputs: Dict[int, Any],
    lengths: List[int],
    device: str,
) -> Tuple[Dict[int, Any], List[float], List[float]]:
    model.load_state_dict(base_state)
    model.train()
    opt = torch_mod.optim.AdamW(model.parameters(), lr=1e-4)
    outputs: Dict[int, Any] = {}
    lat_ms: List[float] = []
    losses: List[float] = []
    for seq_len in lengths:
        opt.zero_grad(set_to_none=True)
        _sync_device(torch_mod, device)
        t0 = time.perf_counter()
        out = _extract_tensor(model(inputs[seq_len]))
        loss = (out.float() ** 2).mean()
        loss.backward()
        opt.step()
        _sync_device(torch_mod, device)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        losses.append(float(loss.item()))
        outputs[seq_len] = out.detach().cpu()
    return outputs, lat_ms, losses


def _run_worker(args: argparse.Namespace) -> int:
    if not args.allow_cost_model:
        os.environ.setdefault("USE_FLAGTUNE_COST_MODEL", "0")

    import torch
    import triton
    import flag_gems

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = torch.float32 if device == "cpu" else dtype_map[args.dtype]
    bkt_mode = (
        ("training" if args.mode == "train" else "inference")
        if args.bucketing_mode == "auto"
        else args.bucketing_mode
    )
    if args.mode == "decode":
        warmup_lens = [args.decode_base_len]
        test_lens = [args.decode_base_len + i for i in range(1, args.decode_steps + 1)]
    else:
        warmup_lens = _parse_int_list(args.warmup_lengths)
        test_lens = _parse_int_list(args.test_lengths)

    model, vocab_size, model_label = _load_model(args, torch, device, dtype)
    model.eval()
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    all_lens = sorted(set(warmup_lens + test_lens))
    torch.manual_seed(args.seed)
    inputs = {
        seq_len: torch.randint(
            0, vocab_size, (args.batch_size, seq_len), device=device, dtype=torch.long
        )
        for seq_len in all_lens
    }
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
    variant = args._worker_variant

    if variant == "aten":
        if args.mode == "train":
            outs, lat_ms, losses = _run_train_pass(
                torch, base_state, model, inputs, test_lens, device
            )
        else:
            _run_infer_or_decode_pass(torch, model, inputs, warmup_lens[:1], device)
            outs, lat_ms = _run_infer_or_decode_pass(
                torch, model, inputs, test_lens, device
            )
            losses = []
        payload: Dict[str, Any] = {
            "total_ms": round(sum(lat_ms), 3),
            "p50_ms": round(statistics.median(lat_ms), 3),
            "losses": [round(x, 6) for x in losses],
            "meta": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "device": device,
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if device.startswith("cuda") and torch.cuda.is_available()
                    else "cpu"
                ),
                "dtype": str(dtype),
                "model": model_label,
                "bucketing_mode": bkt_mode,
                "warmup_lengths": warmup_lens,
                "test_lengths": test_lens,
                "use_flagtune_cost_model": os.environ.get("USE_FLAGTUNE_COST_MODEL", ""),
            },
        }
    else:
        enable_bkt = variant == "on"
        with triton.knobs.autotuning.scope():
            triton.knobs.autotuning.shape_bucketing = enable_bkt
            triton.knobs.autotuning.shape_bucketing_mode = bkt_mode
            with flag_gems.use_gems():
                if args.mode == "train":
                    _, warm_ms, _ = _run_train_pass(
                        torch, base_state, model, inputs, warmup_lens, device
                    )
                    warm_art = _count_cache_artifacts(cache_dir)
                    outs, first_ms, losses = _run_train_pass(
                        torch, base_state, model, inputs, test_lens, device
                    )
                    post_art = _count_cache_artifacts(cache_dir)
                    _, steady_ms, _ = _run_train_pass(
                        torch, base_state, model, inputs, test_lens, device
                    )
                else:
                    model.eval()
                    _, warm_ms = _run_infer_or_decode_pass(
                        torch, model, inputs, warmup_lens, device
                    )
                    warm_art = _count_cache_artifacts(cache_dir)
                    outs, first_ms = _run_infer_or_decode_pass(
                        torch, model, inputs, test_lens, device
                    )
                    post_art = _count_cache_artifacts(cache_dir)
                    _, steady_ms = _run_infer_or_decode_pass(
                        torch, model, inputs, test_lens, device
                    )
                    losses = []
        payload = {
            "bucketing": enable_bkt,
            "warmup_ms_total": round(sum(warm_ms), 3),
            "post_warm_first_ms_total": round(sum(first_ms), 3),
            "post_warm_first_p50_ms": round(statistics.median(first_ms), 3),
            "post_warm_first_p99_ms": round(_percentile(first_ms, 99.0), 3),
            "steady_ms_total": round(sum(steady_ms), 3),
            "steady_p50_ms": round(statistics.median(steady_ms), 3),
            "steady_p99_ms": round(_percentile(steady_ms, 99.0), 3),
            "warmup_artifacts": warm_art,
            "total_artifacts": post_art,
            "post_warm_new_artifacts": max(0, post_art - warm_art),
            "losses": [round(x, 6) for x in losses],
        }

    assert args._worker_result is not None and args._worker_tensors is not None
    args._worker_result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(outs, args._worker_tensors)
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    if not args.model_dir and not args.synthetic_fallback:
        args.synthetic_fallback = True
    if not args.allow_cost_model:
        os.environ.setdefault("USE_FLAGTUNE_COST_MODEL", "0")

    import torch

    base_argv = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        args.mode,
        "--dtype",
        args.dtype,
        "--bucketing-mode",
        args.bucketing_mode,
        "--batch-size",
        str(args.batch_size),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--vocab-size",
        str(args.vocab_size),
        "--warmup-lengths",
        args.warmup_lengths,
        "--test-lengths",
        args.test_lengths,
        "--decode-base-len",
        str(args.decode_base_len),
        "--decode-steps",
        str(args.decode_steps),
        "--seed",
        str(args.seed),
    ]
    if args.model_dir:
        base_argv.extend(["--model-dir", args.model_dir])
    if args.synthetic_fallback:
        base_argv.append("--synthetic-fallback")
    if args.device:
        base_argv.extend(["--device", args.device])
    if args.allow_cost_model:
        base_argv.append("--allow-cost-model")

    results: Dict[str, Dict[str, Any]] = {}
    tensors: Dict[str, Dict[int, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="f09_acc_") as work_dir:
        for variant in ("aten", "off", "on"):
            vdir = Path(work_dir) / variant
            vdir.mkdir(parents=True, exist_ok=True)
            res_json = vdir / "result.json"
            res_pt = vdir / "outputs.pt"
            env = os.environ.copy()
            env["TRITON_CACHE_DIR"] = str(vdir / "triton_cache")
            env["HOME"] = str(vdir / "home")
            if not args.allow_cost_model:
                env["USE_FLAGTUNE_COST_MODEL"] = "0"
            Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
            cmd = base_argv + [
                "--_worker-variant",
                variant,
                "--_worker-result",
                str(res_json),
                "--_worker-tensors",
                str(res_pt),
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout, file=sys.stderr)
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"Worker variant {variant!r} failed (rc={proc.returncode})")
            results[variant] = json.loads(res_json.read_text(encoding="utf-8"))
            tensors[variant] = torch.load(res_pt, map_location="cpu", weights_only=True)

    meta = results["aten"].pop("meta")
    test_lens = meta["test_lengths"]
    on_vs_aten = [
        _parity_metrics(torch, tensors["on"][s], tensors["aten"][s]) for s in test_lens
    ]
    on_vs_off = [
        _parity_metrics(torch, tensors["on"][s], tensors["off"][s]) for s in test_lens
    ]

    parity = {
        "min_cos_on_vs_aten": round(min(m["cos"] for m in on_vs_aten), 6),
        "min_cos_on_vs_off": round(min(m["cos"] for m in on_vs_off), 6),
        "max_abs_err_on_vs_aten": round(max(m["max_abs_err"] for m in on_vs_aten), 6),
        "mae_on_vs_aten": round(max(m["mae"] for m in on_vs_aten), 6),
        "all_finite": all(m["all_finite"] for m in on_vs_aten),
    }
    off_res = results["off"]
    on_res = results["on"]
    first_speedup = off_res["post_warm_first_ms_total"] / max(
        on_res["post_warm_first_ms_total"], 1e-6
    )
    steady_ratio = off_res["steady_ms_total"] / max(on_res["steady_ms_total"], 1e-6)
    art_comp = off_res["total_artifacts"] / max(on_res["total_artifacts"], 1)

    report: Dict[str, Any] = {
        "schema": "f09.acceptance_runner.v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "environment": {
            "python": meta["python"],
            "torch": meta["torch"],
            "triton": meta["triton"],
            "device": meta["device"],
            "device_name": meta["device_name"],
            "use_flagtune_cost_model": meta["use_flagtune_cost_model"],
        },
        "config": {
            "model": meta["model"],
            "mode": args.mode,
            "dtype": meta["dtype"],
            "bucketing_mode": meta["bucketing_mode"],
            "batch_size": args.batch_size,
            "warmup_lengths": meta["warmup_lengths"],
            "test_lengths": test_lens,
            "seed": args.seed,
        },
        "aten_ref": results["aten"],
        "flag_gems_off": off_res,
        "flag_gems_on": on_res,
        "parity": parity,
        "summary": {
            "post_warm_first_speedup_x": round(first_speedup, 3),
            "steady_parity_x": round(steady_ratio, 3),
            "artifact_compression_x": round(art_comp, 3),
            "on_post_warm_new_artifacts": on_res["post_warm_new_artifacts"],
        },
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== FlagOS F09 Acceptance Runner Summary ===")
    print(
        f"  Model / Mode  : {meta['model']} | mode={args.mode} | "
        f"dtype={meta['dtype']} | dev={meta['device_name']}"
    )
    print(f"  Lengths       : warmup={meta['warmup_lengths']} | test={test_lens}")
    print(
        f"  ATen Ref      : total={results['aten']['total_ms']:.2f}ms "
        f"(p50={results['aten']['p50_ms']:.2f}ms)"
    )
    print(
        f"  FlagGems OFF  : post_warm_1st={off_res['post_warm_first_ms_total']:.2f}ms "
        f"steady={off_res['steady_ms_total']:.2f}ms "
        f"artifacts={off_res['total_artifacts']} (+{off_res['post_warm_new_artifacts']} post-warm)"
    )
    print(
        f"  FlagGems ON   : post_warm_1st={on_res['post_warm_first_ms_total']:.2f}ms "
        f"steady={on_res['steady_ms_total']:.2f}ms "
        f"artifacts={on_res['total_artifacts']} (+{on_res['post_warm_new_artifacts']} post-warm)"
    )
    print(
        f"  Speedup/Comp  : post_warm_1st={first_speedup:.2f}x | "
        f"steady_parity={steady_ratio:.3f}x | artifact_comp={art_comp:.2f}x"
    )
    print(
        f"  Parity        : min_cos(ON,ATen)={parity['min_cos_on_vs_aten']:.6f} | "
        f"min_cos(ON,OFF)={parity['min_cos_on_vs_off']:.6f} | "
        f"max_abs={parity['max_abs_err_on_vs_aten']:.6f} | "
        f"mae={parity['mae_on_vs_aten']:.6f} | finite={parity['all_finite']}"
    )
    if args.out is not None:
        print(f"  JSON Written  : {args.out}")
    if not parity["all_finite"] or parity["min_cos_on_vs_aten"] < 0.99:
        print(
            "FAIL: Numerical parity check did not meet threshold (cos >= 0.99, finite)",
            file=sys.stderr,
        )
        return 1
    print("PASS: F09 acceptance smoke check succeeded.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args._worker_variant is not None:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
