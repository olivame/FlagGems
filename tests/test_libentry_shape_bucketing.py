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

import ast
import inspect
import os
import sys
import textwrap
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def _bootstrap_test_env() -> None:
    """Ensure FlagGems + FlagTree import cleanly in both container and host-unit environments."""
    try:
        from triton._C.libtriton import getenv as _g  # noqa: F401
        import flag_gems  # noqa: F401
        return
    except Exception:
        pass

    fg_root = Path(__file__).resolve().parents[1]
    fg_src = fg_root / "src"
    if str(fg_src) not in sys.path:
        sys.path.insert(0, str(fg_src))

    ft_candidate = fg_root.parent / "FlagTree" / "python"
    if ft_candidate.exists() and str(ft_candidate) not in sys.path:
        sys.path.insert(0, str(ft_candidate))

    try:
        from triton._C.libtriton import getenv as _g  # noqa: F401
    except Exception:
        if ft_candidate.exists():
            triton_pkg = types.ModuleType("triton")
            triton_pkg.__path__ = [str(ft_candidate / "triton")]
            triton_pkg.__version__ = "3.3.0"
            triton_pkg.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (int(n) - 1).bit_length()
            triton_pkg.cdiv = lambda x, y: (x + y - 1) // y
            c_pkg = types.ModuleType("triton._C")
            c_pkg.__path__ = []
            lib_stub = types.ModuleType("triton._C.libtriton")
            lib_stub.__path__ = []
            lib_stub.get_cache_invalidating_env_vars = lambda: {}
            lib_stub.native_specialize_impl = lambda *a, **kw: None
            lib_stub.ir = SimpleNamespace()
            lib_stub.passes = SimpleNamespace()
            lib_stub.llvm = SimpleNamespace()
            lib_stub.interpreter = SimpleNamespace()
            ll_mod = types.ModuleType("triton._C.libtriton.linear_layout")
            ll_mod.LinearLayout = object
            lib_stub.linear_layout = ll_mod
            lib_stub.getenv = lambda key, default=None: os.environ.get(key, default)
            lib_stub.getenv_bool = lambda key, default=None: (
                default
                if key not in os.environ
                else os.environ[key].strip().lower() in ("1", "true", "yes", "y", "on")
            )
            c_pkg.libtriton = lib_stub
            triton_pkg._C = c_pkg
            sys.modules["triton"] = triton_pkg
            sys.modules["triton._C"] = c_pkg
            sys.modules["triton._C.libtriton"] = lib_stub
            sys.modules["triton._C.libtriton.linear_layout"] = ll_mod
            import triton.knobs as _tr_knobs
            import triton.runtime as _tr_rt
            from triton.runtime.autotuner import Config as _Cfg
            triton_pkg.knobs = _tr_knobs
            triton_pkg.runtime = _tr_rt
            triton_pkg.KernelInterface = _tr_rt.KernelInterface
            triton_pkg.Config = _Cfg

    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")

        class _StubTensor:

            def __init__(self, dtype="torch.bfloat16", ptr=4096):
                self.dtype = dtype
                self._ptr = ptr

            def data_ptr(self):
                return self._ptr

        torch_stub.Tensor = _StubTensor
        torch_stub.bfloat16 = "torch.bfloat16"
        torch_stub.float16 = "torch.float16"
        torch_stub.float32 = "torch.float32"
        sys.modules["torch"] = torch_stub

        # Stub minimal flag_gems runtime & SQL model modules when running without torch/sqlalchemy
        fg_pkg = types.ModuleType("flag_gems")
        fg_pkg.__path__ = [str(fg_src / "flag_gems")]
        sys.modules["flag_gems"] = fg_pkg

        rt_mod = types.ModuleType("flag_gems.runtime")
        rt_mod.__path__ = []
        rt_mod.device = SimpleNamespace(device_count=1, name="cpu")
        rt_mod.torch_device_fn = SimpleNamespace(current_device=lambda: 0)

        class _TuningMode(str):
            pass

        _TuningMode.DEFAULT = _TuningMode("default")
        _TuningMode.COST_MODEL = _TuningMode("cost_model")
        rt_mod.TuningMode = _TuningMode
        rt_mod.CostModelIntent = SimpleNamespace(AUTO="auto", REQUIRED="required")
        rt_mod.resolve_cost_model_intent = lambda **kw: "auto"
        fg_pkg.runtime = rt_mod
        sys.modules["flag_gems.runtime"] = rt_mod

        be_mod = types.ModuleType("flag_gems.runtime.backend")
        be_mod._state = SimpleNamespace(
            vendor_module=SimpleNamespace(vendor_info=SimpleNamespace(vendor_name="generic"))
        )
        sys.modules["flag_gems.runtime.backend"] = be_mod

        ft_inf = types.ModuleType("flag_gems.flagtune")
        ft_inf.__path__ = [str(fg_src / "flag_gems" / "flagtune")]
        cm_mod = types.ModuleType("flag_gems.flagtune.inference")
        cm_mod.cost_model = SimpleNamespace(_COST_MODEL_DISABLED_OPS=set())
        sys.modules["flag_gems.flagtune"] = ft_inf
        sys.modules["flag_gems.flagtune.inference"] = cm_mod

        cc_mod = types.ModuleType("flag_gems.utils.code_cache")
        cc_mod.config_cache_dir = lambda: Path("/tmp")
        cc_mod.code_cache_dir = lambda: Path("/tmp")
        utils_pkg = types.ModuleType("flag_gems.utils")
        utils_pkg.__path__ = [str(fg_src / "flag_gems" / "utils")]
        sys.modules["flag_gems.utils"] = utils_pkg
        sys.modules["flag_gems.utils.code_cache"] = cc_mod

        models_mod = types.ModuleType("flag_gems.utils.models")

        class _PersistantModel:
            pass

        class _InMemSQLModel(_PersistantModel):

            def __init__(self, db_url=None):
                self._cfg = {}
                self._bench = {}

            def get_config(self, table, key):
                return self._cfg.get((table, tuple(key)))

            def put_config(self, table, key, cfg):
                self._cfg[(table, tuple(key))] = cfg

            def get_benchmark(self, table, key, cfg=None):
                if cfg is None:
                    return None
                return self._bench.get((table, tuple(key), str(cfg)))

            def put_benchmark(self, table, key, cfg, val):
                self._bench[(table, tuple(key), str(cfg))] = val

        models_mod.PersistantModel = _PersistantModel
        models_mod.SQLPersistantModel = _InMemSQLModel
        sys.modules["flag_gems.utils.models"] = models_mod


_bootstrap_test_env()

import importlib  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
from triton.runtime.autotuner import Config  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

lib = importlib.import_module("flag_gems.utils.libentry")


def _make_tensor(dtype=None, aligned=True):
    dt = dtype if dtype is not None else torch.bfloat16
    if hasattr(torch, "empty") and callable(getattr(torch, "empty", None)):
        return torch.empty((16,), dtype=dt, device="cpu")
    return torch.Tensor(dtype=str(dt), ptr=(4096 if aligned else 4100))


class _RecordedKernel:

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def _launch(*args):
            self.launches.append((grid, args))

        return _launch


class _FakeJIT(JITFunction):

    def __init__(self, src: str, dns_names=()):
        self._src = textwrap.dedent(src)
        fndef = next(n for n in ast.walk(ast.parse(self._src)) if isinstance(n, ast.FunctionDef))
        self.__name__ = fndef.name
        self.__globals__ = {}
        self.arg_names = [a.arg for a in fndef.args.args]
        self.params = []
        sig_params = []
        for a in fndef.args.args:
            is_ce = (
                a.annotation is not None
                and "constexpr" in ast.unparse(a.annotation)
            )
            is_dns = a.arg in dns_names
            self.params.append(
                SimpleNamespace(
                    name=a.arg,
                    is_constexpr=is_ce,
                    do_not_specialize=is_dns,
                    default=inspect._empty,
                )
            )
            sig_params.append(
                inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
        self.signature = inspect.Signature(sig_params)

        def _raw():
            return None

        _raw.__name__ = self.__name__
        self.fn = _raw
        self.compiled_kernels = []

    @property
    def cache_key(self) -> str:
        return f"ck_{self.__name__}"

    def parse(self):
        return ast.parse(self._src)

    def run(self, *args, **kwargs):
        rk = _RecordedKernel()
        self.compiled_kernels.append((rk, args, dict(kwargs)))
        grid = kwargs["grid"]
        if callable(grid):
            grid = grid({**dict(zip(self.arg_names, args)), **kwargs})
        rk[(grid + (1, 1))[:3]](*args)
        return rk


def _build_synthetic_libentry(jit_or_tuner, jit_fn: _FakeJIT, cache_binding: bool = True) -> lib.LibEntry:
    e = object.__new__(lib.LibEntry)
    e.fn = jit_or_tuner
    e.jit_function = jit_fn
    e.arg_names = list(jit_fn.arg_names)
    e.signature = jit_fn.signature
    e.specialize_indices = [
        i for i, p in enumerate(jit_fn.params) if not p.is_constexpr and not p.do_not_specialize
    ]
    e.do_not_specialize_indices = [
        i for i, p in enumerate(jit_fn.params) if p.do_not_specialize
    ]
    e.divisibility = 16
    e._tensor_spec_fn = None
    e._has_flagtune_tuner = False
    e._flagtune_tuners = ()
    e._flagtune_state = None
    e.lock = threading.Lock()
    e.kernel_cache = ({},)
    e._cpu_cache = {}
    e._cache_binding = cache_binding
    if cache_binding:
        e._init_cached_binding()
    return e


@unittest.skipUnless(lib._HAS_FLAGTREE_SHAPE_BUCKETING, "Requires FlagTree shape bucketing")
class TestLibEntryFlagTreeBucketing(unittest.TestCase):

    def setUp(self):
        from triton.runtime.adjust_kernel_param import clear_analysis_cache
        from triton.runtime.shape_bucketing import GLOBAL_BUCKET_TELEMETRY, clear_policy_caches

        clear_analysis_cache()
        clear_policy_caches()
        GLOBAL_BUCKET_TELEMETRY.reset()

    def test_disabled_knob_preserves_exact_upstream_behavior_and_new_without_init(self):
        jit_fn = _FakeJIT(
            """
            def add_kern(X, Y, N, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                v = tl.load(X + offs, mask=offs < N)
                tl.store(Y + offs, v, mask=offs < N)
            """
        )
        e = _build_synthetic_libentry(jit_fn, jit_fn, cache_binding=True)
        x_t, y_t = _make_tensor(), _make_tensor()

        with triton.knobs.autotuning.scope():
            triton.knobs.autotuning.shape_bucketing = False
            triton.knobs.autotuning.adjust_block_size = False
            e.run(x_t, y_t, 513, BLOCK=128, grid=lambda m: ((m["N"] + 127) // 128,))
            e.run(x_t, y_t, 514, BLOCK=128, grid=lambda m: ((m["N"] + 127) // 128,))
            # Without shape_bucketing, exact ints 513 and 514 create 2 separate cache entries
            active_cache = e._cpu_cache if lib.torch_device_fn.current_device() == "cpu" else e.kernel_cache[0]
            self.assertEqual(len(active_cache), 2)

    def test_decode_kv_ctx_plus_one_zero_retune_and_lock_free_warm_hits(self):
        attn_src = """
        def decode_attn_kern(Q, K, Out, Q_CTX, KV_CTX, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
            tl.static_assert(BLOCK_N <= HEAD_DIM)
            offs_n = tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, HEAD_DIM)
            q = tl.load(Q + offs_d)
            acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
            for start_n in range(0, KV_CTX, BLOCK_N):
                mask = (start_n + offs_n) < KV_CTX
                k = tl.load(K + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :], mask=mask[:, None])
                acc += tl.sum(k, 0)
            tl.store(Out + offs_d, acc)
        """
        jit_fn = _FakeJIT(attn_src)
        bench_invocations = []

        tuner = lib.LibTuner.get("default")(
            jit_fn,
            jit_fn.arg_names,
            [Config({"BLOCK_N": 32}), Config({"BLOCK_N": 64})],
            ["KV_CTX", "HEAD_DIM"],
            reset_to_zero=None,
            restore_value=None,
            do_bench=lambda fn_call, quantiles=None: (bench_invocations.append(1) or [0.05, 0.04, 0.06]),
        )
        tuner.cache = {}  # isolate test from persistent disk DB
        entry = _build_synthetic_libentry(tuner, jit_fn, cache_binding=True)

        first_run_count = [0]
        orig_first_run = entry._first_run

        def _counted_first_run(*a, **kw):
            first_run_count[0] += 1
            return orig_first_run(*a, **kw)

        entry._first_run = _counted_first_run
        q_t, k_t, o_t = _make_tensor(), _make_tensor(), _make_tensor()

        with triton.knobs.autotuning.scope():
            triton.knobs.autotuning.shape_bucketing = True
            triton.knobs.autotuning.shape_bucketing_mode = "inference"
            triton.knobs.autotuning.adjust_block_size = False

            # Simulate 88 consecutive Decode tokens: KV_CTX = 513 .. 600 (all inside bucket (512, 768])
            for kv_ctx in range(513, 601):
                entry.run(
                    q_t,
                    k_t,
                    o_t,
                    1,
                    kv_ctx,
                    HEAD_DIM=128,
                    grid=lambda m: (1, (m["KV_CTX"] + m["BLOCK_N"] - 1) // m["BLOCK_N"]),
                )

            # Only 2 specializations exist in (512, 768]: unaligned (513) and 16-aligned (528)!
            self.assertEqual(first_run_count[0], 2)
            # BenchmarkCache key must remain exact raw shape (not bucketed)!
            bkey_513 = tuner.get_benchmark_key({"Q": q_t, "KV_CTX": 513, "HEAD_DIM": 128})
            bkey_514 = tuner.get_benchmark_key({"Q": q_t, "KV_CTX": 514, "HEAD_DIM": 128})
            self.assertNotEqual(bkey_513, bkey_514)

            # Verify last launch passed exact unpadded KV_CTX = 600 and computed grid for 600!
            last_kernel, _ = (
                entry._cpu_cache
                if lib.torch_device_fn.current_device() == "cpu"
                else entry.kernel_cache[0]
            )[next(iter(entry._cpu_cache or entry.kernel_cache[0]))][:2]
            total_launches = sum(len(rk.launches) for rk, _, _ in jit_fn.compiled_kernels)
            # 2 first_runs + 86 warm cached launches = 88 total launches
            self.assertEqual(total_launches, 88)
            # Find the launch corresponding to KV_CTX=600
            seen_kv_args = [
                call_args[4]
                for rk, _, _ in jit_fn.compiled_kernels
                for _, call_args in rk.launches
            ]
            self.assertIn(600, seen_kv_args)
            self.assertIn(513, seen_kv_args)
            self.assertNotIn(768, seen_kv_args)  # NEVER padded to bucket ceiling 768!

    def test_pointwise_pure_jit_and_heuristics_boundary_isolation(self):
        pw_src = """
        def pointwise_silu_kern(X, Y, n_elements, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(X + offs, mask=offs < n_elements)
            tl.store(Y + offs, x, mask=offs < n_elements)
        """
        jit_pw = _FakeJIT(pw_src)
        e_pw = _build_synthetic_libentry(jit_pw, jit_pw, cache_binding=True)
        x_t, y_t = _make_tensor(), _make_tensor()

        with triton.knobs.autotuning.scope():
            triton.knobs.autotuning.shape_bucketing = True
            triton.knobs.autotuning.adjust_block_size = False
            # Pure JIT (no autotuner): unaligned sizes across different powers of 2 (101, 503, 4099)
            # all have (is_one=False, is_aligned=False, width="i32") -> 1 single compiled kernel!
            for n_el in (101, 503, 1999, 4099):
                e_pw.run(x_t, y_t, n_el, BLOCK=256, grid=lambda m: ((m["n_elements"] + 255) // 256,))
            self.assertEqual(len(jit_pw.compiled_kernels), 1)

            # Now wrap with @triton.heuristics whose BLOCK depends on N:
            jit_h = _FakeJIT(pw_src)
            heur = triton.runtime.Heuristics(
                jit_h,
                jit_h.arg_names,
                {"BLOCK": lambda a: 64 if a["n_elements"] <= 64 else 128},
            )
            e_h = _build_synthetic_libentry(heur, jit_h, cache_binding=True)
            e_h.run(x_t, y_t, 40, grid=lambda m: (1,))
            e_h.run(x_t, y_t, 56, grid=lambda m: (1,))  # shares BLOCK=64
            e_h.run(x_t, y_t, 72, grid=lambda m: (1,))  # switches to BLOCK=128!
            e_h.run(x_t, y_t, 88, grid=lambda m: (1,))  # shares BLOCK=128
            self.assertEqual(len(jit_h.compiled_kernels), 2)

    def test_libtuner_guard_mismatch_and_training_dw_strides(self):
        mm_src = """
        def mm_dw_kern(A, B, C, M, N, K, stride_am, stride_bk, BLOCK_K: tl.constexpr):
            rk = tl.arange(0, BLOCK_K)
            a = tl.load(A + rk * stride_am, mask=rk < K)
            b = tl.load(B + rk * stride_bk, mask=rk < K)
            tl.store(C + rk, a + b, mask=rk < K)
        """
        jit_mm = _FakeJIT(mm_src)

        def prune_fn(configs, named_args, **kwargs):
            k_val = named_args.get("K", kwargs.get("K", 0))
            if k_val == 1440:
                return [c for c in configs if c.kwargs["BLOCK_K"] == 32]
            return [c for c in configs if c.kwargs["BLOCK_K"] == 64]

        tuner = lib.LibTuner.get("default")(
            jit_mm,
            jit_mm.arg_names,
            [Config({"BLOCK_K": 32}), Config({"BLOCK_K": 64})],
            ["M", "N", "K", "stride_am", "stride_bk"],
            reset_to_zero=None,
            restore_value=None,
            prune_configs_by={"early_config_prune": prune_fn},
            do_bench=lambda fn_call, quantiles=None: [0.05, 0.04, 0.06],
            strategy=["align32"] * 5,
        )
        tuner.cache = {}
        a_t, b_t, c_t = _make_tensor(), _make_tensor(), _make_tensor()

        with triton.knobs.autotuning.scope():
            triton.knobs.autotuning.shape_bucketing = True
            triton.knobs.autotuning.shape_bucketing_mode = "training"
            triton.knobs.autotuning.adjust_block_size = False

            # K=1088 and K=1280 with contiguous stride_am == K hit the EXACT same bucket!
            tuner.run(a_t, b_t, c_t, 4096, 4096, 1088, 1088, 4096, grid=(1,))
            self.assertEqual(tuner.last_dispatch_result.status.value, "COLD_TUNED")
            tuner.run(a_t, b_t, c_t, 4096, 4096, 1280, 1280, 4096, grid=(1,))
            self.assertEqual(tuner.last_dispatch_result.status.value, "HIT")
            self.assertEqual(tuner.best_config.kwargs["BLOCK_K"], 64)

            # K=1440 triggers early_config_prune GUARD_MISMATCH and safely selects BLOCK_K=32
            tuner.run(a_t, b_t, c_t, 4096, 4096, 1440, 1440, 4096, grid=(1,))
            self.assertEqual(tuner.last_dispatch_result.status.value, "GUARD_MISMATCH")
            self.assertEqual(tuner.best_config.kwargs["BLOCK_K"], 32)

    def test_xpu_static_grid_baked_key_separation_vs_callable_grid_reuse(self):
        row_src = """
        def row_scale_kern(X, Y, M, N, BLOCK_N: tl.constexpr):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK_N)
            v = tl.load(X + row * N + cols, mask=cols < N)
            tl.store(Y + row * N + cols, v, mask=cols < N)
        """
        x_t, y_t = _make_tensor(), _make_tensor()
        prev_backend = os.environ.get("FLAGTREE_BACKEND")
        os.environ["FLAGTREE_BACKEND"] = "xpu"
        try:
            with triton.knobs.autotuning.scope():
                triton.knobs.autotuning.shape_bucketing = True
                triton.knobs.autotuning.adjust_block_size = False

                for cache_binding in (True, False):
                    # Static tuple grids are baked into xpubin on XPU -> distinct cache keys
                    jit_static = _FakeJIT(row_src)
                    e_static = _build_synthetic_libentry(
                        jit_static, jit_static, cache_binding=cache_binding
                    )
                    e_static.run(x_t, y_t, 277, 128, BLOCK_N=128, grid=(277, 1, 1))
                    e_static.run(x_t, y_t, 277, 128, BLOCK_N=128, grid=(277, 1, 1))
                    e_static.run(x_t, y_t, 291, 128, BLOCK_N=128, grid=(291, 1, 1))
                    self.assertEqual(len(jit_static.compiled_kernels), 2)

                    # Callable grids remain dynamic at launch on XPU -> bucket reuse across 277 & 291
                    jit_dyn = _FakeJIT(row_src)
                    e_dyn = _build_synthetic_libentry(
                        jit_dyn, jit_dyn, cache_binding=cache_binding
                    )
                    e_dyn.run(x_t, y_t, 277, 128, BLOCK_N=128, grid=lambda m: (m["M"], 1, 1))
                    e_dyn.run(x_t, y_t, 291, 128, BLOCK_N=128, grid=lambda m: (m["M"], 1, 1))
                    self.assertEqual(len(jit_dyn.compiled_kernels), 1)
        finally:
            if prev_backend is None:
                os.environ.pop("FLAGTREE_BACKEND", None)
            else:
                os.environ["FLAGTREE_BACKEND"] = prev_backend


if __name__ == "__main__":
    unittest.main()
