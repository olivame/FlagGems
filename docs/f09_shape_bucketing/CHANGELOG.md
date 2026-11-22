# FlagGems F09 Changelog (§9.3)

All notable F09 dynamic shape-bucketing changes on top of upstream FlagGems `master` (`4d3d9f70`).

## [v0.2.0-f09] — 2026-11-22

### Documentation & Release Notes (`docs/f09-v020-release-notes`)
- Updated `CHANGELOG.md` and `DESIGN_AND_KNOWN_ISSUES.md` for immutable release **`v0.2.0-f09`** (0 Python/Shell runtime files modified).
- Marked **KI-FG-01** (KI-01 Jina `aiu_load_mask`), **KI-FG-03** (`ops/mul.py`, `fused/` scanner coverage, `_thead` `wrapper_divisibility_kw`), and **KI-FG-04** (`LibEntry` L0 thread-safety & `observe_args`/`all_args`) as **RESOLVED in `v0.2.0-f09`** with exact real-model PPU-ZW810E numbers (`USE_FLAGTUNE_COST_MODEL=0`), and cross-referenced FlagTree **KI-07** (`IntervalPolicy`/GraphBucket) & **KI-08** (`cache.py` flock) resolution.

### FG-P1-B (`a4ee460e429ebee75870d03e49b38578acea94d1`) — L0 Thread-Safety, `thead` Dehardcoding & `fused/` Compliance
- **`LibEntry` L0 Thread-Safety & Policy Forwarding (`src/flag_gems/utils/libentry.py`)**:
  - Added `_bkt_l0_lock` (`_get_bkt_l0_lock()` / `_promote_bkt_l0_hit()`) guarding `_store_bkt_l0_entry` and Tier-0.5 -> Tier-0 promotion/eviction while keeping Tier-0 hot reads **100% lock-free** (12-thread × 400-write stress `cap=4`: BEFORE **24 `RuntimeError` + 2 cap overshoots** -> AFTER **0 errors**).
  - Updated `_build_bkt_l0_bucket_key()` to call `policy.observe_args(all_named)` and forward `all_args=all_named` to `policy.normalize_entry_scalar()`, eliminating cross-group collisions under multi-group `IntervalPolicy`.
  - Replaced hardcoded `device.vendor_name == "thead"` with `_supports_cudagraph_bench(vendor_name)` querying FlagTree `BackendBucketingCapabilities.supports_cudagraph_bench` (with `getattr(..., True)` fallback).
  - Introduced read-only `_ModuleCompatProxy` eliminating import-time monkey-patching of `flag_gems.runtime` and `cost_model`.
- **Static Compliance Scanner & Residual Branch Cleanup (`tools/check_bucketing_compliance.py`, `ops/mul.py`, `fused/fused_moe.py`)**:
  - Extended `tools/check_bucketing_compliance.py` to scan `src/flag_gems/fused/` (87 files across `DSA/`, `FLA/`, `mhc/`).
  - Removed op-level `shape_bucketing` branches from `src/flag_gems/ops/mul.py` and `src/flag_gems/fused/fused_moe.py`: **`generic_ops`, `_thead`, and `fused` now all report `op_level_bucketing_branch == 0`**.
- **Tests**: Added 3 regression tests in `tests/test_libentry_shape_bucketing.py` (**13/13 PASS**; combined PPU regression **25/25 PASS**).

### FG-P1-A (`13590357eb0ebf59bffdc6c2c94d8b93c3855b14`) — Jina `bmm` `aiu_load_mask` Decoupling & `_thead` Divisibility Cleanup
- **Jina `bmm` Zero-Recompile Fix (`_thead/ops/bmm.py`, `configs_loader.py`, `_thead/mm_ppu_expand.yaml`)**:
  - Normalized all 4 PPU `bmm` kernels (`bmm_kernel_ppu`, `bmm_narrow_n_kernel_ppu`, `bmm_small_m_kernel_ppu`, `bmm_split_k_kernel_ppu`) to alignment-agnostic `_LOAD_REGULAR` and decoupled `aiu_load_mask` from `@libtuner(key=...)` and kernel signatures (0 `op_level_bucketing_branch`).
- **`_thead` Wrapper Divisibility Cleanup (`_thead/ops/{bmm,mm,quantized_lstm}.py`)**:
  - Moved `EVEN_K`/`EVEN_N`/`EVEN_M`/`EVEN_KI`/`EVEN_KR`/`EVEN_H` into `@triton.heuristics` and removed constant `False` `EVEN_MN`/`EVEN_N` kwargs, reducing `_thead` `wrapper_divisibility_kw` from **3 -> 0**.
- **Real-Model Verification on PPU-8/9 (`USE_FLAGTUNE_COST_MODEL=0`, paired with FlagTree `e4522a5fc1`)**:
  - `jina_reranker_v2_base`: R0 `COLD_TUNED 4 -> 0`, `post_warm_new_hgbin +62 -> 0`, `hit_rate 99.70% -> 100.00%`, `post_warm_first 42,961.15 ms -> 428.34 ms` (`100.3×`), `max_step 41,598.52 ms -> 40.26 ms` (`stall_reduction 1.41× -> 2,019.79×`), steady `1.092×` (`cos_vs_aten=0.999968`); R2-B Zipf P99 `24,706.07 ms -> 4,890.91 ms` (orig 4-len `5.05×`) & `-> 42.50 ms` (rec 7-len `581.4×`, `cold_tuned=0`), restart `new_hgbin=0` (`9.14×`).
  - `jina_embeddings_v3`: R0 `COLD_TUNED 4 -> 0`, `post_warm_new_hgbin +62 -> 0`, `hit_rate 99.89% -> 100.00%`, `post_warm_first 45,220.22 ms -> 1,483.83 ms` (`30.5×`, `1.068×` beats OFF), `max_step 42,877.15 ms -> 128.80 ms` (`stall_reduction 1.74× -> 747.39×`), steady `1.119×` (`cos_vs_aten=0.999314`, `cos_vs_off=0.999845`); R2-B Zipf P99 `25,684.37 ms -> 4,035.08 ms` (orig 4-len `6.37×`) & `-> 133.99 ms` (rec 7-len `191.7×`, `cold_tuned=0`), restart `new_hgbin=0` (`8.14×`).
  - **Milestone**: **18/18 (100%)** real models proven post-warmup `COLD_TUNED = 0` & cross-process restart `new_hgbin = 0`.
- **Tests**: Added `test_jina_bmm_aiu_load_mask_zero_post_warmup_recompile` in `tests/test_training_ops_ppu.py` (**12/12 PASS**).

## [v0.1.0-f09] — 2026-11-22

### Docs, Installer Safeguard & Portable Acceptance Runner (`docs/f09-docs-and-benchmark-runner`)
- Added bilingual CN + EN F09 Quick Start & usage guide (`docs/f09_shape_bucketing/README.md`).
- Added F09 architecture, maintenance/upgrade guide, and 5-field Known Issues register (`docs/f09_shape_bucketing/DESIGN_AND_KNOWN_ISSUES.md`).
- Added F09 release changelog (`docs/f09_shape_bucketing/CHANGELOG.md`) and minimal F09 pointer section in root `README.md`.
- Added hardcoded-path-free, self-contained acceptance runner (`benchmarks/f09_acceptance_runner.py`) supporting `--model-dir` and `--synthetic-fallback` across `infer`, `train`, and `decode` modes with isolated per-variant subprocesses and `USE_FLAGTUNE_COST_MODEL=0` default.
- Hardened `tools/install_f09_editable.sh` to refuse destructive in-place `site-packages/triton/` overwrites unless `--allow-in-place-triton-overlay` is explicitly passed (recommending isolated `--target <dir>`).

### Fix-2 (`3d15211c908915daf5fcb2a02a6fb97946eb3264`) — P0-1 Repro-Driven Bugfixes
- Fixed live `NameError: name 'bucketing' is not defined` at `_thead/ops/_scaled_dot_product_fused_attention_overrideable.py:365` and converted `_sdpa_fwd_c` `L`/`S` parameters from `tl.constexpr` to runtime arguments (zero per-shape recompilation, zero `op_level_bucketing_branch`, `_thead` `shape_stride_constexpr` reduced 19 → 18).
- Fixed stale `SPLIT_K` read in `_thead/ops/bmm.py:_run_ppu_bmm_split_k` to read `launch_meta["SPLIT_K"]` returned by `LibEntry` (matching `mm.py`, eliminating silent half-K reduction corruption on warm cache hits).
- Fixed `DistributedDispatchGuard` L0/L1 hot-cache bypass deadlock in `src/flag_gems/utils/libentry.py` by coordinating L0 hit, L1 hit, and L1 miss via FlagTree Fix-3 re-entrant `dispatch_guard_scope()` / `coordinate()`, keeping `distributed_dispatch_guard=False` completely lock-free and zero-collective.
- Resolved all 11 `>120`-char line-length lint violations across F09 changed files and expanded `tests/test_libentry_shape_bucketing.py` (10/10) + `tests/test_training_ops_ppu.py` (11/11).

### Fix-1 (`5f64176869f982d233d9e5aa5f34f9adbe0c630e`) — Non-Contiguous Linear & Portable Installer
- Updated `_thead/ops/linear.py` to use `reshape(-1, K)` / `reshape(*batch_dims, N)` for non-contiguous inputs (e.g. `nn.TransformerEncoderLayer`).
- Removed op-level `shape_bucketing` branch from `_thead` SDPA overrideable op (`_thead` `op_level_bucketing_branch` → `0`).
- Added portable backend-agnostic `tools/install_f09_editable.sh`.

### PR-3 (`50f79dd5e3b43b6df90e26c3b2d7f023b5ed80a6`) — Static Compliance Scanner
- Added `tools/check_bucketing_compliance.py` static scanner and CI gate (`--root`, `--json`, `--strict-paths`).

### PR-2 (`957111c0e002434b6ed73cf218eb099eef879ea3`) — Zero-Recompile Ops & Training Parity
- Updated 42 generic and `_thead` operators for dynamic-shape zero-recompilation and backward correctness (`layernorm`, `SDPA` backward BHSD/GQA/dropout, `nll_loss`, `mse_loss_backward`, `linear_backward`, `embedding`, `_fused_adam`, `select_backward`, `pad`, `mm`, `bmm`, etc.).
- Added `tests/test_training_ops_ppu.py` live PPU operator regression suite.

### PR-1 (`66a178de3f927ce76fc9ed10c54996c926ed4ec9`) — LibEntry Shape Bucketing & Dual-Path Pointwise
- Integrated FlagTree `OnlineBucketPolicy`, `backend_caps`, `BucketCacheBudget`, and `DistributedDispatchGuard` into `LibEntry` / `LibTuner`.
- Added 2-tier L0 cache (`_bkt_l0_cache` + `_bkt_l0_bucket_cache`) and dual-path `pointwise_dynamic` codegen.
- Added `tests/test_libentry_shape_bucketing.py`.

### Upstream Baseline
- `4d3d9f70` (`fix(kunlunxin): add sparse_bsc_tensor_unsafe and sparse_compressed_tensor overrides (#8844)`).
