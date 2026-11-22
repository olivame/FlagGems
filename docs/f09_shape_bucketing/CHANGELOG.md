# FlagGems F09 Changelog (§9.3)

All notable F09 dynamic shape-bucketing changes on top of upstream FlagGems `master` (`4d3d9f70`).

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
