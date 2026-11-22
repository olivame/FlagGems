# FlagGems F09: Universal & Portable Dynamic Shape Bucketing (中英文快速上手指南)

[中文说明](#中文说明) | [English Quick Start](#english-quick-start)

---

## 中文说明

### 1. 项目概述与定位
FlagGems F09 面向动态序列长度与变长批量负载（大模型推理、微调训练与增量 Decode），在 FlagGems 算子调度层（`LibEntry` / `LibTuner` / `pointwise_dynamic`）深度集成 FlagTree 动态 Shape 分桶运行时，解决 Triton 算子在动态 shape 下的反复 JIT 重编译与冷启动抖动问题：
- **两级 L0 热缓存**：Tier 0 精确 shape 快速命中 + Tier 0.5 分桶级命中自动提升，兼顾首轮零重编译与稳态低开销。
- **双路径 `pointwise_dynamic` 代码生成**：在开启分桶时避免因 shape 变化触发冗余核特化。
- **分布式调度守卫（`DistributedDispatchGuard`）协同**：在多卡并行开启 `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1` 时，`LibEntry` 外层覆盖 L0/L1 命中与未命中并抑制内层重复协商；默认关闭时保持零锁、零集合快路径。
- **全仓静态合规门禁**：提供 `tools/check_bucketing_compliance.py` 自动化检测动态维度 `tl.constexpr`、包装层整除特化、动态 `next_power_of_2` 与算子内 `shape_bucketing` 分支。
- **去硬编码便携评测入口**：提供 `benchmarks/f09_acceptance_runner.py`，支持本地 HuggingFace 权重（`--model-dir`）与零外部依赖合成模型（`--synthetic-fallback`）。

### 2. 支持环境与硬件验证边界
- **依赖软件栈**：Python 3.10+（实测 3.12.3）、PyTorch 2.x（实测 2.10.0）、配套 FlagTree F09 运行时（基于 Triton 3.x，实测 3.6.0）。
- **已验证硬件后端**：T-Head Zhenwu PPU-ZW810E（`_thead` 后端，单卡推/训/Decode 与多卡 `gloo` 回归已实机验证）。
- **跨后端可移植边界**：核心调度层（`libentry.py`、`pointwise_dynamic.py`、`tools/`、`benchmarks/`）遵循后端能力契约，未绑定特定芯片；NVIDIA CUDA、AMD ROCm、Ascend NPU 等其余硬件后端尚待各自实机验收。

### 3. 安装方式（推荐隔离 `--target` 安装）
推荐使用非破坏性 `--target` 模式隔离安装，不修改系统 `site-packages/triton/` 原生 wheel：

```bash
bash tools/install_f09_editable.sh \
    --flagtree /path/to/FlagTree \
    --flaggems /path/to/FlagGems \
    --target /tmp/f09_target

export PYTHONPATH="/tmp/f09_target:${PYTHONPATH:-}"
```

> **安全保护说明**：若省略 `--target`，安装脚本默认拒绝直接覆盖 `site-packages/triton/`；仅在一次性容器或明确需要就地覆盖时，附加 `--allow-in-place-triton-overlay` 参数确认。

### 4. 核心环境变量

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FLAGTREE_SHAPE_BUCKETING` | `0` | 设为 `1` 开启 FlagTree + FlagGems 动态 shape 分桶 |
| `FLAGTREE_SHAPE_BUCKETING_MODE` | `inference` | `inference`（低延迟在线分桶）或 `training`（保护前后向步幅与梯度算子） |
| `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD` | `0` | 多卡并行时设为 `1` 开启跨 Rank 编译协商守卫（默认 `0` 零集合开销） |
| `USE_FLAGTUNE_COST_MODEL` | 未设置 | **离线/受限网络环境强烈建议设为 `0`**，避免 FlagTune 远程拉取 cost-model 压缩包超时 |

### 5. 最小可复现示例与预期输出

```bash
USE_FLAGTUNE_COST_MODEL=0 \
FLAGTREE_SHAPE_BUCKETING=1 \
FLAGTREE_SHAPE_BUCKETING_MODE=inference \
PYTHONPATH=/tmp/f09_target \
python3 - <<'PY'
import torch, triton, flag_gems

assert triton.knobs.autotuning.shape_bucketing is True
with flag_gems.use_gems():
    for seq_len in (64, 92, 128, 188, 250):
        x = torch.randn(2, seq_len, 256, device="cuda", dtype=torch.bfloat16)
        y = torch.nn.functional.layer_norm(x, (256,))
        assert torch.isfinite(y).all()
print("F09_QUICKSTART_OK")
PY
```

预期输出：`F09_QUICKSTART_OK`（进程返回码 `0`）。

### 6. 静态合规扫描、便携评测与测试命令

```bash
# 1) 全仓静态分桶合规扫描（预期 backend:_thead files_scanned=99, op_level_bucketing_branch=0）
python3 tools/check_bucketing_compliance.py --root . --json /tmp/compliance_report.json

# 2) 零外部模型独立验收烟测（infer / train / decode 三种模式）
USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_target \
python3 benchmarks/f09_acceptance_runner.py \
    --synthetic-fallback --mode infer --out /tmp/f09_smoke.json

# 3) 使用本地 HuggingFace 模型运行验收基准
USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_target \
python3 benchmarks/f09_acceptance_runner.py \
    --model-dir /path/to/local_hf_model --mode infer --out /tmp/f09_hf.json

# 4) 单元测试与 PPU 实机训练算子测试
USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_target \
pytest -v -o pythonpath=/tmp/f09_target \
    tests/test_libentry_shape_bucketing.py tests/test_training_ops_ppu.py
```

### 7. 常见问题排查（Troubleshooting）
- **首步或稳态偶发 60s–200s 卡顿**：未设置 `USE_FLAGTUNE_COST_MODEL=0` 导致 FlagTune 在离线环境反复尝试下载远程 cost-model 压缩包；设置 `export USE_FLAGTUNE_COST_MODEL=0` 即可彻底消除。
- **安装脚本提示 `REFUSED: In-place installation would overwrite...`**：为防止破坏系统 `triton` wheel，请改用 `--target /tmp/f09_target` 并通过 `PYTHONPATH` 启用；确需就地安装再追加 `--allow-in-place-triton-overlay`。
- **多卡开启 `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1` 最佳实践**：在 `torch.distributed.init_process_group()` 后由全部 Rank 共同调用 `triton.runtime.init_default_dispatch_guard()` 建立独立 `gloo` 侧通道，并确保所有 Rank 执行一致的受保护算子序列。

---

## English Quick Start

### 1. Overview
FlagGems F09 integrates the FlagTree universal dynamic shape-bucketing runtime into FlagGems (`LibEntry`, `LibTuner`, `pointwise_dynamic`, and key inference/training operators) to eliminate per-shape Triton JIT recompilation storms across variable sequence lengths while preserving numerical parity and steady-state performance.

Key components in this repository:
- `src/flag_gems/utils/libentry.py`: 2-tier L0 cache (exact Tier 0 + bucket Tier 0.5), `BucketCacheBudget`, and re-entrant `DistributedDispatchGuard` coordination across L0/L1 hits and cold misses.
- `src/flag_gems/utils/pointwise_dynamic.py`: Dual-path bucket-aware codegen.
- `tools/check_bucketing_compliance.py`: Static compliance scanner and CI gate (`--strict-paths`).
- `tools/install_f09_editable.sh`: Portable isolated `--target` installer with non-destructive site-packages safeguard.
- `benchmarks/f09_acceptance_runner.py`: Standalone, hardcoded-path-free acceptance runner supporting `--model-dir` and `--synthetic-fallback` across `infer`, `train`, and `decode` modes.

### 2. Installation (Isolated `--target`, Recommended)

```bash
bash tools/install_f09_editable.sh \
    --flagtree /path/to/FlagTree \
    --flaggems /path/to/FlagGems \
    --target /tmp/f09_target

export PYTHONPATH="/tmp/f09_target:${PYTHONPATH:-}"
export USE_FLAGTUNE_COST_MODEL=0
```

> **Note**: Omitting `--target` refuses in-place overwrites of `site-packages/triton/` unless `--allow-in-place-triton-overlay` is explicitly specified.

### 3. Quick Verification Commands

```bash
# Static compliance scan
python3 tools/check_bucketing_compliance.py --root . --json /tmp/compliance_report.json

# Zero-weight synthetic smoke benchmark (isolated ATen / FlagGems OFF / FlagGems ON subprocesses)
USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_target \
python3 benchmarks/f09_acceptance_runner.py \
    --synthetic-fallback --mode infer --out /tmp/f09_smoke.json

# Unit & training operator test suites
USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_target \
pytest -v -o pythonpath=/tmp/f09_target \
    tests/test_libentry_shape_bucketing.py tests/test_training_ops_ppu.py
```

Expected smoke output includes `PASS: F09 acceptance smoke check succeeded.` with `min_cos(ON,ATen) >= 0.999` and `finite=True`.

### 4. Further Reading
- Architecture, upgrade guide, and full 5-field Known Issues register: [`DESIGN_AND_KNOWN_ISSUES.md`](./DESIGN_AND_KNOWN_ISSUES.md)
- Complete F09 commit & release changelog: [`CHANGELOG.md`](./CHANGELOG.md)
