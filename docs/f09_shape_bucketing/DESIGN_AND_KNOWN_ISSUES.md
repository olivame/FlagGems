# FlagGems F09 架构设计、维护升级指南与已知问题清单 (§9.2)

## 1. 总体架构与模块职责

FlagGems F09 在不改变上游 PyTorch ATen 注册接口的前提下，将动态 shape 分桶能力下沉至通用调度层，使单算子无需编写面向特定硬件或分桶开关的条件分支：

```text
PyTorch Op Call (e.g. F.linear / F.layer_norm / SDPA / bmm)
  └─ FlagGems Op Wrapper (runtime dims; no shape_bucketing branches in _thead)
       └─ LibEntry.run()  [src/flag_gems/utils/libentry.py]
            ├─ guard_enabled=False (default): lock-free, zero-collective fast path
            │    ├─ Tier 0 Exact L0 Cache (_bkt_l0_cache)
            │    ├─ Tier 0.5 Bucket L0 Cache (_bkt_l0_bucket_cache -> promote to Tier 0)
            │    └─ L1 Kernel Cache (kernel_cache[device] + BucketCacheBudget LRU)
            └─ guard_enabled=True (FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1):
                 ├─ Outer _coordinate_dispatch_guard() wraps L0 hit, L1 hit, and L1 miss
                 ├─ Skips _run_cached bypass so all ranks negotiate symmetrically
                 └─ Runs _first_run() inside FlagTree dispatch_guard_scope()
                      └─ LibTuner.run() / Autotuner.run() (skips duplicate inner negotiate)
                           └─ FlagTree OnlineBucketPolicy / IntervalPolicy + JITFunction
```

### 1.1 `LibEntry` 两级 L0 热缓存设计
- **Tier 0（`_bkt_l0_cache`）**：按 `(device, bucketing_mode, policy_path, arg_count, exact_scalars, tensor_dtype_and_align)` 索引，重复 shape 直接构造 launch 参数并调用 `CompiledKernel`，开销接近原生缓存绑定。
- **Tier 0.5（`_bkt_l0_bucket_cache`）**：在 Tier 0 未命中且算子无后置 heuristics 时，先调用 `policy.observe_args(all_named)` 并向 `policy.normalize_entry_scalar(..., all_args=all_named)` 透传完整上下文生成分桶级键；命中后在 `_bkt_l0_lock` 保护下安全提升写入 Tier 0（Tier 0 热读保持 **100% 无锁**），实现**未见变长首轮零重编译、次轮零分桶查表开销且多线程并发安全**；同时 `_supports_cudagraph_bench()` 统一查询 FlagTree `BackendBucketingCapabilities.supports_cudagraph_bench`（零厂商硬编码）。
- **L1 缓存与预算控制**：`kernel_cache[device]` 结合 FlagTree `BucketCacheBudget` 按 `max_bucket_artifacts` 淘汰，防止长尾变长负载引起显存/主机缓存无限增长。
- **返回元数据一致性**：`LibEntry.run()` 所有返回路径（`_run_cached`、L0 命中、L1 命中、L1 未命中）统一返回 `(kernel, constexprs)`；依赖调优参数（如 `SPLIT_K`）的算子（`mm.py`、`bmm.py`）必须读取本次调用返回的 `launch_meta["SPLIT_K"]`，禁止读取易被其他 shape 覆盖的 `tuner.best_config`。

### 1.2 `DistributedDispatchGuard` 协同机制
- 解决多卡分布式并行下“部分 Rank 命中 L0/L1 热缓存而另一 Rank 冷编译进入 `negotiate()`”导致的集合通信死锁。
- `LibEntry._resolve_dispatch_guard()` 按优先级解析：`entry.dispatch_guard` → 包装链路内 `tuner.dispatch_guard` → FlagTree 非集合 `get_default_dispatch_guard()`。
- `_coordinate_dispatch_guard()` 复用 FlagTree Fix-3 的可重入 `dispatch_guard_scope()` / `guard.coordinate(local_state)`：
  - 外层 `LibEntry.run()` 仅发起一次协商并在 `finally` 中执行 `post_compile_sync()`；
  - 内层 `LibTuner.run()` 与 `Autotuner.run()` 检测到 `is_in_dispatch_guard_scope() == True` 时自动跳过重复集合；
  - 独立直接调用 `LibTuner.run()`（未经 `LibEntry`）仍正常参与协商；
  - 默认 `distributed_dispatch_guard=False` 时不创建上下文管理器、不执行任何锁或集合通信。

### 1.3 `pointwise_dynamic` 双路径代码生成
- `src/flag_gems/utils/pointwise_dynamic.py` 在 `shape_bucketing` 开启时采用面向动态 shape 的代码生成路径，避免将变长元素数或步幅固化为编译期常量；关闭分桶时完整保持上游原始代码生成行为。

### 1.4 静态合规扫描门禁（`tools/check_bucketing_compliance.py`）
扫描 `src/flag_gems/ops/*.py`、`src/flag_gems/fused/**/*.py`（87 文件）与 `src/flag_gems/runtime/backend/_*/*.py`，检测四类破坏动态分桶的反模式（`v0.2.0-f09` 已实现 `generic_ops`、`_thead`、`fused` 三域 `op_level_bucketing_branch == 0` 且 `_thead` `wrapper_divisibility_kw == 0`）：
1. `shape_stride_constexpr`：动态维度/步幅标注 `tl.constexpr`；
2. `wrapper_divisibility_kw`：包装函数内计算 `EVEN_*` / `ALIGNED_*` / `DIVISIBLE_*` 关键字；
3. `wrapper_next_pow2_block`：包装函数内计算 `BLOCK_* = triton.next_power_of_2(...)`；
4. `op_level_bucketing_branch`：算子内部硬编码 `shape_bucketing` 分支。

支持 `--strict-paths <rel_paths...>` 作为 CI 阻断门禁，确保已收敛算子不发生回归。

---

## 2. 上游同步与补丁维护指南

当上游 `FlagGems` `master` 推进并需要迁移 F09 补丁栈时，建议按以下顺序操作：
1. **按顺序变基/合入补丁栈**：从基线 `4d3d9f70` 起依次应用 PR-1（`libentry.py` + `pointwise_dynamic.py`）、PR-2（算子零重编译与反向修复）、PR-3（合规扫描器）、Fix-1、Fix-2、Docs/Runner (`v0.1.0-f09`)、FG-P1-A (`13590357eb`)、FG-P1-B (`a4ee460e42`)、`v0.2.0-f09` Docs (`ab85d9f5a2`) 与 `v0.3.0-f09` Installer/Docs 同步提交（共 10 个有序提交）。
2. **重点冲突检查点**：
   - `src/flag_gems/utils/libentry.py`：保留 `_coordinate_dispatch_guard`、2-tier L0 缓存以及 `LibEntry.run()` 统一返回 `(kernel, constexprs)` 契约；
   - `_thead/ops/_scaled_dot_product_fused_attention_overrideable.py`：确认 `_sdpa_fwd_c` 的 `L, S` 保持非 `tl.constexpr` 且无 `shape_bucketing` 分支；
   - `_thead/ops/bmm.py` 与 `_thead/ops/mm.py`：确认 Split-K 路径均从 `launch_meta["SPLIT_K"]` 读取分片数。
3. **升级后必跑验收三件套**：
   - 隔离安装：`bash tools/install_f09_editable.sh --flagtree <FT> --flaggems <FG> --target /tmp/f09_verify`
   - 静态合规门禁：`python3 tools/check_bucketing_compliance.py --root <FG> --json /tmp/comp.json`（确认 `_thead` `op_level_bucketing_branch == 0`）
   - 回归测试 + 合成烟测：`pytest -v tests/test_libentry_shape_bucketing.py tests/test_training_ops_ppu.py` 及 `benchmarks/f09_acceptance_runner.py --synthetic-fallback --mode infer`。

---

## 3. 已知问题清单（§9.2 五字段强制规范）

| 问题编号与简述 | 影响范围 | 复现方式 | 规避方法 | 责任人 | 整改计划 |
| --- | --- | --- | --- | --- | --- |
| **KI-FG-01** *(对应 FlagTree KI-01，**RESOLVED in `v0.2.0-f09`**)*：`_thead/ops/bmm.py` `aiu_load_mask` 对齐态翻转导致预热后额外调优 | 使用自定义 eager attention 下沉至 `bmm` 且序列长度跨越 `S % 16 == 0` 与 `S % 16 != 0` 的模型（如 Jina v2/v3） | 在 `v0.1.0-f09` 以非 16 对齐长度预热后输入 `S=288/576` 触发 `COLD_TUNED=4`、`+62 .hgbin` 与 `41.6–42.9s` 毛刺 | 升级至 `v0.2.0-f09` 即可根治；旧版可通过双对齐预热规避 | FlagGems `_thead` 算子维护者 | **已于 `v0.2.0-f09` 解决（`FG-P1-A` `13590357eb`）**：4 个 PPU `bmm` 核函数统一为 `_LOAD_REGULAR` 并从 `@libtuner(key=...)` 解耦 `aiu_load_mask`。PPU-8/9 真机实测（`USE_FLAGTUNE_COST_MODEL=0`）：`jina_reranker_v2_base` R0 `COLD_TUNED 4->0`、`new_hgbin +62->0`、`hit_rate 100%`、`post_warm_first 42,961.15->428.34ms` (`100.3×`)、`max_step 41,598.52->40.26ms` (`stall_reduction 2,019.79×`)、稳态 `1.092×` (`cos_vs_aten=0.999968`)、R2-B Zipf P99 `24,706.07->4,890.91ms` (orig `5.05×`) / `42.50ms` (rec-7wk `581.4×`, `cold_tuned=0`)、重启 `new_hgbin=0` (`9.14×`)；`jina_embeddings_v3` R0 `COLD_TUNED 4->0`、`new_hgbin +62->0`、`hit_rate 100%`、`post_warm_first 45,220.22->1,483.83ms` (`30.5×`, `1.068×` 超越 OFF)、`max_step 42,877.15->128.80ms` (`stall_reduction 747.39×`)、稳态 `1.119×` (`cos_vs_off=0.999845`)、R2-B Zipf P99 `25,684.37->4,035.08ms` (`6.37×`) / `133.99ms` (`191.7×`, `cold_tuned=0`)、重启 `new_hgbin=0` (`8.14×`)，实现 **18/18 (100%) 零重编译** |
| **KI-FG-02**：离线环境下 FlagTune 远程 Cost Model 拉取超时与 60s 静默重试 | 无外网或目标模型包不可达的容器环境中未显式关闭 cost model 的全部评测/训练任务 | 在离线环境省略 `USE_FLAGTUNE_COST_MODEL=0` 运行含 autotune 的模型，触发 `cost_model.py` 远程拉取卡顿 39–264s 并每 60s 静默重试 | 运行前统一设置 `export USE_FLAGTUNE_COST_MODEL=0`（`benchmarks/f09_acceptance_runner.py` 已默认强制设置） | FlagGems FlagTune 集成维护者 | P1 阶段在离线/失败后引入指数退避或快速熔断配置，并保持默认基准关闭远程拉取 |
| **KI-FG-03** *(**RESOLVED in `v0.2.0-f09`**)*：静态合规扫描遗留项（`generic_ops/mul.py`、`fused/` 盲区与 `_thead` 3 处 `wrapper_divisibility_kw`） | `ops/mul.py`、`fused/fused_moe.py` 以及 `_thead/ops/{bmm,mm,quantized_lstm}.py` | 在 `v0.1.0-f09` 执行 `check_bucketing_compliance.py --root .` 检出 `generic_ops` `op_level_bucketing_branch=1`、`_thead` `wrapper_divisibility_kw=3` 且跳过 `fused/` | 升级至 `v0.2.0-f09` 并通过 `--strict-paths` 维持零回归 | FlagGems F09 架构维护者 | **已于 `v0.2.0-f09` 解决（`FG-P1-A` `13590357eb` + `FG-P1-B` `a4ee460e42`）**：扫描器扩展覆盖 `src/flag_gems/fused/`（87 文件）；移除 `ops/mul.py` 与 `fused/fused_moe.py` 分桶分支；`_thead/{bmm,mm,quantized_lstm}.py` 整除参数全部下沉 `@triton.heuristics`，实测 `generic_ops`/`_thead`/`fused` 全部 `op_level_bucketing_branch == 0` 且 `_thead` `wrapper_divisibility_kw == 0` |
| **KI-FG-04** *(**RESOLVED in `v0.2.0-f09`**)*：`LibEntry` L0 缓存多线程并发访问与 `_build_bkt_l0_bucket_key` 未传 `all_args` | 单进程多线程并发调度同一算子或使用多组 `IntervalPolicy` 的动态轴场景 | 在 `v0.1.0-f09` 下 12 线程 × 400 写入 (`cap=4`) 触发 `24` 次 `RuntimeError` 与 `2` 次越界，且 `B=8` 错误命中 `B=4` 组缓存 | 升级至 `v0.2.0-f09` 即可根治 | FlagGems 运行时维护者 | **已于 `v0.2.0-f09` 解决（`FG-P1-B` `a4ee460e42`）**：新增 `_bkt_l0_lock` 保护写入/晋升/淘汰并保持 Tier-0 读 100% 无锁（12 线程并发压测 `24 RuntimeError -> 0`）；`_build_bkt_l0_bucket_key` 补齐 `policy.observe_args()` 与 `all_args` 透传；移除 `vendor_name == "thead"` 硬编码与导入期 Monkey-Patch（`test_libentry_shape_bucketing.py` **13/13 PASS**） |
| **KI-FG-05**：单芯片实机验证边界与 FlagGems 相对 ATen 的算子基线稳态差距 | 非 PPU 硬件后端（CUDA/ROCm/NPU）及小中型模型稳态吞吐绝对值（`v0.2.0-f09` 全量 18 模型透明拆解：FlagGems OFF vs ATen 固有差距 `1.96–8.86×`，F09 ON vs OFF 净稳态加速 **`1.051–1.160×`**，将 ON/ATen 收窄至 `1.86–8.33×`） | 在小模型对比 `aten_ref` 与 `flag_gems_on` 稳态耗时可见算子库基线差距；在非 PPU 设备上尚未完成全量实机回归 | 报告与验收指标严格拆解“FlagGems vs ATen 基线差距”与“F09 ON vs OFF 净收益”；跨芯片上线前先跑 `benchmarks/f09_acceptance_runner.py` 与单测 | F09 联合交付团队 | `v0.2.0-f09` 已完成全部 18 模型稳态归因拆解与去厂商硬编码（`FG-P1-B` + `FT-P1-B`）；待多芯片 CI 就绪后引入 NVIDIA CUDA/Ascend NPU 真机回归并持续优化小形状算子启动开销 |
| **KI-FG-06**：`DistributedDispatchGuard` 要求同通信组内各 Rank 算子调用序列对称 | 开启 `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1` 且存在跨 Rank 不对称控制流分支（如不同 Rank 跳过某算子）的模型 | Rank 0 执行受保护 `LibEntry` 算子而 Rank 1 走控制流分支跳过该算子，将引发集合通信失配超时 | 仅在各 Rank 算子调用序列对称的 DDP/TP/FSDP 路径开启守卫；不对称 MoE 路由请在更高层控制流对齐后再调用受保护算子 | FlagTree + FlagGems 分布式维护者 | P1 阶段支持按算子/通信组细粒度标注对称守卫作用域，并补充更明确的超时诊断日志 |

> **配套 FlagTree `v0.2.0-f09` 已闭环项联动说明**：
> - **FlagTree KI-07（`IntervalPolicy` Auto-PGO & `GraphBucketCompiler`/`DecodeGraphBucketRunner`）— RESOLVED in `v0.2.0-f09`** (`FT-P1-A` `9b92bd75dc`)：联合 FlagGems `a4ee460e42` 在 PPU-10/11 完成真实模型 Auto-PGO V2 闭环实证（`USE_FLAGTUNE_COST_MODEL=0`）— `bge_reranker_v2_m3`（`11,109` 真实样本、`4/4` 组、`7/7` `ft_pgo_v1` 命中）`COLD_TUNED 33->14 (-57.6%)`、冷启 `60,108.1->54,073.0 ms (1.11×)`、未见长度 `13,400.7->274.6 ms (48.80×, 0 new_hgbin, 0 cold_tuned)`、稳态 `1.115×`（`cos_vs_aten=0.999983`）；`qwen3_embedding_0_6b`（`43,659` 真实样本、`5/5` 组、`10/10` `ft_pgo_v1` 命中）6 档 `COLD_TUNED 65->38 (-41.5%)`、冷启 `104,355.2->91,892.7 ms (1.14×)`、未见长度 `2,582.3->1,667.9 ms (19.47×)`、稳态 `1.105×`（超越 ON `1.092×`）、`min_cos_pgo_vs_off=1.000000`，策略全覆盖 7 档 A/B `1,995.0 ms vs 1,994.6 ms`（`1.000×` 精确持平、`0` 调优）。
> - **FlagTree KI-08（多进程共享 `TRITON_CACHE_DIR` 磁盘淘汰竞态）— RESOLVED in `v0.2.0-f09`** (`FT-P1-B` `e4522a5fc1`)：新增 POSIX `fcntl.flock` + `threading.RLock` 混合锁与 100% 无锁热读，6 进程并发淘汰压测 `15 FileNotFoundError -> 0`。
> - **配套 FlagTree `v0.3.0-f09` (`386e29ea60`) Phase 3 多卡能力联动**：`F09-G04` (`025903bf12`) 落地 `comm_profiler.py`、通信校准可解释 `PartitionPlanner` 与 4× PPU 实跑 `ParallelPlanExecutor`（关闭单机多卡 KI-04）；`F09-G05` (`ca4d466c87`) 为 `DistributedDispatchGuard` 补齐 `comm_timeout_s` 有界超时熔断、对端故障隔离与 `CoordinatedStepGuard` 跨卡 OOM 协同恢复（全面加固 KI-06）；本仓 `tools/install_f09_editable.sh` 已同步覆盖全部 20 个 FlagTree F09 运行时文件。
