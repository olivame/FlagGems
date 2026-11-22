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
- **Tier 0.5（`_bkt_l0_bucket_cache`）**：在 Tier 0 未命中且算子无后置 heuristics 时，通过 FlagTree `policy.normalize_entry_scalar()` 生成分桶级键；命中后直接复用已编译内核并将当前精确 shape 提升写入 Tier 0，实现**未见变长首轮零重编译、次轮零分桶查表开销**。
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
扫描 `src/flag_gems/ops/*.py` 与 `src/flag_gems/runtime/backend/_*/*.py`，检测四类破坏动态分桶的反模式：
1. `shape_stride_constexpr`：动态维度/步幅标注 `tl.constexpr`；
2. `wrapper_divisibility_kw`：包装函数内计算 `EVEN_*` / `ALIGNED_*` / `DIVISIBLE_*` 关键字；
3. `wrapper_next_pow2_block`：包装函数内计算 `BLOCK_* = triton.next_power_of_2(...)`；
4. `op_level_bucketing_branch`：算子内部硬编码 `shape_bucketing` 分支。

支持 `--strict-paths <rel_paths...>` 作为 CI 阻断门禁，确保已收敛算子不发生回归。

---

## 2. 上游同步与补丁维护指南

当上游 `FlagGems` `master` 推进并需要迁移 F09 补丁栈时，建议按以下顺序操作：
1. **按顺序变基/合入补丁栈**：从基线 `4d3d9f70` 起依次应用 PR-1（`libentry.py` + `pointwise_dynamic.py`）、PR-2（算子零重编译与反向修复）、PR-3（合规扫描器）、Fix-1、Fix-2 与 Docs/Runner 提交。
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
| **KI-FG-01**：`_thead/ops/bmm.py` `aiu_load_mask` 对齐态翻转导致预热后额外调优 | 使用自定义 eager attention 下沉至 `bmm` 且序列长度跨越 `S % 16 == 0` 与 `S % 16 != 0` 的模型（如 Jina v2/v3） | 在 `max_len=768` 仅用 `%16 != 0` 长度预热后，输入 `S=288/576`（`%16 == 0`）触发 `aiu_load_mask` 键值切换并新增冷调优 | 预热长度集合同时覆盖 `%16 == 0` 与 `%16 != 0` 两类对齐态（例如加入 `128, 288` 与 `264, 524`） | FlagGems `_thead` 算子维护者 | P1 阶段解耦 `bmm.py` `@libtuner(key=...)` 中的 `aiu_load_mask` 或在默认预热推荐中覆盖双对齐态 |
| **KI-FG-02**：离线环境下 FlagTune 远程 Cost Model 拉取超时与 60s 静默重试 | 无外网或目标模型包不可达的容器环境中未显式关闭 cost model 的全部评测/训练任务 | 在离线环境省略 `USE_FLAGTUNE_COST_MODEL=0` 运行含 autotune 的模型，触发 `cost_model.py` 远程拉取卡顿 39–264s 并每 60s 静默重试 | 运行前统一设置 `export USE_FLAGTUNE_COST_MODEL=0`（`benchmarks/f09_acceptance_runner.py` 已默认强制设置） | FlagGems FlagTune 集成维护者 | P1 阶段在离线/失败后引入指数退避或快速熔断配置，并保持默认基准关闭远程拉取 |
| **KI-FG-03**：静态合规扫描遗留项（`generic_ops/mul.py`、`fused/` 盲区与 `_thead` 3 处 `wrapper_divisibility_kw`） | `ops/mul.py` 广播/复数路径、`fused/fused_moe.py` 以及 `_thead/ops/{bmm,mm,quantized_lstm}.py` 的小 M/变长 batch 路径 | 执行 `python3 tools/check_bucketing_compliance.py --root .` 可见 `generic_ops` `op_level_bucketing_branch=1` 及 `_thead` `wrapper_divisibility_kw=3` | 当前已验证主干推理/训练路径不受阻断；新增算子通过 `--strict-paths` 门禁防止扩散 | FlagGems F09 架构维护者 | P1 阶段扩展扫描器覆盖 `src/flag_gems/fused/`，并将上述遗留点统一迁移至 `@triton.heuristics` 与运行时分桶 |
| **KI-FG-04**：`LibEntry` L0 缓存多线程并发访问与 `_build_bkt_l0_bucket_key` 未传 `all_args` | 单进程多线程并发调度同一算子或依赖 `s_contig` 步幅配对的特定动态轴场景 | 在多线程压测下并发读写 `_bkt_l0_cache` / `_bkt_l0_bucket_cache`，或动态轴步幅配对变化时命中 Tier 0.5 | 生产训练/推理采用标准多进程（DDP/TP 每卡单进程）部署，避免单进程多线程并发调用同一 `LibEntry` | FlagGems 运行时维护者 | P1 阶段为 L0 读写/淘汰补齐轻量锁保护，并在 `_build_bkt_l0_bucket_key` 中补全 `all_args` 与 `observe_args()` |
| **KI-FG-05**：单芯片实机验证边界与 FlagGems 相对 ATen 的算子基线稳态差距 | 非 PPU 硬件后端（CUDA/ROCm/NPU）及小中型模型稳态吞吐绝对值（注：F09 相对 FlagGems OFF 稳态为 `~1.01–1.05×` 零损耗） | 在小模型对比 `aten_ref` 与 `flag_gems_on` 稳态耗时可见算子库基线差距；在非 PPU 设备上尚未完成全量实机回归 | 报告与验收指标明确区分“F09 相对 FlagGems OFF 收益”与“算子库相对 ATen 基线”；跨芯片上线前先跑 `benchmarks/f09_acceptance_runner.py` 与单测 | F09 联合交付团队 | P1 阶段引入 NVIDIA CUDA 实机跨后端验证（单测 + 真实模型 + NCCL DDP），持续优化小形状算子启动开销 |
| **KI-FG-06**：`DistributedDispatchGuard` 要求同通信组内各 Rank 算子调用序列对称 | 开启 `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1` 且存在跨 Rank 不对称控制流分支（如不同 Rank 跳过某算子）的模型 | Rank 0 执行受保护 `LibEntry` 算子而 Rank 1 走控制流分支跳过该算子，将引发集合通信失配超时 | 仅在各 Rank 算子调用序列对称的 DDP/TP/FSDP 路径开启守卫；不对称 MoE 路由请在更高层控制流对齐后再调用受保护算子 | FlagTree + FlagGems 分布式维护者 | P1 阶段支持按算子/通信组细粒度标注对称守卫作用域，并补充更明确的超时诊断日志 |
