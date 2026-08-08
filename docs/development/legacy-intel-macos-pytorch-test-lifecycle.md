# Legacy Intel macOS PyTorch Test Lifecycle

- 文档性质：已实施的开发决策记录；不属于公开 API
- 状态：Retired；兼容 lane、依赖 pins 与运行时 skip 已于 2026-08-08 移除
- 统一排期：[Development Priority Roadmap](development-priority-roadmap.md)；不再维护
- 日期：2026-07-28
- 平台支持等级：Unsupported
- 适用范围：仅记录已删除的 `macos-15-intel` / PyTorch 2.2.2 历史验证路径

Intel macOS 的迁移期已经结束。当前源码、dependency metadata 与 CI 均不再包含该
平台的专用兼容路径，也不要求 framework capability 为其增加适配。公开定义以
[平台支持政策](../platform-support.md)为准；下文只保留移除前问题与决策的历史记录。

## 1. 触发问题

sampling request 与 data artifact breaking refactor 完成后，Intel macOS 测试出现过两种
表面相同、实际不同的退出卡死：

1. pytest 已打印完整 summary，但 PyTorch multi-worker DataLoader 的 helper process
   没有被旧版 shutdown path 完整回收；最初由 persistent worker 触发，后续隔离验证
   证明 `persistent_workers=False` 也会命中同一问题；
2. 第一轮修复增加的 leak-guard 自测又使用了真实
   `multiprocessing.get_context("spawn")`，从而由测试自身启动了不属于
   `multiprocessing.active_children()` 的 `resource_tracker`。

runner 取消作业时观察到的进程链为：

```text
pytest
  ├─ multiprocessing resource_tracker (Python)
  └─ DataLoader worker (Python)
       └─ torch_shm_manager
```

PyTorch 2.2.2 的 DataLoader shutdown 会对超时 worker 执行 `terminate()`，但不会像当前
版本一样在 terminate 后再次 join。macOS 的 spawn 与 file-system tensor sharing 又会
额外引入 resource tracker 和 `torch_shm_manager`。因此，仅检查
`multiprocessing.active_children()` 既看不到所有 helper process，也不能证明解释器可
正常退出。

capacity benchmark 被排除为来源：outer tool 与 fresh worker 都由同步
`subprocess.run()` 等待；worker 不使用 DataLoader、multiprocessing 或 shared-memory
tensor transport。若任一 Python worker 尚未退出，该测试本身不可能先返回 PASSED。

## 2. 最终决策

### 2.1 分离被测语义

`test_class_labeled_loader_is_worker_count_independent` 要验证的是：

- `num_workers=0` 与 `num_workers=1` 产生相同 epoch order 和 Tensor；
- 两个 epoch 的 deterministic transform 结果与 worker 数量无关。

worker persistence 不是这项 framework contract 的组成部分。因此该测试在正常支持的
平台使用 `persistent_workers=False`，每个 epoch 由 PyTorch 启动并关闭 worker。

`persistent_workers=True` 仍由独立的无迭代构造测试验证参数透传。该测试不启动 worker，
避免把上游旧版解释器退出机制伪装成 Stochaflow 的数据语义。

### 2.2 不使用 clean-and-green 强制退出

没有采用在 `pytest.main()` 返回后直接 `os._exit(0)`，或无条件清理所有进程后保留成功
返回码的全局 CI wrapper。该方案会跳过 Python `atexit`，把真实 lifecycle leak 隐藏成
绿色结果。

CI 继续保留：

- `-vv --durations=25`，使最后完成的测试和 teardown 时间可见；
- test step 的 10 分钟 hard timeout，保证未知退出问题有最终上界；
- session-level multiprocessing guard，发现普通 `Process` child 时有界
  terminate、kill、join 并让测试失败。

guard 的自测只使用 deterministic process doubles。它不再为了测试 cleanup state
machine 而创建真实 `spawn`/resource tracker。

### 2.3 当时的 deprecated legacy 平台例外

曾实现过一项只在 Intel macOS + PyTorch 2.2.x 运行的 fresh-interpreter regression：
它在独立 POSIX session 中执行 `num_workers=1, persistent_workers=False` 的同一
worker-count case，并设置 60 秒上限。实际 CI 结果是：

- case 的 assertions 已完成；
- child pytest 仍未在 60 秒内退出；
- TERM 后 native helper group 仍存在；
- 强制清理 native manager 在 hosted runner 上不具备可靠、可移植的权限语义。

这证明问题不是 Stochaflow 的 persistence 配置，也不是测试忘记 join 普通
`multiprocessing.Process`，而是该 legacy dependency 组合的上游 multi-worker
interpreter lifecycle。保留一个必然失败的 isolated regression，或通过 `os._exit(0)`
强制变绿，都没有价值。

迁移期最终只在以下精确组合跳过 multi-worker runtime case：

```text
sys.platform == "darwin"
platform.machine() == "x86_64"
torch.__version__.startswith("2.2.")
```

Ubuntu、Windows 和 ARM macOS 继续执行完整 `num_workers=1`、双 epoch deterministic
contract。Intel macOS 当时仍执行 loader construction 和
`persistent_workers=True` 参数透传测试，但不启动 DataLoader worker；这些 skip 现已删除。

保留 Intel CI job 曾是迁移期观测手段，而不是 Supported 等级承诺。该 job 用于暴露
不涉及已知上游缺口的 framework 回归；2026-08-08 已连同整个 compatibility lane
移除。

## 3. 边界与取舍

- 这是已删除的 dependency compatibility lane 的历史测试策略，不改变
  DataLoader public config，也不在 framework runtime 中调用 PyTorch 私有
  `_shutdown_workers()`。
- production config 仍可选择 `persistent_workers: true`；该选项的具体 worker lifecycle
  属于所安装 PyTorch 的行为。Intel macOS + PyTorch 2.2.x 上的 `num_workers>0`
  interpreter shutdown 是明确的上游已知限制，不由 Stochaflow 声称修复。
- session guard 只承诺管理 `multiprocessing.active_children()` 返回的普通 child；
  resource tracker 和 PyTorch native manager 不在该保证内，不能假装 guard 能观察或
  可移植地终止它们。
- 精确 platform/version skip 曾比全局禁用 multi-worker 测试更窄；该 skip 已随
  transitional lane 一起删除，当前支持矩阵重新执行真实 runtime case。
- hard timeout 是未知退出问题的最后 failure bound，不是通过条件。

## 4. 验收

聚焦验证：

```bash
uv run pytest -q \
  tests/test_class_labeled_image_runtime.py \
  tests/test_multiprocessing_leak_guard.py
uv run ruff check \
  tests/test_class_labeled_image_runtime.py \
  tests/test_multiprocessing_leak_guard.py \
  tests/conftest.py
uv run pyright
```

本次 lifecycle 修复的验收以当时的完整 GitHub Actions matrix 为准：Intel macOS job
明确报告该 platform/version skip，并在打印 pytest summary 后自然完成；session guard
cleanup 仍会使 job 失败。这是该修复的历史验收记录，不承诺未来版本继续保留 Intel
lane 或把它作为发布门禁。
