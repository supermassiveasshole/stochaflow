# Legacy Intel macOS PyTorch Test Lifecycle

- 文档性质：已实施的开发决策记录；不属于公开 API
- 状态：Implemented
- 日期：2026-07-28
- 适用范围：`macos-15-intel`、Python 3.12、PyTorch 2.2.2 CI compatibility lane

## 1. 触发问题

sampling request 与 data artifact breaking refactor 完成后，Intel macOS 测试出现过两种
表面相同、实际不同的退出卡死：

1. pytest 已打印完整 summary，但 PyTorch persistent DataLoader worker 没有被旧版
   shutdown path 完整回收；
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

worker persistence 不是这项 framework contract 的组成部分。因此该测试使用
`persistent_workers=False`，每个 epoch 由 PyTorch 正常启动并关闭 worker。

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

### 2.3 对已知旧平台增加 fresh-interpreter 回归

仅在 Intel macOS 且 `torch.__version__` 为 2.2.x 时，测试套件会：

1. 在主 pytest interpreter 中跳过直接执行 worker-count case，避免把上游 helper
   process 注入完整 suite；
2. 在新 POSIX session/process group 中启动一个 fresh pytest interpreter；
3. 通过 test-only environment gate 只执行同一个 worker-count independence case；
4. 最多等待 60 秒；
5. leader 正常退出后，再给 process group 5 秒自然清理窗口；
6. timeout 或残留 group 都执行有界 TERM → KILL；
7. 发生 cleanup 即让测试失败，不保留绿色结果。

测试输出写入普通临时文件，而不是 PIPE。这样即使 descendant 继承 stdout，也不会让
parent 因等待 PIPE EOF 再次卡死。

## 3. 边界与取舍

- 这是 legacy dependency compatibility lane 的测试策略，不改变 DataLoader public
  config，也不在 framework runtime 中调用 PyTorch 私有 `_shutdown_workers()`。
- production config 仍可选择 `persistent_workers: true`；该选项的具体 worker lifecycle
  属于所安装 PyTorch 的行为。
- session guard 只承诺管理 `multiprocessing.active_children()` 返回的普通 child；
  resource tracker 和 PyTorch native manager 由 fresh-interpreter 回归覆盖，不能假装
  guard 能观察它们。
- fresh-interpreter case 只在实际受影响的平台运行，避免为所有平台重复一项昂贵且属于
  上游实现的进程测试。
- hard timeout 是最后的 failure bound，不是通过条件，也不替代 root-cause regression。

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

最终验收以完整 GitHub Actions matrix 为准。Intel macOS job 必须在打印 pytest summary
后自然完成；fresh-interpreter timeout、process-group cleanup 或 session guard cleanup
都必须使 job 失败。
