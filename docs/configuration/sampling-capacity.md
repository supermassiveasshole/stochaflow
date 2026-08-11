# Sampling 结果需要多少内存

一次采样究竟会占多少内存，不能只看 `batch_size`。当前 sampling runtime 会先让
`SamplingBuilder.run()` 返回完整的 `SamplingOutput`，等所有 sample batch 和可选
trajectory 都在内存中以后，才把结果交给 writer 保存。内置 Standard Builder 会把
writer 需要的 Tensor 转到 CPU，但自定义 Builder 的公共
`SamplingBatch.samples: Any` 契约并不强制设备。

这意味着减小 `batch_size` 可以减轻模型单次前向的 accelerator 工作集，却不会减少
固定 `num_samples` 对应的最终 CPU 结果。打开 trajectory 后，每一份保留的中间状态
还会让整批结果再驻留一次；writer 拼接和编码时则可能需要额外副本。先做容量投影，
再决定样本数、trajectory 间隔和 writer，比运行到一半才发现内存不足更可靠。

## 从 shape 算出最低占用

设 $N$ 是样本数，$P$ 是单样本 Tensor 的元素数（不含 batch 维），$d$ 是每个元素的
字节数。一份完整 state snapshot 的原始 Tensor payload 是：

$$
Q = N P d.
$$

若 solver 接受了 $K$ 步，`trajectory.every_steps` 为 $e$，trajectory 会保留 initial、
final，以及每个能被 $e$ 整除的 accepted step。当 $K>0$ 时，保留的 snapshot 数是：

$$
S = 1 + \left\lceil \frac{K}{e} \right\rceil.
$$

对当前内置 Tensor 路径，可以先用下面的下界判断一个请求是否明显过大：

| 阶段 | final-only | 启用 trajectory |
| --- | ---: | ---: |
| Builder 返回后的原始 payload | $Q$ | $Q(1+S)$ |
| `tensor` writer 的结构性峰值下界 | $2Q$ | $Q(1+3S)$ |
| `samples.pt` 与 `trajectory.pt` 的原始 payload | $Q$ | $Q(1+S)$ |

`tensor` writer 会先拼接所有 sample batch，再逐个拼接 trajectory snapshot，最后把
snapshot stack 成一个 Tensor。因此，writer 开始工作并不代表 Builder 的完整结果已经
释放。`image` writer 还会保留拼接后的 sample Tensor，并产生网格、归一化和 PNG/GIF
编码所需的临时内存。

这些数值是由当前数据流得到的逻辑 payload 和分配下界，不是进程 RSS 或显存的跨平台
保证。真实运行还包含模型 activation、allocator、serializer、Python object、backend
workspace 和 OS page cache；文件也会有 header 与编码开销。

## 先投影，再运行小规模基准

仓库在 `benchmarks/sampling_capacity_profiles.yaml` 中保留几种与具体任务无关的
profile：

| profile | 用途 | 是否实际分配完整 Tensor |
| --- | --- | --- |
| `ci_smoke` | 很小的本地和 CI 生命周期检查 | 是 |
| `field3d_preview` | 小规模三维场与 trajectory 写入 | 是 |
| `high_resolution_1024_projection` | 高分辨率图像与长 trajectory 的容量投影 | 否 |
| `field3d_dense_projection` | 较大三维场与长 trajectory 的容量投影 | 否 |

先查看当前 profile：

```bash
uv run python tools/benchmark_sampling_capacity.py \
  --profiles benchmarks/sampling_capacity_profiles.yaml \
  --list
```

只想检查公式和报告格式时，可以运行很小的 smoke profile：

```bash
uv run python tools/benchmark_sampling_capacity.py \
  --profiles benchmarks/sampling_capacity_profiles.yaml \
  --profile ci_smoke \
  --device cpu \
  --result outputs/benchmarks/sampling-capacity-smoke.json
```

投影 profile 不会分配声明中的大 Tensor。下面的命令可以安全比较高分辨率图像和三维场
请求的逻辑 payload：

```bash
uv run python tools/benchmark_sampling_capacity.py \
  --profiles benchmarks/sampling_capacity_profiles.yaml \
  --profile high_resolution_1024_projection \
  --profile field3d_dense_projection \
  --device cpu \
  --result outputs/benchmarks/sampling-capacity-projection.json
```

需要观察当前主机上的实际 writer 峰值时，再运行有界的三维场 profile：

```bash
uv run python tools/benchmark_sampling_capacity.py \
  --profiles benchmarks/sampling_capacity_profiles.yaml \
  --profile field3d_preview \
  --device cpu \
  --result outputs/benchmarks/sampling-capacity-field3d.json
```

每个可执行 profile 都在 fresh subprocess 中运行。工具在启动 worker 前会把投影工作集
和原始 artifact payload 与当前内存、临时文件系统预算比较；默认超过 50% 就拒绝，
并为每个 worker 应用 timeout。`--allow-over-budget` 只表示用户读过投影后接受风险，
不会降低实际占用，也不会把报告中的 70% 参考线变成安全保证。

## 怎样阅读和保存结果

shape、count、dtype 和上面的公式只能证明逻辑驻留下界；某台机器上的 RSS、device
allocated/reserved、wall time 和 throughput，只能说明该主机、backend、PyTorch 版本和
writer 的这一次运行。它们不能证明另一种硬件一定不会 OOM。

记录本地主机基准时，至少保留 OS、Python/PyTorch 版本、device/backend、CPU/GPU
内存、dtype、shape、sample/batch 数、trajectory observation 数、writer 和产物字节数。
机器相关报告属于运行产物，请写入已忽略的 `outputs/benchmarks/`，不要提交 JSON、日志
或某台主机的测量表。需要比较两次运行时，应在同一受控环境中保存报告，或交给外部
实验跟踪系统。

仓库只版本化工具、通用 profile，以及投影公式和执行生命周期的测试。CI 不读取历史
机器结果，也不会把某台机器的数值当作跨平台阈值。

## 请求太大时可以怎样缩小

在当前生命周期内，最有效的办法是减少必须同时存在的结果：

- 降低 `num_samples`；
- 主运行只保存 final output，把 trajectory 放进单独的小样本 preview；
- 增大 `trajectory.every_steps`，减少保留的 observation；
- 让任务 Builder 先完成指标计算，只把 writer 真正需要的场或通道放入
  `SamplingOutput`；
- 先用投影 profile 确认逻辑下界，再在目标设备上逐步增加 batch 和样本数。

只注册一个新 writer 不会把当前生命周期变成 streaming。writer 收到数据时，Builder
的全部 output 已经物化；如果任务确实需要全量 dense trajectory 或无法整体驻留的结果，
需要先设计增量的 sampling/writer 生命周期，而不是让 Builder 越过 writer 直接写文件，
也不应把现有接口称为 streaming。
