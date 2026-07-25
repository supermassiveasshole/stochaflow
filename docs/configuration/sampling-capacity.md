# Sampling artifact 容量边界

## 决策摘要

当前 sampling lifecycle 是**整体物化式**：`SamplingBuilder.run()` 先返回完整
`SamplingOutput`，其中所有 sample batch 和可选 trajectory 都已存在；随后 runtime
才把同一个完整 `SamplingArtifactContext` 依次交给 writer。内置 Standard Builder
会将 writer-ready Tensor 转存到 CPU；公共 `SamplingBatch.samples: Any` contract 不强制
自定义 Builder 的设备。

这个 contract 适合有界的离线采样，但不是 streaming contract：

- Builder 返回前不会把 batch 逐个交给 writer；
- writer 可以逐 batch 编码已收到的数据，但不能消除
  `SamplingOutput` 在 writer 开始前的整体驻留；
- 当前没有 `begin/write_batch/finish`、sink 或自定义 streaming writer 生命周期。

因此，本文把 Physics AI 参考案例限定为“全量 final-only 结果 +
独立的小样本 trajectory preview”。全量 dense trajectory 不在当前支持
边界内。若未来出现真实需求，应先设计新的增量 lifecycle，不应让
Builder 越过 writer 职责直接写文件，也不应把当前 writer 误称为
streaming。

## 可计算的驻留下界

设：

- $N$ 是样本数；
- $P$ 是单样本 Tensor 的元素数，不含 batch 维；
- $d$ 是每个元素的字节数；
- $Q = N P d$ 是一份全量 state snapshot 的 raw Tensor payload；
- $K$ 是 accepted solver step 数；
- $e$ 是 `trajectory.every_steps`。

trajectory 会强制保留 initial 和 final，并保留每个能被 $e$ 整除的
accepted step。当 $K>0$ 时，保留的 snapshot 数为：

$$
S = 1 + \left\lceil \frac{K}{e} \right\rceil.
$$

对当前内置 Tensor 路径：

| 阶段 | final-only | 启用 trajectory |
| --- | ---: | ---: |
| Builder 返回后的 raw payload | $Q$ | $Q(1+S)$ |
| `tensor` writer 的结构性峰值下界 | $2Q$ | $Q(1+3S)$ |
| `samples.pt` + `trajectory.pt` raw payload | $Q$ | $Q(1+S)$ |

`tensor` writer 会先临时拼接 sample batch，并在 `torch.save()` 返回后释放该
Tensor 引用；随后它为每个 snapshot 拼接 batch，再将所有 snapshot `stack`
成一个 Tensor。两阶段峰值分别为 $2Q+SQ$ 和 $Q+3SQ$；因为 $S\ge 2$，
表中 tensor 路径取后者。`image` writer 则会把拼接后的 sample Tensor 保留到
trajectory 组装阶段，因此该路径的结构性下界是 $2Q+3SQ$，此外还有
PNG/GIF 编码开销。

这些数字是根据当前数据流得到的**逻辑 payload 和分配下界**，不是
进程 RSS 的跨平台保证。实际 RSS 还包含 allocator、serializer、Python
object、数据源、模型和 OS page cache。`.pt`/`.npy` 实际文件也会有
header 和 serialization overhead。

## Physics reconstruction 参考 profile

Physics AI 参考案例使用二维 Kolmogorov flow reconstruction：

- 测试集是 4 条时间序列，每条 320 帧；
- 每个 state 由连续 3 帧组成，shape 为 `[3, 256, 256]`；
- 滑动窗口后 $N=4(320-2)=1272$；
- state 为 float32，因此一份全量 3-channel snapshot 为
  `1,000,341,504 B = 954 MiB = 0.931640625 GiB`；
- 只保留中心物理帧时，全量 payload 为
  `333,447,168 B = 318 MiB = 0.310546875 GiB`。

容量投影使用两种 solver 长度：

| profile | partial-noise time | accepted steps | 主输出 trajectory |
| --- | ---: | ---: | --- |
| baseline DFSR | 240 | 30 | 关闭 |
| physics-guided DFSR | 320 | 40 | 关闭 |

例如 batch 8 和 batch 1 的单份 device state payload 分别只是 6 MiB 和
0.75 MiB，但 sampling batch size 不改变全部 final output 的总 payload，也不能代表
accelerator 峰值；模型 activation、physics residual autograd 和 backend workspace 必须在
目标设备上实测。checked-in production config 和目标设备容量测试才是实际 batch size 的
权威。

容量验收默认按最保守的 3-channel final output 计算：Builder 结束时
保留 0.9316 GiB，内置 `tensor` writer 的结构性峰值下界为
1.8633 GiB。真实领域 writer 可以只接收 Builder 归一化后的中心涡量场：
Builder 先在每个 batch 上计算需要的 L2/PDE 指标，再丢弃不需要的两个
边界帧，仅将中心帧放入 writer-ready `SamplingBatch`。这时一份结果是
318 MiB。若同时持久化 low-resolution input、reference 和 reconstruction
三份中心场，raw payload 合计约 954 MiB。

这种领域归一化会减少物化数据，但仍然不是 streaming：writer 仍在
Builder 返回所有中心场 batch 后才开始。

## 为什么不支持全量 dense trajectory

| profile | $K/e$ | $S$ | Builder raw payload | `tensor` writer 结构性峰值下界 | raw artifact payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFSR dense | 30/1 | 31 | 29.8125 GiB | 87.5742 GiB | 29.8125 GiB |
| guided DFSR dense | 40/1 | 41 | 39.1289 GiB | 115.5234 GiB | 39.1289 GiB |
| DFSR every 10 | 30/10 | 4 | 4.6582 GiB | 12.1113 GiB | 4.6582 GiB |
| guided DFSR every 10 | 40/10 | 5 | 5.5898 GiB | 14.9063 GiB | 5.5898 GiB |

参考实现本身也不保存 reverse-state trajectory：它每个 accepted step 后丢弃
旧 state，只纪录 batch 聚合的 L2/PDE 标量。“40 条 trajectory”指物理数据的
时间序列，不是 40 份 reverse sampling snapshot。

因此，`1272 x [3, 256, 256]` 的全量 dense trajectory 属于当前 contract
的明确非支持场景。降低 sampling batch size 只会减少 device-side 单次工作集，
不会改变全部 CPU trajectory 在 writer 前整体驻留的事实。

## 独立 trajectory preview

trajectory 可以在与主重建分开的采样调用中生成。Physics AI 参考项目
对 preview 使用以下容量限制：

- `sampling.num_samples <= 8`；
- `trajectory.every_steps >= 10`；
- accepted steps 不超过主 profile 的 40；
- preview 可以用 tensor/image writer，但不与 1272-sample 主 artifact 合并。

对 8 个 `[3, 256, 256]` float32 sample，每个 snapshot 为 6 MiB。30 步
preview 保留 4 份 snapshot，Builder/raw artifact payload 为 30 MiB，`tensor`
writer 结构性峰值下界为 78 MiB。40 步 preview 保留 5 份 snapshot，
对应为 36 MiB 和 96 MiB。image writer 的实际峰值和文件大小仍需在
目标主机上测量；下方参考主机表已提供一组 high-entropy PNG/GIF 证据。

## 验收证据的层级

| 证据 | 可以得出的结论 | 不能得出的结论 |
| --- | --- | --- |
| shape/count/dtype 与上述公式 | raw payload、snapshot 数和必需的逻辑驻留 | 某种硬件一定不 OOM |
| 当前 Builder/Writer 数据流 | 整体物化和 `cat`/`stack` 带来的结构性分配下界 | allocator 和 serializer 的精确额外开销 |
| 指定主机的 RSS/device 基准 | 该主机、该 backend 和该 writer 的当次容量证据 | Linux/CUDA/MPS/其他 PyTorch 版本的跨平台保证 |
| writer 产物的实际字节数 | 指定格式、编码器和数据的存储成本 | 其他数据分布、压缩级别或文件系统的成本 |

记录主机基准时，至少要保存 OS、Python/PyTorch 版本、device/backend、
CPU/GPU 内存、dtype、shape、sample/batch 数、accepted steps、trajectory
间隔、writer 和产物字节数。RSS、accelerator allocated/reserved、wall time 和
throughput 都是**参考主机证据**，不应写成通用容量保证。

### 复现参考主机基准

受版本控制的 profile 位于 `benchmarks/sampling_capacity_profiles.yaml`。先查看
可执行 profile 和只做数学投影的 profile：

```bash
uv run python tools/benchmark_sampling_capacity.py \
  --profiles benchmarks/sampling_capacity_profiles.yaml \
  --list
```

下列命令在 fresh subprocess 中分别运行 full final-only 和小样本 preview，
同时包含 image/GIF、dense stress、3D 小样本和只做数学投影的大 profile，
并保存与当前参考结果同构的机器可读报告：

```bash
uv run python tools/benchmark_sampling_capacity.py \
  --profiles benchmarks/sampling_capacity_profiles.yaml \
  --profile dfsr_final \
  --profile dfsr_trajectory_preview \
  --profile dfsr_image_preview \
  --profile dfsr_dense_trajectory_stress \
  --profile field3d_preview \
  --profile high_resolution_1024_projection \
  --profile field3d_dense_projection \
  --profile dfsr_full_trajectory_projection \
  --device cpu \
  --result outputs/sampling-capacity.json
```

更换 `--device` 会得到新的参考主机/backend 证据，不是对原报告的
“通过性增强”。投影 profile 不分配对应的大 Tensor，只根据 shape、dtype、
sample 数和 observation 数计算上述结构性 payload。

### 2026-07-22 参考主机结果

已提交的机器可读结果位于
`benchmarks/results/stage6-macos-arm64.json`。参考环境为 macOS 26.4.1 arm64、
Python 3.14.3、PyTorch 2.11.0、16 GiB 有效主机内存；此次使用 CPU
物化 synthetic writer-ready output，每个可执行 profile 运行 1 次不进入统计的
discarded fresh-process run 和 5 个 fresh-process measured repeat。JSON 同时保存当次
tool/profile SHA-256、execution override、临时文件系统空间和可获取的主机内存证据。
其中 `tool_sha256` 始终保留实际测量时的工具来源；未重跑基准的跨平台兼容性维护，
单独记录为 `audited_compatible_tool_sha256` 和说明，不冒充新的测量结果。

| profile | raw output | actual artifacts | median/max peak RSS | max/host | RSS CV | median wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFSR final-only, tensor | 954 MiB | 954.00 MiB | 1.935/2.160 GiB | 13.50% | 8.43% | 1.31 s |
| sparse preview, tensor, 4 states | 30 MiB | 30.00 MiB | 0.365/0.365 GiB | 2.28% | 0.04% | 0.04 s |
| sparse preview, image/GIF, 4 states | 30 MiB | 10.29 MiB | 0.530/0.531 GiB | 3.32% | 0.06% | 0.59 s |
| dense stress, tensor, 31 states | 192 MiB | 192.00 MiB | 0.840/0.841 GiB | 5.25% | 0.02% | 0.24 s |
| small 3D field, tensor, 5 states | 48 MiB | 48.00 MiB | 0.418/0.418 GiB | 2.61% | 0.06% | 0.06 s |

该主机上 DFSR final-only 的 5 次最大 peak RSS 为 2.160 GiB，低于 70%
参考线；high-entropy image/GIF preview 的最大 peak RSS 为 0.531 GiB。这组结果说明
当前参考目标无需先行引入 streaming API，但不改变全量 dense trajectory 的结构性
非支持结论：1272-sample/31-state 投影的
`SamplingOutput` raw payload 为 29.81 GiB，当前 `tensor` writer 结构性峰值下界为
87.57 GiB。

这张表只验收 CPU artifact lifecycle。后续 Physics reference project 已在维护者参考
环境记录真实 mmap 数据、一次 MPS optimizer update，以及完整 30/40-step 单样本
sampling runtime；证据位于
`benchmarks/results/stage7-physics-macos-arm64.json`。该记录仍不包含收敛训练、
1272-sample 全量运行、科学精度复现或 Linux/CUDA 跨平台容量保证，因此不能把任一机器
结果当作通用硬件承诺。

final-only Tensor writer 的 5 次 lifetime RSS high-water 呈现两个分配高水位，因此
CV 为 8.43%。容量决策使用五次中的最大值 2.160 GiB，而不用中位数缩小风险；
由于最大值仍只占主机内存的 13.50%，该变异不改变当前 artifact API 的容量判断。

工具在启动 worker 前会将投影工作集和 raw artifact 与当前内存/临时文件系统
预算比较，默认在超过 50% 时拒绝，并对每个 fresh worker 应用 profile timeout。
`--allow-over-budget` 只是用户阅读投影后的显式风险接受，不会改变 70% 报告参考线。
磁盘预检会按 writer 数累加 input-sized raw payload；自定义格式可以产生更大的
artifact，因此这是防误操作 guard，不是通用磁盘上界。

## Physics AI 参考项目的使用约束

1. 主采样固定为 1272 个 `[3, 256, 256]` float32 生成 state 的
   final-only profile；baseline/guided 分别使用 30/40 个 accepted step。batch size
   按 checked-in config 和目标 accelerator 的实测容量选择。
2. 主配置必须关闭 trajectory，并使用领域 writer 输出场数据和指标；
   image 只是独立 preview artifact。
3. Builder 必须逐 batch 处理 low-resolution/reference input，计算指标后只将
   writer 需要的场放入 `SamplingOutput`。数据源不得因为方便而长期保留
   两份 `[1272, 3, 256, 256]` 展平 Tensor。
4. trajectory preview 必须是单独采样调用，且 `num_samples <= 8`、
   `every_steps >= 10`、accepted steps 不超过 40；writer-ready field 必须转存到
   CPU，不得让全量 device output 在 Builder 返回后继续占用 accelerator。
5. 不得将当前 contract 描述为支持自定义 streaming。若案例需求扩展到
   全量 dense trajectory，必须停止案例实施，回到 sampling lifecycle 设计阶段。
