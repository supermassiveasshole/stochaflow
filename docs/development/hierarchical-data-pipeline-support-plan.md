# 一次准备、跨机器运行的大规模数据管线计划

> 文档类型：功能计划
>
> 工作状态：暂停（Parked）
>
> 当前可用性：现有框架能验证数据产物、用普通 `DataLoader` 训练；维护中的有限 map-style
> recipe 还能在已记录的 epoch 边界重建顺序，
> 但还不能把几 TB 到几十 TB 的本地原始数据可恢复地准备成一份便携数据版本，再复制到另一台机器直接使用。

## 故事从本地已有的几十 TB 数据开始

真实的大数据工作通常不是“让框架替我下载一个数据集”。数据可能早已散落在本地硬盘、NAS、
LMDB、tar、Parquet 或项目自己的目录中，预处理一次需要数小时甚至数天。用户先在 64 GB RAM、
24 GB 显存的 PC 上整理和试跑，之后再把结果复制到公司的 2 TB RAM、八卡 H200 服务器。如果换一台
机器就重新下载、重新解码或重新生成索引，准备工作不仅昂贵，也很难证明两边训练的确是同一份数据。

因此这项功能的中心不是“增加一层显存缓存”，而是：

```text
本地原始数据
  -> 可中断、可继续的数据准备
  -> 不可变且可验证的 prepared dataset
  -> 可恢复地复制到另一台机器
  -> 复核后用该机器自己的资源配置训练
```

准备后的数据版本承担身份和可搬运性；PC 与服务器的运行配置只决定怎样尽快读取它。换路径、worker
数量、RAM 预算或 GPU 数量都不能暗中改变样本、预处理结果和数据身份。这就是本计划所说的
“once prepared, run anywhere”。它并不承诺改变 world size、global batch size 后还能逐步复现同一
训练 run；那是分布式训练和训练恢复共同拥有的另一项保证。

## 数据准备必须独立于训练，而且可以从中断处继续

几十 TB 的转换不能藏在 `train` 的第一次启动中。训练进程不应一边等待数据，一边决定如何扫描、
切分和发布长期数据。未来应有一次独立的数据准备操作：它先盘点有限的源数据，展示样本、预计输出
体积和分片规划，用户确认后才开始昂贵转换。

开始写输出以前，准备过程要固定以下事实：经过认证的源数据快照、稳定样本编号、准备 recipe 及
版本、会影响结果的外部资产、输出 schema、容器格式和分片规划。它们共同形成准备计划的摘要。
任何一项变化都代表一份新数据，不能拿旧进度记录继续拼接。首版面对普通可变目录时，必须先完整
建立并认证输入 snapshot，再开始转换；可信的不可变存储快照或 provider checksum 可以作为另一种
明确的输入证明。一次预扫描并不会把普通目录冻结：转换每个输入时仍要按 snapshot 中的 size/digest
认证实际读取的 bytes，遇到替换、短读或内容变化立即失败。只有框架明确支持并认证的 immutable
filesystem/provider snapshot 才能免去这次逐输入复核。以后若要边转换边认证，只能采用两阶段计划：
最初的 discovery key 只定位未完成 workspace，逐 work unit 的输入摘要归并后才形成最终 source
identity，绝不能把 discovery key 当数据身份。

随后把工作确定性地分成有限单元。producer 只通过 Store 管理的 unit writer 写入任务内容；Store
拥有临时文件句柄、flush/fsync、内容摘要、关闭、原子重命名和 journal commit。每个单元核对样本数、
字节数和 schema 后才提交为已完成，producer 不能自行声明一个未经 Store 认证的 shard digest。进度日志
（journal）只记录这次尚未完成的准备工作和已经核对过的单元。进程被强制结束后，新进程会重新检查
这些单元，跳过完好的分片，只重做临时、缺失或损坏的部分。两个准备进程不能同时写同一个最终目标。

只有全部预定样本恰好覆盖、没有重复或遗漏、所有分片都通过核对后，`DataArtifactStore` 才发布最终
的 managed `DataArtifact`。半成品、进度日志和可丢弃的 scratch 永远不能被 `DataBuilder` 当作训练
输入。现有 Store 已拥有内容身份、逐文件摘要、隔离损坏内容和原子发布；实现应在它的最终发布之前
增加可恢复 workspace，而不是另建第二套较弱的 manifest 或绕过 Store 签发 handle。

当前 Store 会在 producer 返回后一次性枚举、排序并哈希全部输出。数千万文件时，这既可能重复读取
几十 TB 数据，也会让 inventory 构建占用过多内存。新的 Store-managed writer 应在关闭每个分片时
顺便生成大小、记录数和摘要；最终 inventory 采用 root 到 shard/index 的分层摘要，内存只随有界工作
单元增长。writer 的摘要可以消除 producer 自己已经做过的重复读取，
却不能默默取消 Store 当前签发 handle 前的最终完整认证。若受信 shard transaction 要替代某次完整
读回，必须先定义防篡改信任模型并修改 `SPEC.md`/`ARCHITECTURE.md`；在此之前仍执行现有 full verification。

确定性 resize、固定 crop、tokenization 或 codec 编码可以进入 prepared dataset。每个 epoch 变化的
随机 crop、颜色增强和随机 posterior sample 仍留在训练时。否则“准备一次”会不经意地删除任务原有
的数据分布。

## 被复制的是不可变内容，不是某台机器的 cache 目录

最终数据根只保存规范化相对路径、分片和索引。结果清单（manifest）应绑定源快照、准备 recipe、
数据语义、稳定样本覆盖、分片清单和内容摘要；本机盘符、绝对路径、cache 位置、worker 数和内存预算
都不属于数据身份。内部路径还要拒绝 `..`、盘符或 UNC 路径、大小写折叠后的重名、Windows 保留名，
以及依赖 symlink、junction 或 hardlink 才能成立的布局。这样同一 bundle 才能在 Windows PC 与 Linux
服务器之间无损搬运。

框架不需要重写文件传输工具。用户可以用经过选定和验证的 `rsync`、rclone、公司已有同步工具或
对象存储 multipart 模式继续传输：至少跳过已经完成的 shard；能否从单个 shard 的中间继续，取决于
工具参数和存储 backend。Stochaflow 负责提供完整内容清单，并在目标机器采用数据前核对每个声明分片。
接收流程应先写入未完成区域，传完后验证缺失、额外和同尺寸篡改，再按 artifact digest 把对象纳入
本地 Store。adopt 不能伪造原 producer 的 locator。首版的新训练配置要显式引用已 adopt 的 artifact
identity/manifest；兼容 `DataSource` 在当前请求中把它作为 expected identity，Store 完整验证后才签发
handle，`DataBuilder` 同时捕获该次 receipt。这样新服务器不需要旧 checkpoint 或原始目录，也不靠
locator key 猜对象。以后若需要更简短的选择方式，才另行设计窄的 imported-managed-artifact source，
不能让 portable descriptor 自己伪造 handle。安全时可原子移动或直接采用已经位于受控接收区的内容，
不能为了“导入”再复制几十 TB。

复制完成后，即使原始数据和原机器已经离线，目标机器也应能从这份 managed artifact 训练；但对应的
source/materializer 扩展、兼容 recipe 和 `DataBuilder` 仍须安装并能解释同一 domain/schema。若用户
只是给出 referenced folder、LMDB 或远程 locator，它仍可作为原始输入，但不是“准备一次、到处运行”
的最终结果，因为外部内容和绝对位置仍由用户管理。

## 三种“继续”解决的是三个不同的中断

笼统写一个 `resume: true` 会把完全不同的安全边界混在一起。本计划分别定义：

| 中断发生在哪里 | 谁保存继续所需事实 | 首个可用版本的保证 |
| --- | --- | --- |
| 数据准备中断 | preparation workspace 和已提交 work-unit journal | 从已核对分片继续；源快照或 recipe 改变时拒绝旧进度 |
| 数据复制中断 | rsync、rclone 或存储服务自己的传输状态 | 传输工具补齐内容；框架最终按 manifest 验证并采用新 locator |
| 训练中断 | checkpoint 中的数据身份、epoch、shuffle/partition 事实 | 先保证固定拓扑下从完整 epoch 边界继续，运行 cache 与队列重建 |

若一次 epoch 会持续数小时或数天，epoch 边界最终可能不够。届时 reader 要在一个 optimizer step 成功后
提交“下一个逻辑 batch”的游标，包括 epoch、分片顺序位置、分片内记录位置、shuffle 算法版本以及
rank/world-size 约定。预取到 RAM 或 GPU 但尚未消费的 batch 不进入游标；恢复后重新读取即可。
Mosaic Streaming 或 StatefulDataLoader 可以作为候选实现，但必须用选定任务证明精确覆盖与恢复，
不能把 worker 队列快照误称为数据准备 resume。

## 同一份数据在两台机器上只更换资源 profile

数据内容、运行时数据视图、当前机器上的位置和机器资源是四类不同事实：

- prepared artifact identity 说明源快照、recipe、schema、分片和每个稳定 sample ID；
- 训练数据视图说明 split、epoch、shuffle、rank assignment、drop 或 pad 规则，并进入相应恢复检查；
- 当前 locator 只说明这台机器从哪里找到已经认证的数据，不进入 artifact identity；
- 机器 execution profile 说明 reader 私有参数、worker、NUMA 放置以及磁盘、RAM、pinned memory 和
  每张 GPU 的队列预算。

后两类事实不能改写 prepared artifact identity。PC 和服务器使用同一个 digest，只选择不同 profile。
配置名称与字段尚未决定，下面只是资源责任的例子，不是可执行配置或公共 API：

| 资源 | 64 GB RAM / 24 GB VRAM PC | 2 TB RAM / 8×H200 服务器 |
| --- | --- | --- |
| 持久数据 | 本地 SSD/NVMe 上的 canonical artifact，也可直接作为当前 node 的来源 | 共享或集中保存 canonical artifact；按实测需要建立 node-local shard cache |
| 普通 RAM | 留足系统、训练和页缓存空间；少量 worker 与有界解码缓存 | 按 NUMA node 和 rank 分预算；不能让八个进程各自复制一份无界 cache |
| Pinned memory | 只放少数即将传给单张 GPU 的 batch | 每个 rank 独立的小队列，节点汇总后不得超过 pinned-memory 总预算 |
| 设备内存 | 模型、optimizer、激活之外只预留很少待用 batch | 八张卡各自拥有预算；不能把总显存看成一个共享 1 TB 数据 cache |

profile 先给出硬上限和必须保留的 headroom，而不是“尽量占满”。普通 `DataLoader` 只能按 worker、
batch 数和 `prefetch_factor` 限制队列；面对可变 batch，它不能直接保证 pinned memory 的精确字节上限。
首版要么要求任务声明可靠的 `max_batch_bytes`，用它做保守预检和 batch-count 上限；要么增加由 runtime
拥有、按字节回压的 feeder。device prefetch 也只能由后者拥有，不能把现有 `batch.to(device)` 描述成
显存 cache。某个 shard 或 batch 大于单层上限时应绕过该 cache 或明确失败，不能临时把内存翻倍。

## 磁盘、RAM 与显存不是同一种 cache

真正的层级应按生命周期分开：

| 层 | 保存什么 | 丢失后怎样处理 |
| --- | --- | --- |
| Canonical prepared shards | 唯一的数据事实、索引和 manifest | 不能静默重建成另一份内容；损坏必须失败或从可信来源重传 |
| Node-local disk cache | 从 canonical 位置取回的完整 shard | 可按内容摘要重新下载、淘汰或重建 |
| OS page cache | 最近读取的文件页 | 交给操作系统；默认不再造一份与它竞争的 Python 文件 cache |
| 可选 decoded host cache | 确定性解码或变换结果 | 只有复用和测量证明有收益时启用，并按总字节淘汰 |
| Pinned batch queue | 已组 batch、即将复制到设备的数据 | 中断后重建；属于有界传输队列，不是数据集副本 |
| Per-GPU prefetch queue | 已传到一张 GPU、等待计算的极少数 batch | 中断后重建；深度增加不再提升吞吐就保留最小值 |

上游比下游快时必须停止继续生产，这种回压让内存达到平台后保持稳定。缓存命中、队列深度和当前
吞吐都是一次运行的测量，不进入 artifact manifest 或便携 checkpoint。

## 成熟工具负责实现，Stochaflow 只闭合缺失的语义

首个任务先选择一种与数据形态匹配的成熟表示，而不是发明万能格式。MosaicML Streaming/MDS 提供
分片索引、有界本地缓存、预下载、分布式划分和训练恢复，是训练 reader 的强候选；WebDataset/WIDS
适合顺序读取的 tar 媒体分片；Arrow/Parquet 与 Hugging Face Datasets 更适合列式或 map-style 记录。
格式一旦准备完成便属于 artifact layout，不能在 PC 和服务器上临时用不同 shard size。小 shard 会
增加文件、请求和索引开销，大 shard 会增加失败重做、缓存驱逐和单次损坏的影响范围，默认范围必须
由代表性 workload 测量。

这些 reader 都不能自动解决整个产品。MDSWriter 的覆盖模式不是 preparation resume；HF fingerprint
是变换 cache key，不是完整性证明；WebDataset 的高吞吐也不自动提供严格的 mid-epoch 语义。Ray Data
可以并行准备数据，也能直接向 PyTorch/Ray Train 提供 batch；但本计划首版不把它定为维护中的训练
reader，而且它不拥有 Stochaflow 的 artifact identity、发布或 strict cursor 契约。当单机在容量、数据
位置或吞吐上无法承担准备时，才评估 Ray executor。fsspec 可以适配 locator 和 provider cache，却不能
把后端相关的半原子 transaction 当成集群发布真相。

DALI 只在 decode 或 augmentation 实测最慢时作为任务专用执行层；它只有在受支持的 reader/operator
graph 与固定 pipeline identity 下才可能精确恢复，不能替任意 `external_source` 作保证。GPUDirect Storage 只在目标存储、
文件系统和拓扑满足条件且 H2D 路径确为瓶颈时比较。它们都不负责 manifest、样本覆盖、准备 resume
和数据发布，也不会成为 PC 路径的必选依赖。

## 现有 Data 责任保持不变，但需要一个新的准备生命周期

`DataSource` 继续理解外部来源和任务自己的准备语义；`DataArtifactStore` 继续拥有 workspace 安全、
锁、inventory、验证和最终发布；`DataBuilder` 只在 artifact 已发布后创建 reader、Dataset view、
sampler、collate 和 loader。Store 可以提供通用的可恢复分片提交机制，但不能决定图像、文本、轨迹或
latent 的 work unit 与样本字段。

训练运行时负责 batch 到设备的搬运。只有测量表明同步复制正在让设备等待时，才加入独立 copy stream、
完成事件和有界 device prefetch；这些细节不应推给每个 Strategy。

[多设备训练计划](distributed-training-and-inference-support-plan.md)的首个固定单机 DDP 交付负责 rank、
world size、共同更新和进程失败。大数据管线可以先在 PC 与服务器单卡上完成准备、搬运和有界读取。固定八卡训练验收时，
`DataBuilder` 才根据明确拓扑实现可观察的 shuffle、rank/worker 分工、drop/pad 和恢复语义，并证明
各 rank 的 sample-ID union 符合选定 workload 的覆盖规则，而且没有未声明的交集；具体内部算法不强制
套成同一条流水线。数据管线不启动 DDP，DDP 也不猜任务的数据布局；后续多进程采样、FSDP2 和弹性
运行都不是这项八卡训练验收的前置。

## 第一个可用结果先闭合“准备一次、复制后运行”

这项 Parked 工作被选中后，应先指定一份真实、有限、不会边准备边追加的数据，一种成熟分片格式，
以及 PC 和服务器的实际存储与文件系统。首个交付只需要：

- 在 local filesystem 上把该数据可恢复地准备成 managed artifact；
- 使用稳定 sample ID、可搬运相对路径和分层 inventory；
- 提供 inspect、full verify、传输后 verify/adopt 的完整体验；
- 在 PC 单卡和服务器单卡用不同 profile 读取同一 identity；
- 使用 OS page cache，以及普通 PyTorch loader 的 batch-count 上限和可靠 `max_batch_bytes` 预检；若首个
  workload 要求精确的 pinned-memory 字节硬限制，则同时交付 runtime-owned byte-aware feeder；
- 从 preparation shard 边界和训练 epoch 边界继续；
- 报告准备、读取、解码、collate、H2D、计算和各层峰值资源。

固定八卡训练的全局覆盖随后依赖首个固定单机 DDP 能力。mid-epoch cursor、Ray executor、DALI、GDS、远程
对象存储、连续追加、弹性 world-size resume 和跨进程共享 decoded cache 都只在真实证据要求时加入。

其中有一个不能绕开的决定：当前 strict training resume 要求当前请求取得当次 full-verification receipt；
receipt 只是运行时证明，不进入 manifest 或 checkpoint。对几十 TB
数据每次启动都重新逐字节哈希可能需要数小时。实现前必须选择并写入规范：接受每次全量重哈希；或
定义受支持的可信不可变存储证明；或明确区分“完整重哈希”和“可信存储 attestation”的安全假设。
不能为了启动更快，把 manifest/size 检查悄悄改名为 full verification。

## 验收必须同时证明正确、可搬运和有界

小型 fixture 先在每个写入、关闭、提交和发布边界注入崩溃；恢复后的 bytes、inventory 和 artifact
identity 必须与不中断运行一致。真实数据验收还要证明：

- `kill -9` 后完好分片被复用，临时或损坏分片被重做；源快照或 recipe 改变会拒绝旧 journal；
- 复制到不同盘符和 Linux 路径后 identity 不变；缺 shard、多 shard、短读和同尺寸篡改都会失败；
- 没有原始数据时，已安装兼容扩展和 recipe 的环境仍能从 portable artifact 训练；
- PC 和服务器读取相同 logical sample bytes 与 sample ID；epoch 顺序由运行视图决定，随机变换后的
  batch tensor 只有采用相同 stateless transform/RNG 契约时才要求完全相同；RAM、pinned memory 和
  每张 GPU 的占用达到平台后不再增长；
- 首版在 PC 与服务器单卡证明相同 seed/epoch 的顺序可重建；
- 冷、热和 node-local cache 的条件写清，无法真正清空系统 cache 时不把结果标成“冷缓存”；
- 报告分别给出 preparation、存储读取、解码、变换、collate、loader wait、H2D 和 compute 时间，
  以及 cache hit/miss、队列 empty/full、retry 和峰值 disk/RAM/pinned/per-GPU VRAM；
- 增大 worker、cache 或 queue 后吞吐不再显著提高，就停止增加复杂度。

固定八卡的全局样本覆盖、共同更新和失败属于首个固定单机 DDP 阶段；游标与 checkpoint 的中途
联合恢复属于 mid-epoch 阶段。它们都有明确的后续验收，但不阻塞首个“准备一次、复制后单卡运行”的可用结果。

数据准备和跨机器可搬运本身已经是明确用户需求，不需要先证明 GPU 正在空转才值得选择。性能优化
则不同：Ray、DALI、GDS、decoded cache 和更深的 device queue 仍须由同一 workload 的实测瓶颈触发。
本轮只丰富 Parked 计划，不改变根路线图的 `In progress: None` 和 `Next: None`。

## 更详细的实现依据

[维护者技术附录](notes/hierarchical-data-pipeline-support-plan/design-and-research-notes.md)保存候选 workspace、
journal、manifest tree、训练游标、容量计算、故障矩阵、provider 比较和 benchmark 设计。它不是公共 API，
普通使用者无需阅读；具体类型名和配置形状只能在本方向被选中并完成首个 workload 设计后确定。

外部系统资料最后核对于 2026-08-12，实施前须重新检查版本、许可证、平台支持和当前语义：

- [MosaicML Streaming 数据格式](https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/dataset_format.html)、
  [分片读取与缓存](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/shard_retrieval.html)
  和[快速训练恢复](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/fast_resumption.html)；
- [WebDataset/WIDS](https://webdataset.github.io/webdataset/webdataset/)；
- [Hugging Face Datasets 处理与保存](https://huggingface.co/docs/datasets/process)和
  [Apache Arrow Dataset](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html)；
- [Ray Data checkpoint](https://docs.ray.io/en/latest/data/api/doc/ray.data.checkpoint.interfaces.CheckpointConfig.html)、
  [`iter_torch_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.iter_torch_batches.html)
  与[性能建议](https://docs.ray.io/en/latest/data/performance-tips.html)；
- [fsspec 功能与 transaction 边界](https://filesystem-spec.readthedocs.io/en/latest/features.html)；
- [PyTorch StatefulDataLoader](https://github.com/pytorch/data)和
  [stateful data loading 教程](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)；
- [NVIDIA DALI checkpoint 约束](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics_checkpointing.html)、
  [性能调优](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/advanced_topics_performance_tuning.html)
  与[GPUDirect Storage 概览](https://docs.nvidia.com/gpudirect-storage/overview-guide/)；
- [rclone copy](https://rclone.org/commands/rclone_copy/)和
  [Amazon S3 multipart checksum](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)。
