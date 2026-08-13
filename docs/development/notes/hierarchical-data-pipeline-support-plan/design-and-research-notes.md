# 大规模数据管线的设计与工具研究

> 文档性质：维护者技术附录
>
> 当前状态：支撑一项 Parked 功能计划，不代表接口已实现或已排期
>
> 最后核对：2026-08-12

本文展开[主计划](../../hierarchical-data-pipeline-support-plan.md)中的实现问题。下面出现的名称、
目录和配置都只是候选设计，不能执行，也不构成公共 API。稳定行为只有在实现、测试并写入 `SPEC.md`
与 `ARCHITECTURE.md` 后才成立。

## 当前基础与真正缺少的部分

现有 Data 生命周期已经提供一条可靠的最终边界：

```text
external data
  -> DataSource
  -> DataArtifactStore
  -> sealed DataArtifact
  -> DataBuilder
  -> runtime iterables
```

`DataArtifactStore` 已拥有 canonical identity、inventory、locator、lock、staging、quarantine、完整验证、
原子发布和 strict-resume identity 比较。managed object 的最终根只含 `data/`、`inventory/` 和
`manifest.json`，路径不依赖 cache root。这些都是便携 prepared dataset 应继承的能力。

当前缺口发生在最终发布之前和发布之后：

- managed build 每次创建随机 staging，producer 失败后全部清理，无法复用已完成分片；
- producer 返回后才整体扫描、排序和哈希输出，TB 级数据可能重复读取，数千万记录可能占用过多内存；
- 没有面向用户的 portable bundle inspect、verify、receive 或 adopt 流程；
- referenced artifact 的外部位置仍由用户管理，不能充当离线自包含的最终结果；
- loader 参数以 worker 和 batch 数为主，没有跨 worker/rank 的字节预算、层级等待和 cache 观测；
- checkpoint 可以在 epoch 边界重建顺序，但 `DataLoaders` 还不携带 mid-epoch reader cursor。

因此实现重点不是再造一个 `Dataset`，而是连接“可恢复准备—Store 最终发布—便携接收—有界读取”。

## 数据身份、运行视图和机器资源必须分开

设计中至少有三种摘要，不能共用一个模糊的 cache key。

### Prepared artifact identity

它回答“数据是什么”，至少绑定：

- 源快照或输入 inventory 的摘要；
- materializer/recipe 名称、版本和规范化参数；
- 影响结果的 codec、tokenizer、词表或其他外部资产摘要；
- 样本 schema、稳定 sample-ID 规则和总覆盖摘要；
- 确定性变换与 view policy；
- container/writer 版本、compression、storage dtype 和 shard layout；
- root manifest 到 shard/index manifest 的 digest tree；
- 总样本数、总存储字节数和最终 artifact digest。

它不包含 machine path、mtime、inode、cache root、workers、rank、NUMA、RAM/VRAM、queue depth 或
当前性能。追加或修改源数据产生新的 immutable artifact，不能就地改变旧 artifact。

### Runtime data view

它回答“一次 run 以什么顺序看到哪些样本”，可能绑定 artifact identity、split、run seed、epoch、
shuffle 算法及版本、rank/world-size、worker partition、drop/pad 和 global batch 约定。它属于 run 与
checkpoint 的恢复事实，不改变 prepared artifact。

### Execution profile

它回答“这台机器怎样读得快而且不超预算”，包含 locator、backend 私有参数、worker、CPU affinity、
NUMA、node-local cache、host cache、pinned queue、per-rank device queue 和保留空间。它可在 PC 和
服务器之间改变，不得影响样本 bytes 或 logical ordering。

## 可恢复 preparation workspace

workspace 是 Store 管理、producer 使用的未完成工作区。其稳定键来自 source、recipe 和输出规划
摘要，而不是一次进程的 UUID。最终 artifact 尚未发布时，workspace 可以跨进程存活；一旦发布，
它可按保留策略清理。

下面只是便于讨论的目录形状：

```text
preparation/<plan-digest>/
  plan.json                 # 不可修改的规范化计划
  lock                      # 单 writer job
  journal/                  # 已提交 work unit 的追加记录或独立描述
  shards/
    <stable-shard-id>       # 关闭并验证后的分片
  temporary/
    <unit-id>.<nonce>.tmp   # 永远不可被最终 reader 发现
  quarantine/               # 摘要不符或结构损坏的已提交候选
```

这不是第二个 artifact layout。workspace 中的 journal 属于失败恢复状态；最终发布仍生成当前 Store
拥有的 canonical `data/`、`inventory/` 与 `manifest.json`。

### Preparation plan digest

规范化计划至少包含：

| 事实 | 为什么必须固定 |
| --- | --- |
| source snapshot / input inventory digest | 防止一边准备一边混入新文件 |
| recipe/provider/version/resolved config | 防止更换变换后复用旧 shard |
| external asset digests | codec/tokenizer/词表变化会改变输出 |
| sample-key derivation version | concurrency 与文件枚举顺序不能改变身份 |
| output schema/format/writer version | reader 必须知道 bytes 的确切含义 |
| work-unit and shard assignment version | 干净运行与恢复运行必须得到相同布局 |
| deterministic transforms | 随机 epoch augmentation 不得被偷偷烘焙 |

扫描源树时使用规范化相对 logical key，不依赖文件系统枚举顺序或绝对路径。首版处理普通可变目录时，
必须先完整枚举并认证输入 snapshot，再开始转换；否则中断后无法证明剩余文件仍属于同一输入。若源
系统提供可信 immutable snapshot 或 provider checksum，可以免去第二次读取，但必须单独写明 issuer、
信任根和 freshness。普通目录的一次 inventory 不会冻结 bytes：每个 unit 实际读取时仍须对照 snapshot
复核 size/digest，替换、短读或变化一律 fail closed。以后若需要边读取边归并 input digest，只能采用
两阶段设计：discovery digest 仅定位 workspace，最终 source identity 由所有 unit 输入摘要归并而成；
恢复时逐个重新认证尚未完成的输入。

### Work unit transaction

每个 unit 的候选流程为：

```text
claim deterministic unit
  -> Store opens a temporary unit writer
  -> producer writes task-owned records through that writer
  -> Store flushes/fsyncs, hashes and closes
  -> validate schema, sample IDs and exact count
  -> Store atomically moves to stable shard path
  -> durably commit one journal descriptor
```

producer 不能提交自报摘要；文件句柄、bytes/hash、close、rename 和 descriptor 都由 Store 负责。记录为
完成以前的文件一律重做。恢复时重新核对 stable shard 的 size、count、digest 与路径；完好的 unit 跳过，
不符的 unit 先进入 quarantine 再重做。journal commit 以后发生的崩溃不能让 final manifest 提前出现。

首版优先使用每个 unit 一个 canonical、不可修改的 descriptor，并以校验后的临时文件替换；若采用
append log，每条记录必须带 framing、长度和 checksum，恢复时截断 torn tail。提交顺序还要写明文件
内容与父目录的 flush/fsync。workspace 与 stable shard 位于支持原子 rename 的同一文件系统；跨文件
系统 finalization 必须成为显式的复制—验证事务。`kill -9` 恢复与断电/系统崩溃耐久性分开测试，不能
只凭进程级故障注入宣称 power-loss safe。

首版使用一个 coordinator/writer job 和多个有界 worker。两个 coordinator 指向同一 plan/target 时，
后者必须明确失败。对象存储若没有可靠目录 rename，应写 versioned staging prefix，全部验证后最后
发布一个小型 canonical manifest/commit marker；不能把 fsspec 的 transaction 宣称为集群级原子提交。

### 分片大小

分片目标是 preparation policy，一旦发布便成为 artifact layout。选择需要同时测量：

- 较小 shard：更多文件、请求、索引、open/close 与 metadata 开销；
- 较大 shard：中断后重做更多、随机访问和 cache 淘汰更粗、单次损坏影响更大；
- 记录大小方差：只按 record count 可能产生严重 byte-size 倾斜；
- 传输与文件系统：对象存储、NVMe、共享文件系统的最优范围不同。

不能照搬某个库的默认值。若同一 prepared artifact 在 PC 和服务器上表现不同，只能改变 reader/profile
或发布新的 reshared artifact，不能在读取时悄悄重切同一 identity。

## Manifest tree 与有界 inventory

根 `manifest.json` 应保持小且 canonical。它绑定 shard/index 清单的摘要，而不是把十亿条 record 展开
成一个 JSON。候选层次如下：

```text
root manifest
  -> inventory shard descriptors
       -> stable shard id
       -> relative data path
       -> bytes / record count / SHA-256
       -> record-index digest
            -> stable sample IDs and task-owned metadata
```

Store 仍拥有唯一 canonical manifest。task/family 可以拥有 record schema 和 domain fields，但不能把关键
完整性事实藏进一个未经 Store 认证的私有 sidecar。TB 级首版预计需要新的 artifact schema 来表达
digest tree 和有界 inventory；被选中后的第一个设计决定就是验证这一点并同步规范，而不是无限扩张
现有 `domain` mapping 或把 schema 变化藏成内部优化。

writer 在 shard 关闭时产生 snapshot descriptor。这能避免 producer 自己重复计算相同摘要，但不能在
没有新信任模型的情况下替代 Store 当前发布前、发布后和 payload load 后的防篡改验证。finalization
只在以下条件都满足时继续：

- 每个计划 unit 恰好有一个已提交 descriptor；
- sample-ID union 等于计划覆盖，intersection 为空；不能把数十亿 ID 放进内存 `set`，首版采用按
  sample-ID 分区的 external sort/merge，或由 Store 认证的互斥 ID ranges；Bloom filter 只能预筛，不能
  单独证明没有重复或遗漏；
- descriptor 顺序 canonical，relative path 安全且 case-fold 后唯一；
- shard 自记录摘要以来没有改变；
- root/inventory 内存随 shard 或有界 merge block 增长，不随总 record 数无界增长。

显式 full verification 仍可重新读取所有 bytes。manifest verification 只核对 canonical structure、摘要树、
路径和 size 时，必须继续叫 manifest verification，不能把它包装成 full。

## Portable bundle、传输与 adopt

portable bundle 就是自包含的 managed artifact 内容，不是整个 Store namespace，也不包含 locator、lock、
workspace、quarantine 或 node cache。推荐目标流程：

```text
inspect source artifact
  -> copy into controlled partial receive area
  -> authenticate root manifest
  -> verify declared inventory and shards
  -> reject unexpected content
  -> adopt without rewriting identity
  -> publish local locator
  -> resolve through ordinary DataSource request
```

传输工具拥有它明确声明并经本 backend 验证的续传和重试。框架至少让工具跳过已完成 shard；是否能
从 shard 内部继续取决于 rsync/rclone 参数、multipart 实现和存储 backend。框架提供可供传输工具使用的
文件清单，并在最后验证内容。不能把
`rclone copy` 的成功返回当作 Store receipt；同样，Store 不应实现组织账户、凭据、带宽调度、备份或
跨区域复制产品。

adopt 的实现要尽量避免第二份 TB 级复制，也不能替原 producer 制造 locator：

- 同一支持原子 rename 的文件系统内，受控 receive area 可以在验证后 move，不产生第二份 TB 级副本；
- 跨文件系统时，传输规划应从一开始把目标设为 Store 的受控接收区；否则只能执行明确的复制—验证
  事务，并预先报告额外空间，不能同时承诺零复制和原子 move；
- 纳入对象后，新训练配置显式引用 artifact identity/manifest；兼容 `DataSource` 在当前请求中传入
  expected identity，Store 重新验证并签发 handle，`DataBuilder` 捕获当前 receipt；若以后需要独立
  imported-managed-artifact source，必须另行定义窄选择 contract，而不是伪造 producer locator key；
- 若选择“直接只读 portable root”而不是 adopt，必须仍由 Store 验证并签发 handle，且明确谁保证根不变；
- 任何失败都只留下 partial receive，不能创建可供 `require` 命中的 locator。

目标机不需要原始数据源，但仍需要安装兼容的 source/materializer 扩展和 recipe，让 payload loader 与
`DataBuilder` 能解释 artifact 的 domain/schema。source checkout 与 built wheel 验收要覆盖这一点。

## Strict resume 的验证成本是必须先决定的问题

当前 strict training resume 要求当前进程取得 full-verification receipt。几十 TB 数据每次启动重新哈希
可能需要数小时；但 mtime、inode、只读权限或“上次验证成功”都不能自动证明内容未变。

实施前只能显式选择一种保证：

1. 继续每次 full hash，并测量用户是否接受成本；
2. 为受支持的 immutable storage 定义可认证 attestation，例如 provider checksum 或受信内容寻址；
3. 同时支持 full rehash 与 trusted immutable-storage receipt，并在规范中写清不同安全假设。

任何选择都会触及 `SPEC.md` 和 `ARCHITECTURE.md`。attestation 还必须定义 issuer、信任根、freshness、
locator 绑定、撤销/失效，以及远端内容复制成本地 bytes 后是否继续有效。不能为了快而把 manifest+size
校验改名为 full，也不能让一次旧 receipt 跨进程或跨 locator 无条件复用。

## Runtime reader、shuffle 与 cursor

reader 先从 canonical sample plan 得到稳定逻辑记录，再应用运行视图。下面是可观察语义的说明，不是
要求 MDS、WebDataset、Arrow 等内部都实现同一条算法流水线：

```text
canonical sample plan
  -> deterministic epoch shuffle
  -> rank partition
  -> worker partition
  -> bounded decode/transform/collate
```

十亿样本不能靠 RAM 中完整 `randperm`。首个实现可采用两级有界 shuffle：确定性打乱 shard，再在一个
shard 或固定 block 内打乱记录；若使用 shuffle buffer，必须给出 byte limit、统计性质和恢复语义。

epoch-boundary resume 重新构造全部 cache 和 queue。若加入 mid-epoch resume，cursor 保存可以推导
“下一个逻辑 batch”的事实，而不是 pickle worker：

- epoch 与 shuffle seed/algorithm version；
- shard permutation position；
- shard 内 record/block position；
- rank/world-size 与 partition version；
- drop/pad policy、global batch contract 与 cursor schema version。

cursor 不能由 reader 单独发布。它必须与同一时刻的 model、optimizer、scheduler、scaler、global step
和 RNG 一起原子写入训练 checkpoint，只在完整 gradient-accumulation window 的 optimizer step 成功后
前移；checkpoint 发布失败时 cursor 也不能前移。已经预取但没有消费的 batch 只在恢复后重读。若随机
crop、posterior sampling 等运行时变换会影响最终 batch tensor，exact resume 还要求按 sample 派生的
stateless RNG，或把 reader/worker RNG 纳入同一 checkpoint。固定 topology 可以要求 exact resume；
world size、global batch 或 accumulation 改变时首版拒绝继续旧 run，但允许用同一 artifact 开始新 run。

## 每一层的 owner 和容量公式

| 层 | owner | 共同保证 |
| --- | --- | --- |
| Source snapshot / preparation semantics | task `DataSource` | 稳定 sample key、work-unit 语义、输出 schema |
| Workspace / inventory / verify / publish | `DataArtifactStore` | 有界 staging、锁、摘要、隔离、最终 identity |
| Shard reader / decode / transform / batch | task `DataBuilder` | 任务语义、sampler、collate、reader 私有 cache |
| Node-local shard cache | selected reader backend | 以内容摘要为 key 的可丢弃副本；不是 `DataArtifact`，不发布第二套 locator |
| OS page cache | operating system | 框架只观测，不复制竞争性全局 file cache |
| Pinned feeder | loader/runtime | 有界 batch queue、backpressure、源 buffer 生命周期 |
| Device prefetch | training runtime | copy stream、event、per-rank budget、消费顺序 |
| Rank/world size/process failure | distributed runtime | 启动、拓扑输入、共同失败；不猜数据 schema |

候选预检公式：

```text
host node peak =
    coordinator/index overhead
  + ranks * workers-per-rank * worker-private budget
  + ranks * host-prefetch budget
  + ranks * pinned-queue budget
  + reserved OS/application headroom

device peak per rank =
    model + optimizer + activations
  + transfer buffers
  + bounded device-prefetch queue

temporary preparation disk =
    completed reusable shards
  + maximum in-flight unit output
  + final inventory overhead
  + explicit safety headroom
```

这里的 worker-private budget 是单个 worker 的上限。普通 `DataLoader` 首版用可靠的
`max_batch_bytes × prefetch_factor × workers-per-rank × ranks` 做保守预检；它本身不能对可变 batch
兑现精确字节硬限制。若 workload 要求该保证，runtime-owned byte-aware feeder 才是 pinned 队列 owner，
并负责回压和 buffer 生命周期；device prefetch 也只能建立在这个边界上。估算不取代 runtime hard
limit。多进程共享 cache 需要明确唯一 owner、并发和淘汰；没有证据时使用 OS page cache 和独立有界
reader，不先实现跨进程 decoded cache。

## 成熟实现的适用位置

下表比较的是可以叠加的不同轴，不是互斥的整套 provider 排行。一个实现可以同时采用 Parquet 格式、
fsspec locator、Ray preparation executor 和任务自己的训练 reader；也可以使用 MDS 格式与 Mosaic reader。

| 系统 | 可复用能力 | 不能替代的责任 | 采用条件 |
| --- | --- | --- | --- |
| PyTorch DataLoader | worker、batch、pinned memory、prefetch | artifact、字节总预算、prepare resume | 基础路径继续保留 |
| MosaicML Streaming/MDS | shard index/hash、本地/远端、LRU cache、predownload、partition；`StreamingDataLoader` 可保存 mid-epoch state | `MDSWriter` 覆盖输出不是 shard-level preparation resume；不拥有 Store identity | 首选训练 reader PoC；固定 global batch/canonical nodes 等 elastic 条件需另验 |
| WebDataset/WIDS | tar shards、streaming、node/worker split、disk/memory LRU | 通用 schema、严格 cursor/Store publication | 媒体样本天然适合 tar 时 |
| HF Datasets / Arrow IPC / Parquet | HF transform/cache、Arrow mmap/快速 reload、Parquet 压缩列式存储与扫描；IterableDataset 有有限 cursor | fingerprint 不是来源或完整性证明；shuffle buffer/部分 batched map 恢复并非 exact | 按 runtime、快速本地加载或长期存储分别选择，不能把三者混成同一格式 |
| Ray Data | streaming execution、backpressure、CPU/GPU batch transform、cluster executor，并可向 PyTorch/Ray Train 提供 batch | 不拥有 Store artifact identity/publication/strict cursor；checkpoint API 仍为 beta | 单机因容量、数据位置或吞吐不可行时可选；首版不作为 maintained reader |
| fsspec | storage locator、backend adapter、whole-file/block cache | transaction 非集群级 publication truth | 某个选定 provider 需要时 |
| StatefulDataLoader | loader state_dict、worker state、mid-epoch candidate | preparation/transfer resume；IterableDataset 恢复要求兼容 worker topology | beta 语义与 same-num-workers 等限制通过选定任务验证时 |
| NVIDIA DALI | decode、augmentation、受支持 reader/operator graph 的 checkpoint、GPU prefetch | manifest、数据身份、通用 storage cache；任意 external source、GPU dataset 或不同 pipeline 不能假定 exact resume | decode/transform 实测瓶颈且 pipeline identity 固定时 |
| GDS/cuFile/KvikIO | 合适环境的 storage-to-GPU transport | decode、shuffle、cache identity、resume | 拓扑受支持且 I/O/H2D 实测瓶颈时 |

首轮 PoC 必须覆盖 Windows 原生或 WSL 与 Linux server，不能只因 Linux benchmark 优秀就锁定一个在 PC
不可用的强制依赖。外部依赖的 version/license/wheel/maintenance 在方向被选中后重新核对。

对适合随机访问、固定大小 tensor 或列式记录的任务，mmap-friendly layout 是候选；它不是所有媒体和
streaming 格式的强制要求。如果原始数据已经是兼容、不可变、可移植且可认证的分片表示，应优先验证
并采用这些 bytes，而不是为了统一外观逐样本重写、额外占用一份 TB 级空间并破坏已有 streaming 优势。

## 故障矩阵

| 故障点 | 预期行为 |
| --- | --- |
| 扫描源数据时退出 | 没有 shard commit；同 plan 可重新扫描，源摘要变化则拒绝旧 workspace |
| 写临时 shard 时 kill | 临时文件不被信任；重启后删除或覆盖，只重做该 unit |
| shard 关闭后、journal commit 前退出 | 重启核对候选；可按确定规则重新提交或重做，不产生 duplicate |
| journal commit 后退出 | descriptor 与 stable shard 完好则跳过；不提前发布 final manifest |
| finalization 中退出 | final locator 不可见；恢复后从完整 committed set 重做 finalization |
| disk full | 停止调度新 unit，保留已提交进度，报告所需 headroom，不发布半成品 |
| committed shard 被篡改 | quarantine 该 shard 并重做；若源不可用则明确失败 |
| source/recipe/writer version 改变 | plan digest 不同，旧 journal 不得继续 |
| 复制中断 | 目标只有 partial receive；传输工具续传，Store locator 仍不可见 |
| 缺失、额外、短读、同尺寸篡改 | receive verification 失败；不得 adopt |
| node-local cache 损坏 | 按 canonical digest 重新取回，或在 canonical 来源不可用时失败 |
| worker 失败 | 首版单卡 operation 明确失败，不静默跳样；rank 共同失败留给 Distributed 阶段 |
| pinned/device queue OOM | 预算预检或 hard limit 明确失败/降到已验证 profile，不无界重试 |

## 验证与 benchmark 设计

### CI 中的语义测试

- 使用小 shards 在每个 transaction 边界故障注入；
- clean 与 resumed preparation 的 bytes、shard descriptors、manifest、artifact identity 完全相同；
- source、recipe、external asset 或 assignment version 改变拒绝旧 journal；
- 数百万 synthetic records 的 inventory build 内存保持有界；
- portable path 拒绝 absolute、`..`、UNC/drive、reserved name、case-fold collision、link/reparse；
- partial receive 不可 resolve；同文件系统 adopt 不做第二份 TB 级复制并保持 identity，跨文件系统按
  传输规划或显式复制—验证事务验收；
- missing/extra/short-read/same-size mutation 都失败；
- opaque custom payload/第二种 task family 证明没有通用图像 schema；
- epoch resume 重建 queue；若有 cursor，则 overflow/skipped step 不提交 cursor；
- 单卡 fixed topology 的顺序、drop/pad 与失败传播；Distributed 阶段另验全局 sample union/intersection；
- source checkout 与 built wheel 的 preparation/inspect/verify/adopt 行为一致。

### 真实 PC 与服务器验收

记录 storage、filesystem、CPU sockets、NUMA、RAM、GPU、PCIe/NVLink、driver、CUDA/PyTorch、reader 与
writer 版本。先在 PC 准备并 kill/restart，再部分复制到服务器、续传、verify/adopt。两边用同一 artifact
跑相同逻辑样本 bytes 与 sample-ID 测试，再分别选择资源 profile。epoch 顺序属于运行视图；随机变换后的
最终 batch tensor 只有采用相同的 stateless transform/RNG 契约时才要求完全相同，不能把三种保证混写。

Preparation 报告：scan/read/decode/transform/compress/write/hash 时间，input/output bytes/s、records/s，
shard size/count 分布，journal progress/rework/quarantine，peak RAM/temp disk/final ratio。

Runtime 每 rank 报告：useful storage bytes/s、samples/s、read/decode/transform/collate/loader wait/H2D/compute，
host-cache hit/miss/evict bytes，普通/pinned/per-GPU memory peak，queue occupancy/empty/full/backpressure，
covered/dropped/padded/duplicated sample counts。

冷 cache 必须有可重复的清理或新 locator 条件；没有权限真正清空 OS cache 时，报告为“首次 reader
访问”而不是冷 cache。每增加一层都与前一层相同 workload 对照；收益消失后不再实施更复杂优化。

TB 级与八卡结果属于有记录的硬件验收，不伪装成普通 pytest。性能阈值在真实 baseline 后写下，至少
包含最低吞吐或最大 feed wait、最大内存、允许 preparation/verification 开销和扩展效率。

## 被选中后的实现顺序

这不是当前排期，而是在根路线图选中本方向后建议的依赖顺序：

1. 选定真实 finite workload、format、PC/server storage 和 strict-verification 决策，并决定新的 artifact
   schema 怎样表达 digest tree 与有界 inventory；
2. 实现 resumable workspace、work-unit commit、bounded inventory 和最终 Store publication；
3. 实现 portable inspect/verify/receive/adopt，并完成跨绝对路径 identity 验收；
4. 接入一个成熟 reader，完成 PC/server 单卡 profile、epoch resume 和观测；
5. 若 epoch 时长需要，验证并实现稳定 mid-epoch cursor；
6. Distributed 提供固定 rank/world-size 后完成八卡 exact coverage；
7. 仅按测量选择 Ray、decoded cache、DALI、GDS 或更深 device prefetch。

任何阶段都不建立 universal `Dataset`、DataLoader、sampler、reader、batch 或 storage-service registry；任务
仍通过 `DataSource`/`DataBuilder` 解释自己的数据。

## 实施前必须作出的决定

- 第一份真实 workload、源快照方式、规模、schema 和成熟 storage format；
- 新训练配置怎样显式选择已 adopt 的 artifact identity/manifest，并由兼容 `DataSource` 在当前请求中
  取得 full-verification receipt；
- preparation resume、epoch resume、mid-epoch resume 分别承诺到哪里；
- portable bundle 采用 verify+adopt，还是允许受控只读根直接运行；
- TB 级 strict full verification 的成本与可信 immutable-storage attestation 边界；
- 新 artifact schema 怎样表示 digest tree、有界 inventory、版本升级与旧 artifact 读取边界；
- stable sample ID、shuffle、partition、cursor 的算法和版本化；
- 首版是否只支持固定 world size；
- PC/server profile 的显式字节预算、headroom 与 NUMA 记录；
- workspace 保留期限、显式清理、quarantine 与 disk-full 行为；
- Windows/WSL/Linux 的首轮支持矩阵。

## 参考资料

- [MosaicML Streaming dataset format](https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/dataset_format.html)
- [MosaicML parallel conversion](https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/parallel_dataset_conversion.html)
- [MDSWriter](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html)
- [MosaicML shard retrieval](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/shard_retrieval.html)
- [MosaicML fast resumption](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/fast_resumption.html)
- [MosaicML elastic determinism](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/elastic_determinism.html)
- [WebDataset/WIDS](https://webdataset.github.io/webdataset/webdataset/)
- [Hugging Face Datasets processing](https://huggingface.co/docs/datasets/process)
- [Hugging Face cache fingerprints](https://huggingface.co/docs/datasets/v3.3.2/en/about_cache)
- [Hugging Face map-style and iterable resume](https://huggingface.co/docs/datasets/v2.20.0/about_mapstyle_vs_iterable)
- [Apache Arrow memory and memory mapping](https://arrow.apache.org/docs/python/memory.html)
- [Ray Data checkpointing](https://docs.ray.io/en/latest/data/api/doc/ray.data.checkpoint.interfaces.CheckpointConfig.html)
- [Ray Data performance](https://docs.ray.io/en/latest/data/performance-tips.html)
- [Ray Data PyTorch batch iteration](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.iter_torch_batches.html)
- [fsspec features and transactions](https://filesystem-spec.readthedocs.io/en/latest/features.html)
- [PyTorch StatefulDataLoader](https://github.com/pytorch/data)
- [PyTorch stateful data loading tutorial](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
- [NVIDIA DALI checkpoint constraints](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics_checkpointing.html)
- [NVIDIA DALI performance tuning](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/advanced_topics_performance_tuning.html)
- [NVIDIA GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/)
- [rclone copy](https://rclone.org/commands/rclone_copy/)
- [Amazon S3 multipart checksums](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)
