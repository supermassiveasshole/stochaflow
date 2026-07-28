# 数据构建

`data` 只选择一个注册的 `DataBuilder`。核心不会根据 YAML 重新组装 Dataset、split、
Sampler、collate 或 DataLoader，而是一次性接收 builder 已经组装好的 loader：

```mermaid
flowchart LR
    A["data.name / data.params"] --> B["data_builders Registry"]
    B --> C["DataBuilder.build()"]
    C --> D["DataSource.materialize()（source-aware recipe）"]
    D --> E["DataArtifact[payload]"]
    E --> V["Builder 验证 binding 与 accepted contract"]
    V --> F["recipe-private partition / Dataset / transform / sampler / loader"]
    F --> G["DataLoaders"]
    G --> H["Runner / Trainer"]
```

```python
@dataclass(frozen=True, slots=True)
class DataLoaders:
    train: Iterable[Any]
    validation: Iterable[Any] | None = None
    test: Iterable[Any] | None = None
    steps_per_epoch: int | None = None
    artifact_bindings: DataArtifactBindings | None = None
```

loader 必须可重复迭代，不能直接返回一次性的 generator/iterator。训练 loader 没有
可用的 `len()` 时，builder 必须显式提供正数 `steps_per_epoch`；有长度时可省略。核心
通过实际调用 `len()` 探测能力，并把 `TypeError` 解释为未知长度，因此底层
`IterableDataset` 没有长度的标准 PyTorch `DataLoader` 可以配合显式步数使用。
`--limit-batches` 会在最终训练步数上取更小值。validation 和 test 不要求长度，CLI
limit 仍可限制无限或未知长度的可重复 iterable。

## `image` recipe

`image` 提供常见的单源图像训练组合：

```yaml
data:
  name: image
  params:
    source:
      name: torchvision
      params:
        dataset: CIFAR10
      materialization:
        cache_root: ./data
        policy: ensure
        verification: manifest
        verification_workers: null
    partition:
      mode: holdout
      validation_size: 5000
    image:
      size: [32, 32]
      channels: 3
      normalize: true
      random_horizontal_flip: true
    loader:
      batch_size: 128
      num_workers: 4
      shuffle: true
      drop_last: true
      pin_memory: false
      persistent_workers: true
      prefetch_factor: null
      steps_per_epoch: auto
```

`verification_workers: null` 自动选择 `min(8, logical CPUs)` 个 artifact 哈希线程；
也可以配置 `1..8` 范围内的整数。训练 CLI 的 `--artifact-verification-workers N`
仅覆盖本次运行，包括 strict resume，不改变 artifact identity 或 digest。

`source` 只有一种规范结构：`name` 选择已注册的 `ImageDataSource`，`params` 是该
source 的私有参数，`materialization` 声明缓存、获取策略与验证强度。
`materialization` 是必填 mapping；其中字段省略时依次使用
`cache_root: ./.stochaflow-cache`、
`policy: ensure` 和 `verification: full`。持久化实验配置建议显式写出三者，避免工作目录
改变缓存位置。内置 `torchvision` 支持 MNIST、CIFAR10 和 Flowers102；
`policy: ensure` 允许在 exact managed artifact 不存在时下载并发布，
`policy: require` 则只接受已有的 exact artifact。`verification` 可选 `manifest` 或
`full`；strict resume 会强制完整验证。

所有 DataSource 都通过 framework `DataArtifactStore` 使用同一个 schema-v2 cache
lifecycle。`managed` source 把实际数据发布到 cache；`referenced` source 只发布
inventory/sidecar，外部数据不复制。两者返回同一个 `DataArtifact` runtime handle，
ownership strategy 记录在 `artifact.identity.kind`。source 不应自行维护 manifest、
identity、current pointer、lock 或 publication 状态机。

schema-v2 cache 固定在
`<cache_root>/data-artifacts/v2/<kind>/<artifact-type-digest>/` 下，并包含 immutable
objects、locators、locks、staging 与 quarantine。`require` 是完全只读的；`ensure`
才会在缺失或已持久化内容验证失败时构建、隔离和修复。旧 cache layout、manifest、
locator、identity 与 checkpoint binding 不受支持，也不会自动迁移；升级后应重新
materialize 并创建新 run。

现有本地/NFS 图片目录使用 `image_folder`，它创建 referenced artifact：缓存中只保存
canonical manifest 和分片 inventory，不复制原图。第一次 `ensure` 会完整枚举并读取
每个文件，以一次内容读取记录规范路径、字节数、SHA-256 和图片宽高。宽高与内容摘要
一起进入 canonical inventory 和 artifact identity，后续 resolution bucket 不需要再次
解码图片来推断尺寸。`manifest` 校验索引、路径和大小，`full` 还会重新计算全部内容哈希
并核对尺寸。Dataset 只按 manifest 的固定顺序读取，并在解码同一份字节前校验 SHA-256，
因此外部文件被替换时会 fail-stop；但外部目录被删除后，Stochaflow 无法仅凭 reference
artifact 重建图片。示例：

```yaml
source:
  name: image_folder
  params:
    root: G:/datasets/images
    layout: flat
  materialization:
    cache_root: ./.stochaflow-cache
    policy: ensure
    verification: full
```

当前内置 image recipes 不支持 Hugging Face streaming。streaming 需要
`IterableDataset`、shuffle buffer 与 iterator-state resume 的独立生命周期；需要该
能力的项目应提供自己的 `DataBuilder`，并明确其恢复契约。

partition 是这个 recipe 的私有能力：

| mode | 行为 |
| --- | --- |
| `none` | 完整训练集，不额外创建 validation。 |
| `official` | 使用 source 提供的原生 train/validation/test。 |
| `holdout` | 从有限、可索引训练集确定性划分 validation。 |
| `kfold` | 使用指定的 `num_folds` 和 `fold_index` 构建一个 fold。 |

K-fold 配置只代表一次独立运行；需要五折时执行五次配置或由外部 sweep 展开。默认
batch 为 `(images, {})`。`loader.pin_memory` 的可移植默认值为 `false`；CUDA 用户可在
测量吞吐后显式开启，MPS 用户应保持关闭。

## `class_labeled_image` recipe

`class_labeled_image` 消费 DataSource 发布的标准
`ClassLabeledImageFolderArtifactPayload`。payload 必须认证连续的
`class_mapping`（label 为 `0..N-1`）和每条记录的 `class_label`；Builder 不从目录名
猜测标签，也不读取 source 私有参数。该 payload 类型可以表达 native validation，但
当前 Builder 的 accepted contract 更窄：

- finite、authenticated、map-style train inventory；
- 连续 class mapping 和逐样本 class label；
- `validation is None`；
- optional official test；
- 使用正数 `validation_per_class` 从 train 派生 holdout。

配置示例：

```yaml
data:
  name: class_labeled_image
  params:
    source:
      name: my-project.animals
      params: {}
      materialization:
        cache_root: ./.stochaflow-cache
        policy: require
        verification: full
    partition:
      validation_per_class: 300
      seed: stable-validation-v1
    image:
      size: [128, 128]
      channels: 3
      normalize: true
      random_horizontal_flip: true
    loader:
      batch_size: 8
      num_workers: 2
      shuffle: true
      drop_last: true
      pin_memory: true
      persistent_workers: true
      prefetch_factor: 4
      steps_per_epoch: auto
```

Builder 从 source 的 train inventory 中按 `(partition seed, authenticated sample
identity)` 每类精确保留 `validation_per_class` 条，并保证每类至少剩一条训练记录。
`partition.seed` 可为整数或非空字符串；省略时使用 experiment seed。当前 recipe 要求
source 不提供原生 validation，test 可选。默认 batch 为
`(images, {"class_label": labels})`，其中 labels 是 `torch.long`。

训练 sampler 总是把 epoch 随 index 传给 Dataset。crop 与 flip 使用由 run seed、epoch
和认证 sample identity 派生的无状态随机值，因此 `num_workers=0`、persistent workers
和 epoch-boundary strict resume 会产生相同的 batch Tensor。新来源只有既能 materialize
为上述完整 accepted artifact contract，又需要同样的 derived holdout、transform、
sampler、loader、resume 和 batch 语义时，才只需注册 `ImageDataSource`。相同 payload
type 不自动代表 recipe 兼容。

## `super_resolution` recipe

这个 recipe **只构建数据**：它把每个 batch 组织为
`(high_res, {"low_res": low_res})`，不提供开箱即用的端到端超分辨率训练或采样。
内置 `gaussian_denoising` TrainingBuilder 只接受空 condition mapping，因此不能直接消费
这里的 `low_res`。条件 Gaussian 超分辨率还需要项目自己提供 condition-aware model、
TrainingBuilder/TrainingStrategy 和 SamplingBuilder；它们可以继续复用内置离散 Gaussian
Process 以及 DDPM/DDIM。完整组合见
[条件 Gaussian 超分辨率教程](../tutorials/super-resolution.md)。

在线 bicubic 模式从 HR 图像生成 LR condition：

```yaml
data:
  name: super_resolution
  params:
    source:
      name: image_folder
      params:
        root: ./data/hr
        layout: flat
      materialization:
        cache_root: ./.stochaflow-cache
        policy: ensure
        verification: full
    partition:
      mode: holdout
      validation_size: 0.1
    image:
      high_resolution: [256, 256]
      low_resolution: [64, 64]
      channels: 3
      normalize: true
      random_horizontal_flip: true
    low_resolution:
      kind: bicubic
    loader:
      batch_size: 16
      num_workers: 4
      steps_per_epoch: auto
```

`bicubic` 模式要求 LR 的高度和宽度均不大于对应 HR 维度。resize
结果会在归一化前截断到 `[0, 1]`，因此启用 `normalize` 后 LR condition
始终保持在 `[-1, 1]`。

在线模式先对 HR 做对齐 crop 和可选 flip，再生成 LR。配对模式使用以下 source，并将
`low_resolution.kind` 设为 `paired`：

```yaml
source:
  name: paired_image_folders
  params:
    high_resolution_root: ./data/hr
    low_resolution_root: ./data/lr
    layout: flat
  materialization:
    cache_root: ./.stochaflow-cache
    policy: ensure
    verification: full
```

文件按规范相对路径和 stem 稳定匹配；重复、缺失和大小写冲突都会被拒绝。LR/HR 共享
几何变换并校验整数尺度。默认 batch 为
`(high_res, {"low_res": low_res})`。

(multi-resolution-image-recipe)=
## `multi_resolution_image` recipe

高级图像 recipe 保留多 source 权重、分辨率 bucket、同 bucket batch、动态像素预算和
确定性 `set_epoch`：

```yaml
data:
  name: multi_resolution_image
  params:
    sources:
      - id: digits
        sampling_weight: 0.4
        source:
          name: torchvision
          params:
            dataset: MNIST
          materialization:
            cache_root: ./.stochaflow-cache
            policy: ensure
            verification: manifest
      - id: flowers
        sampling_weight: 0.6
        source:
          name: image_folder
          params:
            root: ./data/flowers
            layout: flat
          materialization:
            cache_root: ./.stochaflow-cache
            policy: ensure
            verification: full
    image:
      channels: 3
      normalize: true
      random_horizontal_flip: true
    batching:
      buckets:
        - {name: square_32, height: 32, width: 32}
        - {name: square_64, height: 64, width: 64}
      base_bucket: square_64
      dynamic_batch_size: true
    partition:
      mode: holdout
      validation_size: 0.1
    loader:
      batch_size: 64
      steps_per_epoch: auto
```

bucket metadata、source id、采样索引和 partition helper 都是 recipe 私有实现，不属于
扩展 API。动态 batch size 将 `base_bucket` 的像素量作为基础预算：

$$
B_b = \max\left(1, \left\lfloor B_0
\frac{H_{base}W_{base}}{H_bW_b}\right\rfloor\right)
$$

`sampling_weight` 若启用，所有 source 都必须提供有限正数；NaN 和 Infinity 会在配置
边界被拒绝。reference source 的 bucket 直接读取 identity 认证的 inventory 宽高；
torchvision managed artifact 在首次物化时生成并认证尺寸 sidecar。构造
`MultiResolutionDataset` 不调用 `dataset[index]`，因此不会在 DataLoader worker 启动前
重复读取、哈希或解码全量图片。运行期只保留紧凑的 source/bucket 数值 code；batch
sampler 根据当前是否启用加权混合，只构造 bucket 索引或 `(source, bucket)` 索引中的一种。

## Epoch 与恢复边界

启用训练 shuffle 时，`image`/`super_resolution` 使用 recipe 私有的 epoch-aware
sampler；关闭 shuffle 时按 Dataset 顺序读取，不构造该 sampler。
`class_labeled_image` 无论是否 shuffle 都通过 sampler 把 epoch 传入无状态增强，
`multi_resolution_image` 使用自己的 `set_epoch()` batch sampler。因此在相应 sampler
实际启用、数据与配置不变时，strict resume 可以重建对应 epoch 的索引/batch 顺序。

每个 source 的完整 artifact identity 会按 source id 排序后写入 run manifest 与
checkpoint。strict resume 在构建数据前注入 checkpoint 中的 expected bindings；任何
缺失、source/materialization/content/manifest identity 不一致都会在恢复 model、
optimizer 或 scheduler 之前失败。runtime partition 与其他 data recipe policy 不属于
artifact identity，由 checkpoint 的 resolved config 与 experiment seed 固定。managed
artifact 可从固定来源与 materialization recipe 重新构建；referenced artifact 只保证对
当前外部字节 fail-stop，不宣称拥有或版本化外部目录。

POSIX 路径读取与缓存变更使用从文件系统根逐级 no-follow 的 descriptor-relative 操作。
Windows 会拒绝 symlink/junction/reparse point，并在操作前后复核祖先与目标 identity；
但 Windows 的发布、隔离和递归删除仍受限于 pathname API。若另一个进程同时拥有缓存目录
写权限，框架只能在检测到替换时 fail-stop，不能承诺像 POSIX `openat`/`renameat2` 那样
对错误目标绝对零副作用。生产缓存应通过 ACL 限制为训练账户独占写入。

checkpoint 不保存 DataLoader iterator、worker 或 transform 的随机状态。使用
`image`、`super_resolution` 或 `multi_resolution_image` 中 worker-local 的随机
crop/flip，不保证与不中断运行产生逐位相同的 batch Tensor；
`class_labeled_image` 已内置 stateless `(seed, epoch, sample identity)` 增强。其他
需要逐位恢复保证的任务也应采用无 worker 随机增强、无状态增强，或在自定义
DataBuilder 中明确自己的恢复契约。

## 选择扩展路径

| 需求 | 接入方式 |
| --- | --- |
| 新来源满足现有 Builder 的完整 artifact contract，且所需 runtime recipe 完全一致 | 只实现并注册 DataSource |
| 一次性 Python 实验，Dataset/DataLoader 已经组合完成 | 直接调用 `Trainer.fit(train_iterable, num_epochs=10, validation_dataloader=...)` |
| 新的 partition、Dataset、sampler、streaming、resume 或 batch 语义 | 注册一个 recipe-level DataBuilder |

DataBuilder 对应可配置、可重建的一份 runtime recipe，不与 dataset 名称、DataSource 或
具体 Dataset 类一一对应。

## 自定义 DataBuilder

复杂数据逻辑应直接使用 Python 和 PyTorch，而不是扩展一份通用 YAML 拓扑：

```python
from torch.utils.data import DataLoader

from stochaflow.extensions import (
    DataBuilder,
    DataLoaders,
    REGISTRIES,
)


@REGISTRIES.data_builders.register("my-project.physics-data")
class PhysicsDataBuilder(DataBuilder):
    def build(self) -> DataLoaders:
        dataset = StreamingPhysicsDataset(**self.context.params)
        loader = DataLoader(
            dataset,
            sampler=PhysicsSampler(dataset),
            collate_fn=physics_collate,
        )
        return DataLoaders(train=loader, steps_per_epoch=1000)
```

```yaml
extensions:
  plugins: [my-project]
data:
  name: my-project.physics-data
  params:
    path: data/simulation.zarr
```

Trainer 递归迁移 structured batch 中的 Tensor，但不解释 state、target、condition 或
模型签名。batch 与训练语义的适配属于 TrainingStrategy。完整注册规则见
[扩展与 Registry](extensions.md)。
