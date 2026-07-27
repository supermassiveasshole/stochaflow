# Framework Capability Proposal: Artifact Metadata, Provenance and Capacity Model

- 文档性质：开发草案；不属于当前公开 API 或正式用户文档
- 状态：提案，尚未进入实现
- 制定日期：2026-07-27
- 主要输入：当前 AFHQ-v2 showcase、core data artifact、run manifest、
  checkpoint、sampling manifest 与 Physics capacity 工具

## 1. Motivation

AFHQ-v2 showcase 已经证明：一个可审计、可恢复、可评估的 dataset workflow，
不能只返回一个目录或一个 PyTorch `Dataset`。它还需要回答：

- 数据集和 artifact 是什么；
- source 在哪里，使用了哪个 checksum/version；
- artifact 由哪些输入和 transformation 产生；
- materialization 使用了什么代码和参数；
- resolved config、seed、extension 与运行环境是什么；
- artifact 有多大，某个 workload 预计或实际消耗多少资源。

这些问题并不属于 AFHQ-v2 的猫、狗、野生动物分类语义。ImageNet、physics field、
LLM corpus 也会遇到同样的问题。如果每个 example 都自行定义 source lock、
manifest、canonical hash、code identity、environment report、resource report 和
result sidecar，Stochaflow 会得到多套相似但不兼容的事实模型。

因此，答案是否定的：dataset identity、artifact metadata、source provenance、
transformation history、reproducibility information、resource evidence 和
configuration tracking 不应由每个 example 重复实现。

但这不意味着 core 应立即设计一个覆盖所有数据、模型、checkpoint、指标仓库和远程
catalog 的“宇宙级 Artifact 系统”。当前已有真实重复模式，足以抽取一个较小的公共层：

1. 可移植的 `ArtifactDescriptor`；
2. 描述一次 invocation 的 `ExecutionRecord`；
3. 区分 artifact footprint、resource estimate 和 resource observation 的资源证据模型。

这三个能力应建立在现有 `DataArtifactIdentity`、`DataArtifactBindings`、resolved
config、extension provenance 和 checkpoint lineage 之上，而不是替换它们。

## 2. Current Problems

### 2.1 当前已有的 framework 基础

本提案不是从空白开始。当前 core 已经拥有：

| 当前能力 | 位置 | 已解决的问题 |
| --- | --- | --- |
| `DataArtifactIdentity` | `src/stochaflow/data/artifacts.py` | location-independent data identity；source、materializer、content 与 manifest digest |
| managed/referenced artifact | `src/stochaflow/data/artifacts.py` | 区分 framework-owned content 与 external referenced content |
| `DataArtifactBindings` | `src/stochaflow/data/artifacts.py` | 将稳定 artifact identity 绑定到 run role，并支持 strict resume |
| safe artifact I/O、locking、publication | `src/stochaflow/data/artifact_io.py`、`artifact_store.py` | link-safe I/O、canonical serialization、materialization lock、atomic cache publication |
| resolved config | `src/stochaflow/utils/config.py`、`scripts/experiment_runner.py` | 保存训练使用的完整 resolved config |
| run/checkpoint lineage | `scripts/experiment_runner.py`、`utils/checkpoint.py` | config source、overlay history、resume lineage、runtime options |
| extension provenance | `src/stochaflow/utils/plugins.py`、`utils/run_manifest.py` | entry-point、distribution、version、target 与显式 version acceptance |
| sampling manifest | `src/stochaflow/sampling/runtime.py` | checkpoint、resolved sampling config、selected components、output paths |

其中 `DataArtifactIdentity` 和 `DataArtifactBindings` 已经是正确的严格恢复边界。
它们不应因为新增 metadata 而被废弃。

当前缺少的是 identity 以外的公共描述能力：

- `DataArtifactIdentity` 是 equality/resume contract，不是完整的 dataset catalog
  record；
- `DataLoaders` 只向 runner 返回 identities，没有返回可持久化的 descriptor；
- core run manifest 保存 resolved config 和 extension provenance，但没有统一的
  config digest、software/source digest、environment snapshot 或 artifact
  descriptors；
- capacity、evaluation 和 preparation 各自定义 report/manifest schema。

### 2.2 AFHQ-v2 中的 domain-specific logic

以下逻辑应继续由 AFHQ-v2 extension 拥有，不应进入通用 core：

| Domain-specific concern | 当前实现示例 | 保留在 example 的理由 |
| --- | --- | --- |
| 官方 source 位置和发布约束 | `resources/afhq-v2.lock.yaml` | URL、archive 名称、官方发布状态属于 AFHQ |
| AFHQ source contract | `_preparation/contracts.py`、`source_lock.py` | `cat/dog/wild`、class mapping、split/count、512 RGB PNG 是 dataset 语义 |
| ZIP layout 与成员检查 | `_preparation/archive.py` | archive tree 和合法 member 规则属于该 source |
| 图片 decode/resize/encode | `_preparation/image_transform.py` | Pillow、RGB、Lanczos、PNG encoding 是具体 transformation |
| prepared image path layout | `_preparation/publication.py` | `train/class/file.png` 是该 artifact type 的布局 |
| 标准 payload 适配 | `artifact.py`、`stochaflow_ext/source.py` | DataSource 将 AFHQ source contract 映射为认证的 class-labeled artifact |
| validation recipe 参数 | production/smoke YAML | 每类保留 300 张是这个训练 recipe 的选择；通用分层算法由内置 Builder 提供 |
| AFHQ evaluation protocol | `tools/evaluation_config.py`、`evaluation_metrics.py` | class allocation、FID/KID scope 和 metric protocol 是 showcase 规则 |
| capacity workload matrix | `tools/capacity_config.py`、`capacity_trials.py` | micro-batch、precision、warmup、measured update 和 effective batch policy 是该 benchmark 设计 |

“具体 transformation 属于 domain”与“transformation history 属于 framework”
并不冲突。Core 不实现 Lanczos resize，但应提供一个标准 record，让 AFHQ 声明它执行了
哪个 transformation、使用了什么版本和参数。

同理，Core 不决定 AFHQ 的 source URL 或 checksum，但应提供标准 `SourceReference`
保存这些事实。

### 2.3 AFHQ-v2 中重复维护的 framework concerns

当前 showcase 自行维护了以下 framework-level concerns：

| Framework concern | 当前 AFHQ 实现 | 与已有/未来代码的重复 |
| --- | --- | --- |
| canonical JSON 与 digest | `_preparation/identity.py`、`tools/capacity_provenance.py` | core `artifact_store.py` 已有另一套 canonical JSON/digest |
| source identity 与 lock schema | `_preparation/contracts.py`、`source_lock.py` | ImageNet、physics、LLM source 都需要 name/version/URI/digest/license |
| dataset metadata | `source_lock.py`、`publication.py` | name、version、homepage、license、citation、schema/count 被写入 AFHQ manifest |
| transformation recipe history | `_preparation/planning.py`、`publication.py` | recipe ID/version、library version、parameters 和 recipe digest 是通用 lineage 事实 |
| prepared manifest schema | `_preparation/publication.py`、`prepared_artifact.py` | schema version、source、preparation、counts、inventory、artifact digest 被本地定义和解析 |
| code identity | `tools/capacity_provenance.py` | core/extension distribution version 和 Python tree hash 不是 capacity 特有 |
| resolved config identity | `tools/capacity_provenance.py` | core 已保存 resolved config，但 capacity 又自行计算 canonical SHA-256 |
| environment snapshot | `tools/capacity_provenance.py` | platform/Python/Torch/CUDA/cuDNN/device 是通用 execution provenance |
| dependency versions | `tools/evaluation_result.py` | 与 capacity code identity、extension provenance 部分重叠 |
| input/output file records | `tools/evaluation_result.py` | path/bytes/SHA-256 是通用 artifact reference |
| immutable result manifest | `tools/evaluation_result.py` | result JSON、digest sidecar、manifest 又形成一套 artifact identity |
| atomic result publication | `tools/evaluation_workspace.py` | staging/no-replace publish 与 data artifact publication 是同类 lifecycle concern |
| resource report envelope | `tools/capacity.py` | schema version、config、code、environment、data binding、trials 由 example 自行拼装 |

仓库中的 Physics example 进一步验证了重复模式：它也自行实现 device/environment
检测、parameter/buffer bytes、host RSS、CUDA/MPS peak memory、phase measurement 和
JSON report。仓库级 sampling-capacity benchmark 又有第三套 profile schema、
environment/resource capture 和报告逻辑。

这已经是“真实重复模式”，不是仅根据 AFHQ 一个 case 预想出来的抽象。

### 2.4 现有 identity 为什么还不够

`DataArtifactIdentity` 的当前字段适合 strict resume：

```text
artifact_type
source_name
source_digest
materializer_name
materialization_digest
artifact_digest
manifest_sha256
```

它有意紧凑、严格、location-independent，但不能直接表达：

- dataset 的 display name、semantic version、license、citation；
- modality-neutral schema 和 split statistics；
- 多个 source 或多个 input artifact；
- 有序 transformation history；
- transformation 的完整参数、software version 和 code revision；
- config、seed、runtime environment；
- artifact storage footprint；
- resource estimate 与 machine-specific observation 的证据区别。

这些内容不应全部塞回 `DataArtifactIdentity`。否则一个可选 statistic 或 dependency
version 的变化会改变 strict resume equality，identity 也会同时承担 cache key、
human metadata、lineage graph 和 runtime audit 四种职责。

## 3. Design Goals

### 3.1 Goals

1. **Example 不重复实现 framework concern。** Core 提供稳定外壳、canonical
   serialization、digest、config/software/environment snapshot 和资源报告语义。
2. **保留现有 identity/resume contract。** Metadata、provenance observation 或
   resource measurement 不参与 `DataArtifactIdentity` equality。
3. **Artifact 与 execution 分离。** Artifact 描述可移植事实；一次具体运行的 config、
   environment、timestamps 和 resource measurements 属于 execution record。
4. **Domain-neutral outer schema，domain-owned payload。** Core 定义字段角色和
   序列化规则，不定义 image class、physics grid、tokenizer schema 或 LLM sequence
   semantics。
5. **支持多个 input 和有序 transformations。** 不要求首版建立全局 provenance
   database 或完整 DAG service。
6. **资源证据必须标明语义。** Intrinsic footprint、预测值和机器实测值不能混为
   `capacity: dict`。
7. **默认可序列化、可比较、可演进。** 公共 value objects 使用 immutable dataclass、
   strict schema version 和 JSON-safe values。
8. **API 小而可扩展。** 首版不增加 artifact registry、dataset registry、universal
   metadata provider registry 或自动 resource estimator registry。

### 3.2 Non-goals

首版明确不做：

- 替换 Stochaflow `DataSource`/`DataBuilder` 或 PyTorch `Dataset`/`DataLoader`；
- 创建通用 Dataset/Sampler/DataLoader YAML graph；
- 建立远程 artifact catalog、metadata database、object store 或 lineage server；
- 定义覆盖 image、table、physics field、token 和 multimodal data 的统一 schema
  language；
- 自动扫描所有 Python dependencies、Git repository、container 和 cloud resources；
- 把 resource estimate 当作 OOM 保证；
- 把 machine observation 写进 artifact identity；
- 为 descriptor、metadata 类型或 resource report 新增 registry；
- 让 core 按 artifact type 或 dataset name 分支。

### 3.3 核心边界

```text
DataSource / Sampling runtime / Evaluation tool
                    |
                    | produces
                    v
          ArtifactDescriptor
          - identity/reference
          - provenance
          - typed metadata
          - intrinsic footprint
                    |
                    | bound/recorded by DataBuilder + runner
                    v
            ExecutionRecord
            - input/output references
            - resolved config snapshot
            - software/code snapshot
            - environment snapshot
            - seed/runtime options
            - resource estimate/observation
```

Artifact descriptor 不拥有 runtime orchestration。Execution record 不构建 Dataset、
model、optimizer 或 sampler。Resource report 不执行 workload。它们都是由现有
composition/runtime 边界产生的 immutable evidence。

## 4. Proposed Architecture

### 4.1 Artifact identity、descriptor 与 binding

本提案区分三个概念：

| 概念 | 作用 | 是否用于 strict resume |
| --- | --- | --- |
| `DataArtifactIdentity` | 当前 data artifact 的完整稳定 equality contract | 是，保持不变 |
| `ArtifactReference` | 跨 artifact type 的紧凑引用：type + content digest | 否；首版只用于 lineage 和 execution I/O |
| `ArtifactDescriptor` | metadata、provenance、footprint 的可移植描述 | 否；可单独计算 descriptor digest 用于审计 |

首版不抽走或重写 `DataArtifactIdentity`。Data artifact 可通过 adapter 产生通用引用：

```text
ArtifactReference.artifact_type = data_identity.artifact_type
ArtifactReference.sha256 = data_identity.artifact_digest
```

完整 legacy identity 继续存入 `DataArtifactBindings` 和 checkpoint。Descriptor 可同时
包含一个 namespaced `stochaflow.data_identity` projection，或者由 runner 在相邻字段
持久化；最终字段布局应在实现 RFC 中确定，但不能让通用引用替代 strict resume
identity。

`ArtifactDescriptor` 的 content identity 与 descriptor identity 也必须分开：

- content digest 回答“是不是同一个 artifact content”；
- descriptor digest 是整个 canonical descriptor 的 SHA-256，回答“描述记录是否完全
  相同”；
- metadata、software evidence 或 optional statistics 的增加可以改变 descriptor
  digest，但不能改变 content identity。

### 4.2 Artifact provenance

`ArtifactProvenance` 首版采用“有序 transformations + input references”，而不是内嵌
完整递归图：

```text
ArtifactProvenance
├── sources[]          external origins
├── inputs[]           framework artifact references with semantic roles
├── transformations[] ordered generation history
└── software[]         producer software/code identities
```

每个 artifact 因而可以回答：

- **来自哪里：** `sources`；
- **使用什么输入：** `inputs`；
- **如何生成：** `transformations`；
- **使用什么版本代码：** `software`；
- **经过哪些步骤：** ordered transformation records。

一个 transformation record 至少包含：

- stable `name`；
- producer-owned `version`；
- JSON-safe `parameters`；
- canonical `parameters_sha256`；
- optional software references。

AFHQ 的 resize recipe 会成为一条或多条 transformation records，但 core 不理解
`LANCZOS` 或 PNG compression 的业务含义。`validation_per_class` 是运行时 Builder
recipe 参数，不应伪装成 materialized artifact transformation。

首版 provenance 不嵌套 input descriptor。多个 descriptor 通过
`ArtifactReference` 相连；调用方可以从 run directory、artifact catalog 或未来 resolver
加载被引用 descriptor。这样可以避免 manifest 无限递归和重复。

Source reference 至少支持：

- `kind`，例如 `http-archive`、`local-tree`、`object-store`、`generated`；
- sanitized `uri`；
- optional upstream `version`；
- optional digest；
- optional license/citation references。

Source URI 不能记录 credentials、signed query 或 secret-bearing headers。Framework
应提供显式 sanitization policy，不能默认把 downloader 的完整 request 打进 manifest。

### 4.3 Dataset metadata

`DatasetMetadata` 是 `ArtifactDescriptor.metadata` 的一个标准 typed payload，不是新的
runtime `Dataset` 基类，也不是新的 DataBuilder config。

最小字段建议为：

- `name`；
- optional semantic `version`；
- optional `homepage`、`license`、`citation`；
- `schema`：modality-owned JSON-safe mapping；
- `partitions`：partition name 到 count/optional statistics 的映射；
- `statistics`：dataset-level、method-qualified statistics；
- `extensions`：namespace-qualified extra metadata。

字段归属如下：

| 用户问题 | 推荐位置 |
| --- | --- |
| dataset name/version | `DatasetMetadata` |
| source URL/checksum/version | `ArtifactProvenance.sources` |
| input artifacts | `ArtifactProvenance.inputs` |
| preprocessing pipeline | `ArtifactProvenance.transformations` |
| class/token/field schema | `DatasetMetadata.schema` |
| train/validation/test sample count | `DatasetMetadata.partitions` |
| mean/std、token length、physical ranges | `DatasetMetadata.statistics`，并记录 method/scope |
| storage bytes/file count | `ArtifactFootprint` |

`schema` 和 `statistics` 需要可扩展，但不能成为无规则垃圾桶：

- outer fields 严格；
- extension keys 使用反向域名或 distribution namespace；
- statistic 必须能够声明 scope、method 和是否 exact/estimated；
- 大型 per-file inventory 不内嵌 metadata，只保存独立 artifact reference 和 digest。

### 4.4 Execution reproducibility and configuration tracking

Artifact lineage 与某次运行的完整 reproducibility evidence 不应混在一个对象中。
新增 `ExecutionRecord`，由 train/sample/evaluate/capacity runtime 在 invocation
边界创建。

`ExecutionRecord` 最小包含：

- operation kind，例如 `training`、`sampling`、`evaluation`、`capacity-profile`；
- `ConfigSnapshot`：完整 resolved config、canonical SHA-256、optional sanitized source；
- seed 与明确的 runtime options；
- selected framework components；
- core 和 extension `SoftwareReference`；
- optional code revision/source-tree digest；
- `EnvironmentSnapshot`；
- input/output `ArtifactReference`；
- lineage，例如 resume/checkpoint parent；
- resource reports。

现有 extension provenance 不应被新的 software snapshot 复制成另一套不兼容语义。
推荐映射为：

```text
ExtensionPluginProvenance
          |
          v
SoftwareReference
```

其中原来的 entry-point、distribution、version、target 仍由
`ExtensionPluginProvenance` 验证；`SoftwareReference` 只为统一 report rendering
提供 projection，并可选增加 source revision/tree digest。

默认 capture policy 应较保守：

- 总是记录 Stochaflow、selected extension distributions、Python、Torch 和 execution
  device 的稳定版本信息；
- Git revision、dirty state、source tree digest、container image digest 只有在可获得且
  用户启用时记录；
- 不扫描整个 environment 的全部 packages；
- 不记录 hostname、username、absolute home path、environment variables 或 secrets，
  除非某个明确 capability 经过过滤后需要它；
- `package_root` 这类本机绝对路径不进入 portable record。

“记录了 environment”不等于“保证可复现”。Record 是审计证据；真正环境锁定仍依赖
project lockfile、released distributions、container 或外部 environment management。

### 4.5 Capacity / Resource Description

`capacity` 不应成为一个语义模糊的 universal mapping。当前需求实际包含三类不同事实：

#### A. Artifact footprint

这是 artifact 自身的、可观察的结构或存储事实，适合放在 descriptor：

- file count；
- stored bytes；
- optional logical payload bytes；
- optional item count（若它不是 dataset metadata 已有的 sample count）。

推荐命名为 `ArtifactFootprint`，并暴露为：

```python
artifact.descriptor.footprint
```

Dataset resolution 不属于 footprint；它属于 schema。Sample count 通常属于 partition
metadata。只有 storage/file/logical payload 才是通用 footprint。

#### B. Resource estimate

这是指定 workload/config 下、运行前的预测：

- expected host/device memory；
- expected storage；
- estimated FLOPs、device-hours 或 wall-time range；
- scaling assumptions；
- estimator name/version 和输入 digest。

Estimate 不是 artifact 固有属性。相同 dataset 配不同 model、batch size、precision、
sampler 或 writer 会产生不同估计。因此它应属于 `ExecutionRecord.resources` 或独立
`ResourceReport`，不能写入 dataset identity。

#### C. Resource observation

这是指定 machine/backend/run 下的实测：

- peak allocated/reserved VRAM；
- RSS；
- throughput、wall time；
- data-wait/compute ratio；
- artifact output bytes；
- precision/device support 和 OOM/non-finite status。

Observation 必须绑定 `EnvironmentSnapshot`、`ConfigSnapshot`、input artifacts 和
measurement protocol。它只证明该受控环境中的一次结果，不是跨机器 capacity guarantee。

#### 结论

Capacity 应成为一个 framework-level **resource evidence family**，但不是一个独立运行时
base class，也不是 `artifact.metadata["capacity"]`：

```text
ArtifactDescriptor.footprint

ExecutionRecord.resources[]
├── ResourceEstimate
└── ResourceObservation
```

首版只稳定 outer report envelope 和 evidence kind。AFHQ/Physics/sampling benchmark
继续拥有自己的 workload、measurement phase 和 metric names。等至少两个 workload
证明有共同 estimator lifecycle 后，再考虑窄 `ResourceEstimator` protocol；首版不建
registry。

### 4.6 Serialization model

公共记录必须满足：

1. immutable dataclasses；
2. JSON-safe leaf values；
3. strict outer fields；
4. integer `schema_version`，不接受 bool/float；
5. canonical UTF-8 JSON 作为 digest authority；
6. YAML 只作为 human-readable rendering，不作为 hash canonical form；
7. `.to_dict()` / `.from_dict()` round trip；
8. deterministic ordering；
9. SHA-256 首版唯一支持，避免无真实需求的 algorithm negotiation；
10. 大 inventory、Tensor、logs 和 binary payload 只通过 artifact reference 引用。

建议 core 提供一套公共 canonical serialization helper，消除 AFHQ preparation、
capacity、evaluation 和 core artifact store 中的重复实现。它应位于通用 artifact
namespace，而不是继续从 private data store helper 反向依赖。

### 4.7 Integration with existing architecture

#### Core module

建议新增：

```text
src/stochaflow/artifacts/
├── __init__.py
├── descriptors.py
├── provenance.py
├── execution.py
├── resources.py
└── serialization.py
```

该 namespace 表示 framework records，不接管 data/sampling/checkpoint 的 runtime
lifecycle。

Safe I/O、locking 和 atomic publication 是相邻的 framework lifecycle concern，但不应
被塞进 descriptor value objects。Phase 1 复用现有 core internal primitives；Phase 2
只有在 data publication 与 evaluation publication 的 no-replace、recovery 和安全路径
语义完全对齐后，才把共同部分移到通用 internal artifact I/O module。领域 layout
validation 仍留在 producer。

#### Public API

公共 API 从 `stochaflow.artifacts` 导出。Extension 需要构造 dataset descriptor 时，
再从 `stochaflow.extensions` 精选导出最小 value objects。不要把内部 collectors、
filesystem helpers 或 every report subtype 全部暴露给 extensions。

#### Data artifacts and DataBuilder

推荐 additive integration：

- `ManagedDataArtifact` / `ReferencedDataArtifact` 增加 optional
  `descriptor`；
- `DataLoaders` 增加 optional descriptor bindings；
- `DataArtifactBindings` 保持 identity-only、compact、strict；
- `DataBuilderContext.expected_artifacts` 和 strict resume comparison 保持不变；
- runner 将 identities 写入 checkpoint，将 descriptors 写入 run manifest；
- checkpoint 最多保存 descriptor digest/reference，不默认复制完整 statistics 和
  provenance。

DataBuilder 仍然是运行时数据组合入口。兼容标准 payload 的新来源可以只注册
DataSource 并复用内置 Builder；不同 payload 或 batch 生命周期才注册自定义
DataBuilder。Core 不因 descriptor metadata 改变 batch contract。

#### Registry

首版不新增 registry：

- artifact type 是 descriptor 中的 stable string，不是 construction dispatch；
- dataset metadata 是 value object，不是 component；
- provenance collector 由 runner 直接调用；
- resource estimate/observation 是 evidence，不是 executable strategy。

若以后出现多个可替换的 artifact store、resolver 或 estimator，应该为那个明确
collaboration 定义窄 protocol，再评估 registry，而不是现在预先注册所有 artifact
类型。

#### Run manifest and checkpoint

`run_manifest.yaml` 应逐步加入：

- `execution_record`；
- `artifact_descriptors`；
- canonical record digests。

为 backward compatibility，迁移期保留当前 `config`、`extension_plugins`、
`selected_components`、`runtime_options` 和 `data_artifacts` 字段。

Checkpoint 中继续保存：

- resolved config；
- extension provenance；
- selected components；
- strict `data_artifacts` identities；
- lineage。

完整 environment snapshot、capacity observations 和 large descriptors 不应默认复制到
每个 epoch checkpoint。它们留在 sibling run manifest/report，通过 digest/reference
关联。

#### CLI

Phase 1 不需要新增用户必须理解的 config 字段或 CLI flag。Train/sample CLI 自动记录
公共 execution/artifact records。

未来只有在 AFHQ 和第二个不同 modality 已验证统一 descriptor 后，才考虑：

```text
stochaflow artifact inspect <manifest-or-artifact>
```

首版不创建 `artifact create`、catalog sync 或 remote lookup 命令。

#### AFHQ-v2 example

迁移后，AFHQ 仍负责：

- source contract 的 AFHQ-specific values 和 validation；
- download、ZIP inspection、image transformation；
- 将 AFHQ source contract 适配为标准 class-labeled artifact；
- evaluation metric protocol；
- capacity trial workload。

Framework 接管：

- class-labeled partition、Dataset、Sampler、collate 和 DataLoader；
- canonical record serialization/digest；
- source/provenance/transformation record model；
- dataset metadata/footprint envelope；
- config/software/environment snapshot；
- execution record；
- resource estimate/observation envelope；
- common artifact/result references。

## 5. API Sketch

以下仅表示职责和最小形状，不是最终命名或可直接实现的完整代码。

### 5.1 Recommended minimal API

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

type JsonScalar = None | bool | int | float | str
type JsonValue = (
    JsonScalar
    | tuple[JsonValue, ...]
    | Mapping[str, JsonValue]
)
type JsonObject = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_type: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceReference:
    kind: str
    uri: str | None = None
    version: str | None = None
    sha256: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactLink:
    role: str
    artifact: ArtifactReference


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    name: str
    version: str
    parameters: JsonObject
    parameters_sha256: str


@dataclass(frozen=True, slots=True)
class SoftwareReference:
    name: str
    version: str | None
    revision: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    sources: tuple[SourceReference, ...] = ()
    inputs: tuple[ArtifactLink, ...] = ()
    transformations: tuple[TransformationRecord, ...] = ()
    software: tuple[SoftwareReference, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactFootprint:
    file_count: int | None = None
    stored_bytes: int | None = None
    logical_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor[MetadataT]:
    identity: ArtifactReference
    provenance: ArtifactProvenance
    metadata: MetadataT
    footprint: ArtifactFootprint | None = None
    schema_version: int = 1
```

Dataset specialization：

```python
@dataclass(frozen=True, slots=True)
class DatasetPartitionMetadata:
    sample_count: int | None
    statistics: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    name: str
    version: str | None = None
    homepage: str | None = None
    license: JsonObject | None = None
    citation: str | None = None
    schema: JsonObject = field(default_factory=dict)
    partitions: dict[str, DatasetPartitionMetadata] = field(
        default_factory=dict
    )
    statistics: JsonObject = field(default_factory=dict)
    extensions: JsonObject = field(default_factory=dict)


type DatasetDescriptor = ArtifactDescriptor[DatasetMetadata]
```

Execution 与 resource evidence：

```python
@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    resolved: JsonObject
    sha256: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    python: str
    platform: str
    framework: JsonObject
    device: JsonObject


@dataclass(frozen=True, slots=True)
class ResourceReport:
    evidence: Literal["estimate", "observation"]
    subject: JsonObject
    protocol: JsonObject
    metrics: JsonObject
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    operation: str
    config: ConfigSnapshot
    software: tuple[SoftwareReference, ...]
    environment: EnvironmentSnapshot
    seed: int | None = None
    components: JsonObject = field(default_factory=dict)
    runtime_options: JsonObject = field(default_factory=dict)
    lineage: JsonObject = field(default_factory=dict)
    inputs: tuple[ArtifactLink, ...] = ()
    outputs: tuple[ArtifactLink, ...] = ()
    resources: tuple[ResourceReport, ...] = ()
    metadata: JsonObject = field(default_factory=dict)
    schema_version: int = 1
```

实现时 constructor 需要 defensive copy/freeze 和真正的 recursive JSON value
validator；`frozen=True` 本身不会冻结传入的 mapping。Path、Tensor、module 或任意
object 都不能混入 canonical record。Serializer 再将 immutable tuple/mapping 投影为
JSON list/object。

### 5.2 AFHQ descriptor example

```python
descriptor = ArtifactDescriptor(
    identity=ArtifactReference(
        artifact_type="stochaflow.class-labeled-image-folder.v1",
        sha256=prepared.artifact_digest,
    ),
    metadata=DatasetMetadata(
        name="AFHQ-v2",
        version="2",
        homepage=lock.homepage,
        license={"name": lock.license_name, "url": lock.license_url},
        citation=lock.citation,
        schema={
            "modality": "image",
            "format": "PNG",
            "mode": "RGB",
            "size": [resolution, resolution],
            "class_mapping": dict(lock.contract.class_mapping),
        },
        partitions={
            split: DatasetPartitionMetadata(
                sample_count=sum(class_counts.values()),
                statistics={"class_counts": dict(class_counts)},
            )
            for split, class_counts in plan.counts.items()
        },
    ),
    provenance=ArtifactProvenance(
        sources=(
            SourceReference(
                kind="http-archive",
                uri=lock.url,
                version="AFHQ-v2",
                sha256=source.sha256,
            ),
        ),
        transformations=(
            TransformationRecord.from_parameters(
                name="stochaflow.afhq-v2.rgb-lanczos-png",
                version="1",
                parameters=dict(plan.recipe),
            ),
        ),
        software=software_references_for_current_extension(),
    ),
    footprint=ArtifactFootprint(
        file_count=len(prepared.image_records),
        stored_bytes=sum(
            record.size_bytes for record in prepared.image_records
        ),
    ),
)
```

这里仍然没有让 core 理解 AFHQ classes、Pillow recipe 或 validation split。

### 5.3 Alternatives considered

#### Alternative A: `artifact.metadata / artifact.provenance / artifact.capacity`

优点：

- 调用直观；
- 与用户心智模型接近。

问题：

- `capacity` 混淆 footprint、estimate 和 observation；
- 把 execution-specific environment/config 塞进 artifact 会破坏可移植性；
- 对 checkpoint、evaluation result 和 dataset 的 runtime object 形成不必要继承层级。

结论：保留 `artifact.descriptor.metadata` 和 `.provenance` 的可发现性，但将
`.capacity` 改为 `.footprint`；estimate/observation 放入 `ExecutionRecord`。

#### Alternative B: `DatasetSpec / ArtifactDescriptor / ExperimentContext`

优点：

- typed object 边界清晰；
- pre-materialization 与 post-materialization 可以显式区分。

问题：

- `DatasetSpec` 容易复制 `ComponentConfig` 与 DataBuilder private params；
- `ExperimentContext` 容易复制 `StochaflowConfig`、runner options 和现有
  `TrainingBuilderContext`；
- 会诱导 core 定义通用 dataset construction schema。

结论：首版只引入 post-materialization `DatasetMetadata` 和 record-oriented
`ExecutionRecord`。没有第二个真实 pre-materialization consumer 前，不引入
`DatasetSpec`；不用 `ExperimentContext` 这个易与 runtime context 混淆的名称。

#### Alternative C: 只使用 `dict[str, Any]`

优点：

- 实现快；
- extension 自由。

问题：

- 当前 AFHQ 已经展示了多套 schema/version/hash/parser；
- 字段语义、equality、unknown-field policy 和 migration 无法统一；
- resource estimate/observation 很容易再次混淆。

结论：使用 strict typed outer envelope，并为 domain schema/statistics/metrics 保留
namespaced JSON mappings。

#### Alternative D: 立即用新的 `ArtifactIdentity` 替换 `DataArtifactIdentity`

优点：

- 表面上只有一套 identity。

问题：

- 破坏已验证的 strict resume 和 managed/referenced data lifecycle；
- 通用 content reference 无法自然替代 source/materializer/manifest 多重 digest；
- 当前没有 checkpoint、sampling result、evaluation result 三种 artifact 都需要相同
  equality contract 的证据。

结论：首版 additive。先用 `ArtifactReference` 连接通用 descriptor，保留
`DataArtifactIdentity`；在多个 artifact family 迁移后再单独评估 identity unification。

## 6. Migration Plan

### Phase 1: 引入基础 abstractions，保持 backward compatibility

目标：

- 新增 `stochaflow.artifacts` value objects 与 canonical serialization；
- 新增 `ArtifactDescriptor`、`DatasetMetadata`、`ArtifactProvenance`、
  `ConfigSnapshot`、`ExecutionRecord`、`ArtifactFootprint` 和 `ResourceReport`；
- 为现有 `DataArtifactIdentity` 提供 `ArtifactReference` adapter；
- `DataArtifact`、`DataLoaders` 的 descriptor 字段为 optional；
- runner 可写新 records，但继续写旧 manifest 字段；
- checkpoint/resume 仍只依赖现有 `DataArtifactBindings`；
- core train/sample manifest 开始复用同一 config/software/environment snapshot
  helper。

Phase 1 验收：

- deterministic serialization 与 digest tests；
- strict parser、unknown-field、schema version、invalid JSON value tests；
- descriptor/config/execution round-trip tests；
- legacy DataBuilder 不返回 descriptor 时行为不变；
- strict resume identity comparison 完全不变；
- manifests 不记录 secret-bearing URI、environment variable 或本机 package root；
- public API 不导出 filesystem mutation helpers。

### Phase 2: AFHQ-v2 使用 framework capability

目标：

- AFHQ materializer 生成 `DatasetDescriptor`；
- prepared manifest 使用 framework descriptor serialization，保留旧字段或提供
  v1 reader adapter；
- DataBuilder 返回 identity bindings 和 descriptor bindings；
- training run manifest 自动包含 AFHQ descriptor；
- capacity tool 使用 core `ConfigSnapshot`、software/environment capture 和
  `ResourceReport`；
- evaluation result 使用 common artifact references、execution record 和 immutable
  descriptor digest；
- AFHQ-specific source validation、image transformation、payload adaptation、
  evaluation protocol 和 trial execution 保持不变。

Phase 2 验收：

- AFHQ artifact identity 与当前版本相同，现有 checkpoint strict resume 不受影响；
- descriptor 能回答 source、input、transformation、software、dataset schema、
  partitions 和 footprint；
- capacity output 明确区分 observation 与 estimate；
- evaluation/capacity 不再分别实现 distribution version、canonical config hash 和
  environment snapshot；
- manifest migration reader 能读取现有 prepared artifact。

在 Phase 2 结束前，使用 Physics example 作为第二个结构不同的 validation case：

- physics dataset schema 使用 field/channel/time/grid semantics，而不是 image fields；
- physics capacity observation 使用相同 outer resource envelope，但保留自己的 phase
  metrics；
- 若需要在 core 按 `artifact_type == ...` 或 dataset name 分支，说明 abstraction
  边界错误，应停止并修订 proposal。

### Phase 3: 删除 example 私有实现

在 AFHQ 与至少一个不同 modality 验证后：

- 删除 AFHQ `_preparation/identity.py` 的 canonical serialization；
- 删除 `tools/capacity_provenance.py` 中已由 core 提供的 config/code/environment
  helpers；
- 删除 evaluation result 中重复的 dependency/file identity helpers；
- 将 AFHQ prepared manifest reader/writer 缩减为 domain validation +
  framework descriptor adapter；
- 逐步迁移 Physics 和 repository sampling-capacity reports；
- 文档稳定后将用户-facing 内容移入正常 docs tree，并删除或归档本开发草案。

只有在所有支持的旧 manifest/checkpoint 迁移窗口结束后，才能考虑移除旧字段。现有
`DataArtifactIdentity` 是否泛化或重命名，应作为独立 RFC，不与本次 additive migration
绑定。

### 6.4 预计修改面

| Area | Phase 1 | Phase 2/3 |
| --- | --- | --- |
| core module | 新增 `stochaflow.artifacts` | 根据第二个 modality 收紧 schema |
| public API | 导出最小 record types | 稳定验证过的 extension constructors |
| registry | 无变化 | 仍默认无变化 |
| serialization | 新增 canonical JSON authority | AFHQ/Physics report 迁移 |
| DataArtifact | optional descriptor | AFHQ descriptor required |
| DataLoaders | optional descriptor bindings | AFHQ 返回 descriptors |
| run manifest | additive execution/artifact records | 逐步淘汰重复字段 |
| checkpoint | 保持 identity；可加 descriptor digest/ref | 不嵌入大 descriptor |
| CLI | 无新必填参数 | 验证后可考虑只读 inspect |
| AFHQ example | 暂不改变 | 删除通用 metadata/provenance/resource boilerplate |

## 7. Risks and Open Questions

### 7.1 Risks

#### Scope creep

“Artifact”很容易扩展成 storage backend、catalog、remote resolver、lineage database、
schema registry 和 workflow engine。首版必须坚持 record model，不接管这些 runtime。

#### Identity instability

若 metadata、environment、timestamp 或 statistics 被错误纳入 content identity，
相同 content 会在不同机器得到不同 identity。实现必须分别测试 content identity、
descriptor digest 和 execution record digest。

#### Metadata dumping ground

无限制 `metadata: dict` 会重新制造不兼容 schema。Outer typed fields、namespace、
JSON validation 和文档化 extension policy 必须在首版完成。

#### Expensive or mutable statistics

Dataset statistics 可能需要全量扫描，也可能因 sampling method 改变。Statistics 必须
标明 exact/estimated、scope、method 和输入 digest；descriptor creation 不应默认触发
昂贵扫描。

#### Code provenance cost and false confidence

递归 hash source tree 会增加 startup cost，editable install 还可能包含 untracked 或
generated files。默认只记录 distribution/version；tree digest、Git revision 和 dirty
state 应显式启用并说明覆盖范围。

#### Privacy and secrets

Source URI、absolute path、hostname、environment variables、cloud identifiers 和 CLI
arguments 都可能泄露信息。Serialization helper 需要明确 sanitization contract。

#### Resource evidence misuse

用户可能把一次 CUDA measurement 当成跨平台 guarantee。Schema 和文档必须要求
`evidence`、environment、protocol、assumptions，且禁止省略 measurement context。

#### Manifest/checkpoint growth

Per-file inventory、完整 dependency list、statistics 或 capacity trial details 可能非常
大。Checkpoint 只保存 identity/reference；run directory 保存 descriptor/report；
large payload 单独成为 artifact。

#### Schema evolution

严格 schema 会产生 migration 成本。每个 top-level record 必须独立 version，
reader 明确支持哪些版本；不能用“忽略所有 unknown fields”掩盖语义变化。

### 7.2 Open Questions

1. `ArtifactDescriptor` 是否直接携带完整 legacy `DataArtifactIdentity` projection，
   还是由 run manifest 在相邻字段保存？建议 Phase 1 prototype 比较重复量后决定。
2. Descriptor bindings 应新增独立 `ArtifactDescriptorBindings`，还是让
   `DataLoaders` 返回 `dict[str, ArtifactDescriptor]`？建议优先独立 immutable binding
   collection，避免 mutable dict 和 role ordering 问题。
3. `DatasetMetadata.statistics` 的最小公共 statistic record 是否在 Phase 1 稳定，
   还是先保留 namespaced JSON？建议只稳定 scope/method/evidence outer shape。
4. 默认 software capture 是否包括 source-tree digest？建议默认否；release version +
   extension provenance 常开，tree digest opt-in。
5. `EnvironmentSnapshot` 的 device schema 是否应覆盖 CPU/MPS/CUDA，还是先用
   namespaced mappings？建议稳定 backend/name/memory bytes 等少量字段，其余
   namespaced。
6. Resource estimate 与 observation 是否共用一个 `ResourceReport` envelope，还是两个
   dataclass？建议序列化 envelope 共用，Python types 分开，以防调用端混用。
7. AFHQ prepared manifest 是否直接升级 schema version，还是先把 descriptor 作为 v1
   additive field？当前 parser 使用 exact keys，无法静默 additive；需要明确 v1-to-v2
   reader，而不是放宽所有 unknown fields。
8. Evaluation result、sampling output 和 checkpoint 何时正式成为 descriptor-backed
   artifacts？建议先迁移 data + evaluation result；checkpoint 只做 reference consumer，
   等 post-training evaluation proposal 的 artifact contract 稳定后再扩展。

### 7.3 Decision gates

进入实现前应确认：

1. 是否接受“保留 `DataArtifactIdentity`，新增通用 `ArtifactReference`”的 additive
   路线；
2. 是否接受 capacity 三分法：footprint / estimate / observation；
3. 是否接受 `ExecutionRecord`，而不是把 config/environment 全部塞入 artifact；
4. 是否同意 Phase 1 不新增 registry 和 CLI command；
5. 是否以 AFHQ + Physics 作为 public API 稳定前的最小跨 modality gate。

若以上决策成立，本提案的最小实现顺序应为：

```text
canonical serialization
    -> artifact/provenance value objects
    -> execution/config/software/environment records
    -> optional DataArtifact/DataLoaders integration
    -> AFHQ migration
    -> Physics validation
    -> remove duplicated example utilities
```

该顺序先统一事实模型，再迁移具体 DataSource；不会让 runner、DataBuilder 或 artifact
storage 在抽象尚未验证前承担新的 orchestration 职责。
