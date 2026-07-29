# Framework Capability Proposal: Artifact Metadata, Provenance and Capacity Model

- 文档性质：延期的开发决策记录；不属于公开 API
- 状态：Deferred
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)；当前 latent codec
  identity 只记录加载与验证所必需的事实，不重新开启通用 descriptor
- 更新日期：2026-07-27
- 延期原因：当前优先完成 data artifact producer lifecycle 与数据组合边界；尚没有足够的
  跨领域重复模式支持统一 metadata/provenance/capacity API

## 1. 当前决策

本提案暂不实施。当前 core 不新增：

- `ArtifactDescriptor`、`ArtifactReference` 或 `DatasetMetadata`；
- 通用 provenance graph、transformation history 或 code/environment snapshot；
- `ExecutionRecord`、resource estimate/observation envelope；
- `artifact.metadata`、`artifact.provenance` 或 `artifact.capacity`；
- artifact catalog、descriptor registry、CLI 查询或 runner integration。

AFHQ capacity/evaluation、sampling capacity 与 extension plugin provenance 继续保留
各自已有且边界明确的语义。本轮不因名称相似而合并它们。Physics capacity 只作为
历史重复模式证据；retained-example cleanup 后不再是 maintained capability。

## 2. 已解决的 artifact 基础问题

数据 artifact 的 producer lifecycle 已由独立重构解决，不依赖本提案：

```text
DataSource
    -> DataArtifactStore
    -> schema-v2 manifest / inventory / identity
    -> DataArtifact[payload]
    -> DataBuilder
```

managed 与 referenced producer 共用一个 framework lifecycle，包括：

- canonical JSON 与 digest；
- immutable object、locator、locking、staging、publication 和 quarantine；
- source/materializer/content/artifact/manifest identity；
- manifest/full verification；
- strict-resume expected identity；
- `DataArtifactBindings`。

这些字段是验证和恢复所需的最小 identity，不是一个通用 metadata 或 provenance model。
manifest 中的 `domain` 只保存 producer 加载与验证 typed payload 所必需的事实。它不是
任意 metadata bag，也不承诺描述 lineage、license、citation、runtime recipe 或资源需求。

当前 schema v2 是 breaking contract：旧 identity、manifest、locator、cache 与 checkpoint
binding 不读取、不适配、不迁移。

## 3. 仍然存在但尚未统一的问题

未来真实案例可能再次证明以下重复：

- dataset 的人类可读名称、版本、schema、native partitions 与统计；
- source citation/license、输入引用与 transformation sequence；
- code/config/environment snapshot；
- storage/sample footprint；
- 某个 workload 的资源 estimate；
- 某次运行在特定机器上的 resource observation。

这些概念不能全部塞入 `DataArtifactIdentity` 或 Dataset object：

- runtime derived split、augmentation、sampler、batching 和 packing 属于 resolved data
  recipe，不属于 materialized artifact；
- GPU memory/compute 由 model、precision、batch 和 runtime 共同决定，不是 dataset
  metadata；
- observation 是一次执行证据，不是静态 capacity guarantee；
- extension plugin provenance 已有独立 checkpoint 语义，不能被第二套近似模型替换。

## 4. 重新开启本提案的 decision gates

只有满足以下条件才重新设计：

1. 至少两个独立 producer 需要同一组可移植 dataset metadata，而不是仅需要私有
   `domain` payload facts。
2. 至少两个 workflow 需要共享相同 provenance 或 execution-record consumer。
3. 至少两个 benchmark/report 需要相同 resource envelope，并能明确区分
   footprint、estimate 和 observation。
4. 存在具体读取方，能够证明数据必须进入公共 API、serialization 或 CLI。
5. 新能力可以作为相邻 immutable evidence，不改变 schema-v2 artifact identity 与
   DataSource/DataBuilder 责任边界。

重新开启时应先做跨 AFHQ、第二个真实 trajectory/physics producer、
LLM/streaming 或其他真实案例的 semantic inventory，
再提出最小 capability。不得从一个 example 反推宇宙级 descriptor。

## 5. 保留的设计原则

1. Example 不应重复实现已经稳定、确有多个使用方的 framework concern。
2. Framework abstraction 必须来源于真实重复模式。
3. API 优先简单，并允许 family-specific extension。
4. Identity、metadata、provenance、runtime recipe 和 resource observation 必须保持不同
   语义。
5. Capacity 必须至少区分 artifact footprint、workload estimate 与 runtime
   observation，不能是一个无约束字典。

在 decision gates 满足前，本文件只记录延期边界，不是 active implementation plan。
