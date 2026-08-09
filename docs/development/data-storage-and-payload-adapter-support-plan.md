# 扩展 Data storage 与 semantic payload 表示

> 工作状态：暂停
>
> 当前结论：本地 managed/referenced artifact 与 family-owned payload 已能表达当前用例。
> 只有多个真实表示重复同一缺口，或现有 read boundary 无法安全拒绝失败时，才新增 adapter。
>
> 规范来源：[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 排期权威：[`ROADMAP.md`](../../ROADMAP.md)

## 完成后用户能做什么

用户可以通过经过验证的窄 adapter 使用新的 storage representation 或 remote provider，
而兼容的 `DataBuilder` 仍只依赖 semantic payload contract，不需要按存储格式增加 core 分支。

## 当前仓库已经支持什么

- `DataArtifactStore` 支持本地 managed content 与 filesystem-referenced content。
- `DataArtifact[P]` 的 payload 由 family 拥有，可以是任意项目 dataclass 或对象。
- Manifest、content digest、producer identity 和 strict-resume validation 已统一。
- Family-local source Registry 与 Builder payload validation 不依赖 `IMAGE_DATA_SOURCES`。

## 还没有支持什么

- provider-neutral remote object-store adapter；
- semantic inventory 与 folder、shard、LMDB 等 storage locator 的公共拆分；
- 通用 artifact-to-Dataset adapter Registry；
- ImageNet shard/native-validation recipe；
- 当前 manifest/cache read boundary 无法表达的新 failure 类型。

## 什么时候可以开始

至少两种真实 storage representation 或两个独立消费者必须需要相同的窄 contract；或者必须
提供一个可复现 failure，证明现有 manifest、cache 和 read-boundary validation 不能安全拒绝。
单一 provider 的字段列表或未来数据集名称不足以启动。

## 要完成哪些工作

### 分离稳定语义与存储偶然性

- 动作：比较真实 producer/consumer 的 semantic inventory、locator、内容验证和失败规则。
- 原因：按格式建立全局 adapter 会把 storage 偶然性变成 core API。
- 影响范围：family payload、DataSource、Store load callback 和 Builder compatibility。
- 交付物：兼容矩阵、拒绝路径和最窄可共享 capability。
- 验证方法：两个 representation 可替换，且 Builder 不读取 provider 名称或具体存储类。
- 完成条件：共享字段对所有消费者含义一致，未共享部分继续由 family/project 拥有。

### 验证新 provider 或 read-boundary failure

- 动作：实现一个真实 adapter 或失败 fixture，并复用现有 identity、receipt 和 strict-resume lifecycle。
- 原因：新边界必须证明它比 project-private Source helper 更可靠或更可复用。
- 影响范围：DataArtifactStore 的窄 provider boundary、错误分类和公开迁移说明。
- 交付物：独立 extension、corruption/staleness tests、cache reuse tests 和兼容文档。
- 验证方法：错误内容在 read boundary 失败，合法 provider 不需要 runner/core dispatch 分支。
- 完成条件：现有 local managed/referenced 行为不变，provider 私有字段不进入稳定 identity。

## 如何证明已经完成

- 至少两个真实 representation 的 substitution tests。
- corruption、staleness、mutation、cache hit 和 strict-resume failure tests。
- 非图像 family-local payload 与 built-in image recipe 均无需 core 特判。
- 公开文档区分 semantic contract、storage locator 和 provider responsibility。

## 明确不包含什么

- 不建立组织级 catalog、metadata warehouse 或 capacity scheduler。
- 不定义 universal payload schema 或通用 artifact-to-Dataset graph。
- 不承诺所有 remote provider、shard format 或数据库。
- 不把 Dataset、sampler、collate 或 loader 放进 `DataArtifactStore`。

## 详细设计和研究资料在哪里

- [原始 storage、payload、ImageNet 与 adapter 研究](notes/data-layer-composition-boundary-review/research-archive.md)
- [Artifact metadata/provenance/capacity 提案](artifact-metadata-provenance-capacity-model-proposal.md)
- [当前 Data pipeline](../configuration/data-pipeline.md)
