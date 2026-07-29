# Data Artifact Producer Lifecycle Refactor

- 文档性质：已实施的开发决策记录；不属于公开 API
- 状态：Implemented
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)；作为 prepared
  posterior artifact 基座，不重开 producer lifecycle
- Example retirement：本文 Physics/KD 矩阵记录实施时的验收证据；retained-example
  cleanup 后由最小 referenced/no-artifact fixtures 保留 contract coverage，不代表继续
  维护两个 project
- 日期：2026-07-27
- 最近更新：2026-07-29
- 兼容性：breaking；不保留旧 artifact 格式或 API

## 1. 决策

Stochaflow 只保留一条高层 producer lifecycle：

```text
DataSource[P]
    -> DataArtifactStore
    -> DataArtifact[P]
    -> DataBuilder
    -> DataLoaders
```

`managed` 与 `referenced` 是内容 ownership strategy，不再是两套 runtime handle：

- managed：framework cache 拥有并认证 materialized content；
- referenced：cache 只拥有索引/sidecar，represented content 保留在外部 root。

两者共用 concrete schema-v2 `DataArtifactIdentity`、`DataArtifact`、
`DataArtifactBindings`、manifest、cache layout 与 strict-resume lifecycle。

本次不新增 Dataset/Sampler/DataLoader registry、通用 YAML graph、artifact descriptor、
metadata/provenance/capacity model 或 catalog。

## 2. 最终公共 API

`stochaflow.extensions` 公开：

- `DataArtifactIdentity`：固定 schema v2，记录 kind、artifact/source/materializer 名称与
  digest、content/artifact digest 和 `manifest_sha256`；
- `DataArtifact[P]`：`root`、`identity`、`payload`，以及派生 `kind` 和
  `manifest_path`；
- `DataArtifactStore`：`materialize_managed(...)` 与
  `materialize_referenced(...)`；
- `ManagedDataArtifactBuild`、`ReferencedDataArtifactBuild`；
- `DataArtifactLoadContext`、`DataArtifactValidationError`；
- `canonical_artifact_json_bytes()`、`canonical_artifact_digest()`；
- `DataSourceMaterializationConfig`、`DataSourceContext`。

Producer callback 边界：

- managed `build(data_root)` 写入 staging `data/`，framework 扫描并哈希全部普通文件；
- referenced `build(data_root)` 只写 index/sidecar，并返回 external content digest；
- `load(context)` 无副作用；framework-owned 文件只从已验证的 staging/final
  `context.data_root` 读取，referenced producer 可同时使用声明并由 closure 捕获的
  external roots，但不能执行 acquisition、写入或重新物化；
- persisted candidate 或 represented content 不满足契约时抛
  `DataArtifactValidationError`；包括普通 `OSError` 在内的配置/编程错误不会被当成
  可修复 cache corruption。每个加载边界在 callback 前执行一次所需强度的验证；
  `full` 对 framework-owned stored files 只做一次完整 hash。callback 返回后只做
  link-safe 元数据复查，严格比较路径、数量、size、device/inode、mode、mtime 与
  ctime；`manifest` verification 同样拒绝可观察到的同尺寸 mutation。

`DataSource` 仍负责 acquire、validate、transform、materialize；`DataBuilder` 仍是 runtime
recipe composition root，拥有 binding、partition、Dataset、transform、sampler、collate
和 loader。

## 3. Manifest、identity 与 cache

所有 artifact 使用严格 canonical `manifest.json` envelope：

```json
{
  "schema_version": 2,
  "kind": "managed",
  "artifact_type": "...",
  "source": {"name": "...", "digest": "..."},
  "materializer": {"name": "...", "digest": "..."},
  "content_digest": "...",
  "stored_files": {
    "digest": "...",
    "record_limit": 100000,
    "record_count": 0,
    "shards": []
  },
  "domain_digest": "...",
  "domain": {"schema_version": 1},
  "artifact_digest": "..."
}
```

Canonical JSON 使用 UTF-8、排序键、紧凑 separators、末尾换行并拒绝非字符串 key、
非 JSON-safe 值、NaN 与 Infinity。root envelope 严格拒绝未知/缺失字段。

Framework 为 `data/` 生成排序、分片的 `{path,size_bytes,sha256}` inventory，并拒绝
symlink、reparse point 和特殊文件。managed content digest 来自 owned inventory；
referenced content digest 来自 producer 的 external inventory。`domain_digest`、
`artifact_digest` 与 `manifest_sha256` 分层计算，manifest SHA 不嵌入自身。

Cache layout：

```text
<cache_root>/data-artifacts/v2/<kind>/<artifact-type-digest>/
  objects/<artifact_digest>/
  staging/<uuid>/
  locators/<locator-digest>.json
  locks/locators/
  locks/objects/
  quarantine/objects/
  quarantine/locators/
```

`require` 完全只读。`ensure` 在 locator lock 内复查，只对缺失、I/O 错误或
`DataArtifactValidationError` 执行隔离和重建。expected identity 绕过当前 locator，
强制 full verification，且不改写 locator。publication 采用 no-replace 语义，并验证
concurrent winner 与最终路径 payload。

Fresh materialization 的 verified staging 与 published final object 是两个独立验证
边界，各自执行一次内容认证和一次 loader 后元数据复查。cache hit 与 strict resume
只对 final object 执行这一流程。`ArtifactVerificationObserver` 只报告一次
`phase="validate"` 的内容 hash；元数据复查不产生进度事件。referenced artifact 的
external represented content 仍由 producer 在 `load(full)` 中认证，framework 的后置
复查只覆盖 manifest、inventory 与 sidecar。该复查用于约束无副作用 callback，不构成
恶意同机写入的安全边界；缺少稳定 inode/ctime 时，能够恢复 size/mtime 的极端
same-size mutation 可能无法检测。

最终状态机还固定了以下 fail-closed 规则：

- object 只能在对应 digest 的 object lock 内隔离；
- locator 指向另一个 framework-valid producer object 时，只隔离并重建 locator，不移动
  可能被其他 locator 使用的 object；
- 同 digest 的有效不同 identity 是 collision/producer contract bug，不覆盖 winner；
- expected identity 在 producer `load` 与任何 cache mutation 前完成 framework identity
  预检；
- object root、inventory shard、stored file 与目录拓扑必须精确匹配 manifest，不接受
  未认证的额外 entry；
- build callback 不得替换 framework 创建的 staging `data/` 根目录。

## 4. Breaking changes

已删除且没有 alias：

- `ManagedDataArtifact`、`ReferencedDataArtifact`；
- `ManagedDataArtifactIdentity`、`ReferencedDataArtifactIdentity`；
- `artifact_root`、`index_root` 与 caller-provided manifest path；
- producer-owned final identity、manifest、locator、lock/publication lifecycle。

旧 manifest、locator、cache layout、identity 与 checkpoint binding 不读取、不双写、不
迁移。旧 cache 保留在磁盘但不会被发现；用户必须重新 materialize。保存旧 artifact
binding 的 checkpoint 不能 strict resume，应创建新 run。

## 5. Built-in 与 example 矩阵

| 使用方 | Strategy | Framework 拥有 | 使用方仍拥有 |
| --- | --- | --- | --- |
| Torchvision | managed | schema-v2 lifecycle、owned inventory、identity | download、dataset partitions、dimensions payload |
| image/paired folders | referenced | schema-v2 lifecycle、index publication、identity | image scan、external inventory、PIL/domain payload |
| AFHQ-v2 | managed | manifest/inventory/cache/identity/lock/publication | official archive、source lock、ZIP 审计、decode/resize、class payload |
| Physics reconstruction | referenced | manifest/cache/identity/lock/publication | `.npy` schema、external hash、trajectory range、mmap payload |
| Knowledge Distillation | none | 无 data artifact | synthetic splits 由 resolved config 与 seed 构造 |

AFHQ 的 capacity/evaluation report 仍是 example-private；Physics sampling observations
仍是 sampling task input；KD teacher bootstrap 仍是 TrainingBuilder 的模型构造输入。
它们都不进入通用 data artifact metadata。

## 6. 数据组合边界

Source-only 复用仍要求同时满足：

1. 目标 Builder 的完整 accepted artifact contract；
2. 相同的 partition、Dataset、transform、sampler、collate、loader、resume 与 batch
   语义。

否则使用 recipe-level `DataBuilder`。完全 synthetic recipe 可以没有 DataSource 和
artifact binding。这个 producer refactor 没有改变这些规则。

## 7. 验收结果

最终验收覆盖：

- managed/referenced contract tests 覆盖 ensure/require、manifest/full、expected
  identity、corruption/repair、locator/object 双锁、collision、strict object layout、
  safe paths 与 atomic publication；
- independent custom DataSource 只使用 `stochaflow.extensions` 的 store；
- Torchvision、image folders、paired folders、AFHQ、Physics 与 KD 均遵守上述矩阵；
- 配置参考、公开 extension/data/checkpoint 文档与三个 example 文档同步；
- repository search 不再出现旧 runtime handle、旧 AFHQ manifest/cache lifecycle 或
  extension-local final identity/publication 实现。

2026-07-27 的最终结果：

```text
focused artifact/built-in/example suite: 174 passed
repository pytest:                      1035 passed, 13 skipped (CUDA/BF16)
ruff:                                   passed
pyright:                                0 errors, 0 warnings
config reference generator --check:     passed
Sphinx -W:                              passed
uv build:                               passed
```

2026-07-29 将每个加载边界的第二次完整 hash 收敛为严格元数据复查：

```text
focused artifact/store/recipe suite:   277 passed
repository pytest:                     1121 passed, 14 skipped (CUDA/BF16)
ruff:                                  passed
pyright:                               0 errors, 0 warnings
Sphinx -W:                             passed
```

metadata/provenance/capacity 提案保持 Deferred，直到出现满足其 decision gates 的真实
重复模式。
