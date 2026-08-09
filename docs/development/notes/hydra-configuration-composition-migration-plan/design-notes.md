# Hydra 配置组合设计附录

> 本文保存候选设计细节，不表示 Hydra 已安装、已进入依赖或已成为公共接口。
> 可执行状态与任务以
> [`hydra-configuration-composition-migration-plan.md`](../../hydra-configuration-composition-migration-plan.md)
> 为准。
>
> 最后核对：2026-08-09。真正实施前必须重新核对 Hydra 官方版本、override
> 语法、插件行为和安全建议。

## 要解决的具体问题

Plain YAML 仍是当前稳定输入。候选迁移只解决 fresh training 配置越来越长、共享片段
难以复用和命令行 override 难以预览的问题。它不接管 checkpoint resume、sampling 或
Evaluation 的独立配置归属。

候选数据流是：

```text
trusted config root + Defaults List
    -> Hydra composition
    -> fully resolved primitive mapping
    -> Stochaflow load/validate
    -> extension preflight and explicit activation
    -> existing Registry/Builder/runtime
```

Hydra 的输出必须在进入 Stochaflow 前变成普通 primitive mapping。之后的验证、对象构造、
run directory、日志、checkpoint、manifest 和结果仍由现有 Stochaflow 生命周期负责。

## 受信任的配置根

- 配置搜索路径由应用明确提供，不扫描工作目录或任意 installed package。
- Defaults List 只引用受信任 root 下的已知 group。
- `_target_`、任意 Python import 和递归对象实例化不进入用户配置契约。
- 插件 discovery/provenance preflight 与代码 activation 保持两步；composition 不能借机
  导入尚未验证的 extension module。
- resolved mapping 必须通过当前 `StochaflowConfig` 和 Registry contract 验证。

## 候选 authoring groups

Group 应围绕读者要替换的配置责任，而不是镜像 Python 包目录。候选包括 experiment、
data recipe、model、process、training method、optimizer/scheduler provider、runtime 和
extension selection。每个 group 展开后仍必须得到当前 schema 能表达的普通字段。

不新增 universal component graph。复杂多模型、teacher、codec 或 inference asset 继续由
TrainingBuilder/SamplingBuilder 等 Python 组合点负责。

## 共用的单次训练调用

候选 `TrainingInvocation` 只描述一次已经完成 composition 的 training 调用：resolved
primitive config、明确 output destination、受限 runtime overrides 和显式 extension plan。
它不得重新定义训练配置 schema，也不得把 sampling/Evaluation 作为隐式尾部动作。

这个入口由[显式顺序工作流计划](../../default-workflow-pipeline-support-plan.md)统一拥有。
CLI、Hydra 和 HPO 最终都消费同一 single-run implementation，不各自复制训练 runner；
Hydra 本文只负责把已组合、已解析的普通配置交给它。

## Preview 与 override

在创建正式 output 前，用户应能查看：

- Defaults 展开后的来源；
- fully resolved primitive mapping；
- extension selection 和 provider identity；
- Stochaflow validation errors；
- 最终 output destination。

Override 只允许修改当前 schema 中明确可修改的字段。未知 key、删除必填字段、改变
checkpoint 固定 identity 或跨 operation 偷渡配置必须明确失败。

## Resume 和采样边界

- Resume 继续读取 checkpoint authority，并验证当前 invocation 与保存状态兼容。
- Sampling 使用自己的完整 sample config；不从 training Hydra compose tree 继承 seed、
  sampler、writer 或 task option。
- Hydra 迁移完成后，另行审查 sampling invocation 是否需要更好的 authoring API。
- 已发布的 resolved config 和 manifest 不依赖 Hydra runtime 才能读取。

## 暂停的 multirun 与 sweep

Hydra multirun、sweeper 和 launcher 不属于第一轮。它们只有在 single-run library entry、
HPO objective/budget、trial isolation、resume identity 和 output layout 都稳定后才重新讨论。
外部 launcher 继续拥有队列、资源和集群控制。

## 失败与验证

- 不受信任搜索路径、未知 Defaults entry 和非法 override 在 I/O 前失败。
- Plain YAML 与 Hydra 展开后的同一 mapping 构造相同组件并得到等价 manifest。
- Extension preflight 发生在 import 前；activation 只使用 sealed plan snapshot。
- CLI/Hydra/library single-run 的 resume、错误类型和 output publication 保持一致。
- 配置 reference 仍由 Stochaflow schema 生成，不由 Hydra group 文件反向定义。
