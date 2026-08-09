# Hydra 全新训练配置迁移计划

> 工作状态：候选
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## 完成后用户能做什么

用户可以用受控的 Hydra `Defaults List` 和少量配置组组合一次全新训练配置，先查看最终的基础
类型映射并检查错误，再通过与普通 CLI 共用的 Python 入口启动一次训练。这个入口由
[显式顺序工作流计划](default-workflow-pipeline-support-plan.md)统一交付；Hydra 只消费它。
检查发生在读取数据或创建正式输出之前。

Hydra 只负责配置编写和组合。Stochaflow 仍负责类型与跨字段校验、extension 激活、
Registry/Builder 构造、运行目录、日志、checkpoint、manifest 和结果发布。

## 当前仓库已经支持什么

- train 与 sample 已使用各自独立、完整的普通 YAML 配置；sample 不与训练配置或 checkpoint
  中可变的默认值合并。
- checkpoint、严格恢复、sampling 和 Evaluation 的约定已由根级规范与公开文档定义。
- 基础类型映射会经过类型和跨字段校验，再由 Registry、Builder 与受限的原生 provider resolver
  构造运行对象。
- MNIST 与 AFHQ-v2 是持续维护的全新训练对照案例。

当前没有 Hydra/OmegaConf 依赖、`Defaults List`、配置组、Hydra launcher、Sweeper、multirun 或
`.hydra/` 输出流程。

## 还没有支持什么

- 没有受控的 Hydra 启动入口、明确的项目配置根目录或配置树规模上限。
- 没有允许覆盖项清单、最终配置预览、组合摘要或支持 Hydra 的 `--check`。
- 还没有由显式顺序工作流计划交付、供普通 YAML 与 Hydra 共用的不可变单次训练调用。
- 没有适用于已安装 wheel 的外部 extension 配置根目录约定或项目模板。
- 没有 MNIST/AFHQ-v2 普通 YAML 与 Hydra 行为一致的证据。

## 什么时候可以开始或重新审查

本计划只有在 [`ROADMAP.md`](../../ROADMAP.md) 指定负责人、范围、依赖和退出方案后才从候选
转为实施。启动前还必须：

- 固定普通 YAML 的当前行为，并准备 MNIST/AFHQ-v2 有界对照用例；
- 用当前 lockfile 重新核验 Hydra/OmegaConf 版本、resolver/import/search path 的安全面和许可证；
- 审核配置树规模上限、允许覆盖项、extension 启动方式和原生 provider 边界；
- 证明可选安装不会改变未使用 Hydra 的 train/resume/sample/evaluate 路径；
- 明确 Hydra 不管理正式输出、当前工作目录、`.hydra/` 快照或第二套日志。
- 明确单次训练 Python 入口由显式顺序工作流计划负责，本文不建立第二个入口。

Hydra 候选完成后，
[Sampling Invocation Post-Hydra Review](sampling-request-config-refactor.md)仍保持暂停，只有根路线图
另行选择才会重审。

有限 multirun 是独立的后续构想，不属于本计划首版。只有单次训练入口稳定，且两个真实使用方
需要同一种有限笛卡尔积搜索时，才由 Hydra 配置计划负责人重新审查进程隔离、资源预算、输出
管理和失败汇总；自适应 HPO 仍由独立 HPO 计划负责。

## 要完成哪些工作

### 建立受控的配置组合入口

- 动作：实现明确的配置根目录、有限的 `Defaults List`、允许覆盖项清单、基础类型转换、最终配置
  预览和组合摘要。
- 原因：Hydra 必须是可审计的配置入口，不能成为第二套运行时 schema 或任意 Python 对象工厂。
- 影响范围：可选启动入口、配置编写、预览/检查和配置来源记录。
- 交付物：受控的搜索路径、配置树规模上限、允许覆盖项、基础类型映射和清晰的错误信息。
- 验证方法：缺失项、未知项、循环引用、越出根目录、resolver 副作用、插值失败、无权覆盖和
  `DictConfig` 泄漏都会被明确拒绝。
- 完成条件：现有 Stochaflow loader 仍作最终解释；Hydra 对象不进入 Builder、manifest、最终
  配置或 checkpoint。

### 消费普通 CLI 与 Hydra 共用的单次训练入口

- 动作：复用显式顺序工作流计划交付的不可变训练调用、extension 预检查、Builder 构造和完整
  单次运行流程；本文只把 Hydra 组合结果交给该入口。
- 原因：两个配置入口不应复制调度代码，也不应产生不同的失败或输出顺序。
- 影响范围：Hydra adapter、extension 激活和配置来源记录；不重新定义训练 Python 入口。
- 交付物：同一基础类型映射可从普通 CLI 或 Hydra 传入唯一的单次训练接口。
- 验证方法：两条路径得到相同的类型化配置、组件身份、checkpoint 语义、manifest 事实和
  失败顺序。
- 完成条件：CLI 和 Hydra adapter 都只做参数转接；训练入口的 request/result 由工作流计划拥有，
  Hydra 不改变工作目录、不创建正式输出、不配置第二套 logger。

### 证明 MNIST 与 AFHQ-v2 在两种入口下行为一致

- 动作：为内置 MNIST 和由 extension 提供的 AFHQ-v2 编写全新训练组合配置，与现有普通 YAML
  配置做有界对照。
- 原因：一个内置案例和一个真实 extension 同时通过，才能证明启动方式没有破坏开放扩展边界。
- 影响范围：仓库配置、extension 项目配置根目录、smoke tests 和审计 manifest。
- 交付物：两条持续维护的配置、最终语义映射和配置来源证据。
- 验证方法：核对原生 provider、extension 来源、inference recipe、manifest 和有界运行结果；
  另在已安装 wheel 环境测试外部 extension。
- 完成条件：Hydra 的配置来源记录可以不同，但最终运行配置和训练语义相同。

### 交付项目模板、配置参考和迁移说明

- 动作：为 extension 项目模板增加可选配置根目录，并说明配置编写、预览/检查、错误处理、与
  普通 YAML 共存及卸载后的回归行为。
- 原因：只有源码仓库环境可用，不等于已安装的 extension 也能编写 Hydra 配置。
- 影响范围：项目模板、配置文档、生成的配置参考和严格文档构建。
- 交付物：已安装 wheel 示例、迁移指南、错误参考和可选依赖说明。
- 验证方法：生成项目后在独立环境运行普通 YAML/Hydra 有界 smoke；移除 Hydra 后普通操作仍
  通过。
- 完成条件：生成的配置参考不会把 Hydra 配置组误写成公共运行时 schema。

## 如何证明已经完成

- 普通 YAML 与 Hydra 全新训练得到相同的最终运行配置和有界结果。
- 预览和检查在 artifact I/O 与正式输出之前给出稳定、可定位的错误。
- 任意 `_target_`/import、两套输出管理和两套 extension 发现机制被测试阻断。
- 外部 extension 与已安装 wheel 路径可用；未安装 Hydra 时普通路径完全可用。
- 公共变化同步 SPEC、ARCHITECTURE、ROADMAP、CHANGELOG、配置与迁移文档。
- sampling review 仍是独立暂停项，不因本候选完成而自动变更。

## 明确不包含什么

- 不用 Hydra 组合严格恢复、sample、Evaluation、资源证据或可观察性恢复配置。
- 不使用 `hydra.utils.instantiate()`、`_target_`、class name 或 arbitrary import path。
- 不把 Registry 全量镜像为配置组，不允许任意递归覆盖。
- 不引入 Sweeper、launcher plugin、自适应 HPO 或通用工作流调度。
- 不把 Physics/KD 清理作为前置或交付。

## 详细设计和研究资料在哪里

- [Hydra 组合迁移设计笔记](notes/hydra-configuration-composition-migration-plan/design-notes.md)
- [显式顺序工作流与单次操作入口](default-workflow-pipeline-support-plan.md)
- [当前配置工作流](../configuration/workflows.md)
- [当前配置兼容与迁移](../configuration/compatibility-and-migration.md)
- [Sampling Invocation Post-Hydra Review](sampling-request-config-refactor.md)
- [Hydra 官方文档](https://hydra.cc/docs/intro/)
- [OmegaConf 官方文档](https://omegaconf.readthedocs.io/)

启动时必须基于当时版本另写短期调研结论。旧实施阶段只留在 Git 历史，不驱动本文。
