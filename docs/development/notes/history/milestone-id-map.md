# 历史计划 ID 映射

- 文档性质：历史索引，不参与五种排期状态
- 用途：解释旧 Roadmap、Git 记录和设计归档中的短 ID
- 当前排期权威：[`ROADMAP.md`](../../../../ROADMAP.md)
- 当前工程说明：[开发方向与执行顺序](../../development-priority-roadmap.md)

这些 ID 不再表示当前优先级、状态或执行顺序。旧文档曾跨计划重复使用字母和数字，甚至出现
同名碰撞。新的 Roadmap 使用完整能力名称；只有查阅旧提交、旧评论或设计归档时才需要本表。

## 1. 已完成主路径的旧 ID

| 旧 ID | 当时含义 | 现在怎样理解 |
| --- | --- | --- |
| `B0` | 建立可运行、可验证的 repository baseline | Done；稳定行为由规范、测试和公开文档拥有 |
| `A0` | 修正 ADM topology 与 checkpoint compatibility | Done；当前 Gaussian architecture 见 `ARCHITECTURE.md` |
| `B1` | 切分 train/sample configuration authority | Done；当前规则见 `SPEC.md` 与配置文档 |
| `A1` | learned-range Gaussian 与数值修正 | Done；当前 family contract 见正式架构和测试 |
| `A2` | class-aware AFHQ Evaluation readiness | Done；当前 profile 见 AFHQ 公开文档 |
| `A3` | learned-range-v AFHQ quality validation | Done；结果由公开 AFHQ 文档、CHANGELOG 与 Git 追溯 |

旧主路径现在统一写成：

```text
pixel Gaussian foundation
  -> standalone Evaluation
  -> live epoch Evaluation and metric-selected checkpoint
  -> AFHQ learned-range-v quality validation
  -> waiting for the next product decision
```

历史 loss-weighting 实验曾使用 `P2` 名称；它已退休，不能与旧 Roadmap 的 P2 优先级混读，
也不构成当前 recipe 或 future implementation condition。

## 2. Evaluation 旧 ID

| 旧 ID | 当时含义 | 现在怎样理解 |
| --- | --- | --- |
| `E0` | structured training outcome foundation | Done；由 training outcome contract 拥有 |
| `E1` | standalone checkpoint Evaluation | Done；由 Evaluation contract 拥有 |
| `E2` | prediction artifact 与 offline replay | Done；由 Evaluation file contract 拥有 |
| `E3` | AFHQ profile，以及旧文档中未拆开的其他 task profiles | AFHQ 部分 Done；future task 必须交付自己的 profile |
| `E4` | 通用结果比较或选择政策 | Parked；只有真实负责人和需求才能开始 |
| `D1`–`D3`（Evaluation 计划） | reference cache、性能优化、结果比较政策 | Parked 的计划内标签，不是全局执行顺序 |

## 3. Latent 与 Stable Diffusion 旧 ID

| 旧 ID | 当时含义 | 现在怎样理解 |
| --- | --- | --- |
| `L0` | pretrained codec readiness | Candidate codec/latent 方向的第一步 |
| `L1` | AFHQ latent correctness path | Candidate codec/latent 方向的一条小型完整路径 |
| `L2` | prepared data、optimizer-step budget 与 codec asset bundle | Candidate latent 方向的生产训练准备 |
| `L3` | open-data small DiT baseline，再进入更大正式运行 | Candidate latent 产品路线 |
| `LD2` / `LD3` | 较早版本中的 codec 与 AFHQ latent 阶段名 | 已由完整能力名称替代 |
| `LD4A` | prepared posterior moments artifact | Candidate latent 方向的生产训练准备 |
| `LD4B` | optimizer-step production loop | Candidate latent 方向的生产训练准备 |
| `LD4C` | run-level codec asset bundle | Candidate latent 方向的生产训练准备 |
| `Q0` | 曾把 Hydra configuration 与 latent Evaluation 合并 | 已拆分；Hydra 独立，latent Evaluation 归具体任务 |
| `S0` / `S1` | Roadmap 中对 SD native sampling 与 training 的粗分组 | 由 Stable Diffusion 计划的分层说明替代 |
| `SD0`–`SD10`（SD 计划） | codec、parity、sampling、training 及后续层级 | Parked 的计划内标签；开始前必须重新核对 |

## 4. Hydra 与 Sampling 旧 ID

| 旧 ID | 当时含义 | 现在怎样理解 |
| --- | --- | --- |
| `C0` / `C1` | plain-YAML train/sample authority cutover | Done；checkpoint-v12 配置归属已稳定 |
| `H0` | Hydra bootstrap、可信 config root 与依赖边界 | Candidate Hydra 方向的计划内步骤 |
| `H1` | library-first training call 与 composed config 入口 | Candidate；workflow 计划复用该库入口，不另建一套 |
| `H2` | preview、检查和受限 override | Candidate Hydra 方向的计划内步骤 |
| `H3` | maintained fresh-training parity 与文档验收 | Candidate Hydra 方向的计划内步骤 |
| `H4` | multirun/sweep | Parked；不属于 Hydra 第一轮工作 |
| `R0`–`R2`（Sampling 复审） | Hydra 完成后的 sampling config 复审步骤 | Parked；Hydra 达到 Done 前不可执行 |

## 5. Workflow、Recipe 与 task 旧 ID

| 旧 ID | 当时含义 | 现在怎样理解 |
| --- | --- | --- |
| `R0`（Workflow 计划） | 术语与 library run API | Candidate workflow 方向；与 Sampling 的同名 ID 无关 |
| `R1` | Recipe manifest 与 first-party catalog | Candidate built-in Recipe 工作 |
| `SR0` | deterministic super-resolution baseline | Candidate SR 方向 |
| `SR1` | SR metrics 与 formal Evaluation | Candidate SR 方向 |
| `SR2` | conditional Gaussian super-resolution | Candidate SR 方向的后续步骤 |
| `LG0` | latent generation Recipe | Candidate；依赖 codec/latent 完整任务 |
| `SD0`（Workflow 计划） | Stable Diffusion Recipe publication | Parked；与 SD 计划的同名层级无关 |
| `CM0` | consistency 计划重新核对 | Candidate consistency 方向 |
| `CM1` | consistency 完整任务路径 | Candidate consistency 方向 |
| `R2`（Workflow 计划） | inference bundle 与重复调用对象 | Candidate workflow 方向的后续步骤 |
| `W0A` | 内置任务与显式顺序组合 | Candidate；具体任务被选择后才实施 |
| `W0B` | 是否需要 core workflow orchestrator | Parked；重复控制逻辑已造成维护问题后再决定 |

## 6. Scale umbrella 与分布式旧 ID

| 旧 ID | 当时含义 | 现在怎样理解 |
| --- | --- | --- |
| `X0` | 把 distributed、HPO、新 family、workflow 和 generic assets 合并的远期集合 | 已拆为独立 Parked 方向，不再作为一个工作项 |
| `D0`–`D7`（Distributed 计划） | characterization、DDP、checkpoint、sampling、FSDP2 与 hardening | Parked 的计划内标签；有 profiling 证据后重新核对 |

## 7. 查阅规则

- 当前 Roadmap、开发导览或执行顺序不再使用旧短 ID。
- 专项计划的设计归档可以保留旧 ID，以便对应原始研究和 Git 讨论；它不改变状态。
- 同名 ID 必须同时带所属计划名称，例如 “Workflow `R0`” 或 “Sampling `R0`”。
- 新工作使用完整名称；如确需机器可读标识，应在被选中后定义带命名空间的 ID。
- 本表只解释历史，不恢复已退休能力，也不改变任何 Candidate 或 Parked 构想。
