# 自动化模型调优：Provider 调研、API 草案与历史映射

> 本附录是 [`自动化模型调优计划`](../../automated-model-tuning-plan.md) 的时效性材料。
> 它不定义当前 API，也不证明任何依赖在当前 lockfile、平台或许可证约束下可用。
> 实施启动前必须重新核对上游公开文档、受支持版本、恢复语义和安全边界。
> 最后核对：2026-08-09

## 调研问题

重新启动调研时至少回答：

1. provider 是否把 search algorithm、trial scheduler、resource placement 和 storage 清楚
   分层；Stochaflow 能否只实现窄 adapter。
2. provider 是否支持 ask/tell 或 function trainable、epoch report、prune、失败分类和同一
   study restore。
3. Grid、Random、TPE 与 GP-based BO 的真实实现身份、seed 和 conditional-space 行为能否
   写入 provenance。
4. controller restart、worker restart、trial checkpoint restore 与 study storage 各自保证
   什么；哪些只是 best-effort。
5. optional dependency 是否可与当前 Python、PyTorch、Windows/Linux 和 CUDA 组合安装。
6. provider 是否把对象序列化、动态 import、dashboard、remote code 或网络服务带入默认
   路径；能否显式关闭。
7. license、维护活跃度、发布频率、漏洞处理和长期 lockfile 成本是否可接受。

## 候选 provider 轴线

| 候选 | 可能的价值 | 必须重新验证的风险 | 候选定位 |
| --- | --- | --- | --- |
| Ray Tune | search、scheduler、resource、restore 与 process isolation 较完整 | 依赖体量、版本耦合、对象序列化、Windows/CUDA、restore 细节 | 完整 engine adapter |
| Optuna | ask/tell、sampler、pruner、storage 较窄 | process/resource orchestration 需另有 owner；分布式 storage 语义 | 轻量 engine 或 Ray search backend |
| Hydra Sweeper | 与 Hydra authoring 接近 | 不能成为 training runtime；multirun 与本计划启动顺序未定 | 只作为 authoring/launcher 候选 |
| W&B Sweeps | 托管 observability 与 agent workflow | 网络、账户、offline/resume、artifact authority 与供应商锁定 | 可选 integration，不是 core dependency |
| KerasTuner | 研究 API 与策略参考 | 面向 Keras lifecycle，不能直接拥有 Stochaflow training | 设计参考，不预设 provider |
| Ax / BoTorch | 约束、多目标、batch BO 与研究级 acquisition | API/版本变化、数据模型复杂度、engine 职责缺失 | 外部 BO provider 候选 |
| Bayesian Optimization package | 经典 GP-BO 的轻量基线 | conditional/constraint/storage/parallel 能力有限 | 小型独立 provider 候选 |

成熟方案优先不等于预先选择 Ray 或 Optuna。启动时应以一个真实 workload 做最小
spike，并把拒绝理由和取舍写入新的短期实施记录。不得把多个 provider namespace 镜像成
Stochaflow registry，也不得为统一表面重写完整 controller。

## 独立 BO 仓库路线

独立 BO 仓库是保留的外部 provider 路线，不是 Stochaflow core 的新算法包。只有通用
candidate/observation contract 已稳定、该 optimizer 能被至少一个非 Stochaflow workload 独立
复用，而且 Grid/Random 与一个成熟 provider 已证明 study lifecycle 正确后，才重新审查接入。

| 独立 BO 仓库负责 | Stochaflow 负责 |
| --- | --- |
| parameter-domain 数学表示、initial design、surrogate/noise model | 合法 config patch、trial/run identity 与资源隔离 |
| acquisition 与 sequential/batch candidate generation | DataBuilder/TrainingBuilder/Trainer 生命周期 |
| BO state、算法 benchmark 与 regret 分析 | validation objective、study resume、artifact lineage 与 provenance |

首选接入方式是该仓库提供成熟 engine 的标准 searcher adapter；若它确实需要无该 engine 的
native host，再评审窄 `CandidateGenerator`/`StudyBackend` adapter。Stochaflow core 不反向依赖
该仓库，外部算法也不读取完整训练配置、checkpoint 或 task-private batch。

## 搜索与预算备忘

- Grid：只接受有限离散值。适合回归、消融和小空间 correctness oracle。
- Random：高维 sanity baseline。必须记录 sampler seed 和 domain serialization。
- TPE：适合混合或 conditional domain；不是经典 Gaussian-process BO 的别名。
- GP-based BO：适合较低维、昂贵 objective；必须记录 kernel/acquisition/provider identity，
  并明确 noisy objective 假设。
- Scheduler/pruner：控制资源分配和提前停止，与候选生成算法正交。
- Hyperband/ASHA：需要统一 resource step、最小资源和最大预算，不能从 scheduler constructor
  字段名猜训练语义。
- Trainer early stopping 是单个 trial 内的纵向停止策略，budget pruner 是 trial 间的横向比较。
  启用 Hyperband/ASHA 等 budget pruner 时默认关闭前者；若两者同时声明且没有明确仲裁规则，
  Builder 必须 fail closed，不能让完成顺序隐式决定结果。
- Constraints、batch BO、multi-objective：只有 provider contract、serialization、resume 和
  result comparison 全部明确后才开放。
- Population-based training：会中途改写配置、复制 checkpoint，属于不同 lifecycle，不是
  本计划的自然延伸。

预算至少区分：最大 trial 数、study timeout、每 trial epoch 上限、train/validation batch
上限、并发上限和 retry policy。epochs 是首个 resource step 候选；scheduler horizon 仍由
完整 training config 声明，通用 tuner 不通过参数名推导 `T_max` 或 warmup。

## 配置草案

以下仅用于启动时讨论字段所有权，不能直接生成 parser：

```yaml
study:
  name: ddpm-mnist-baseline
  output_dir: outputs/tuning/ddpm-mnist-baseline
  seed: 20260725

base_config: ../ddpm_mnist.yaml

objective:
  metric: valid/metrics/reconstruction_mse
  direction: minimize

parameters:
  learning_rate:
    target: /optimizer/params/lr
    distribution:
      type: float
      low: 0.00001
      high: 0.001
      log: true
  batch_size:
    target: /data/params/loader/batch_size
    distribution:
      type: categorical
      choices: [64, 128, 256]
  model_channels:
    target: /model/params/model_channels
    distribution:
      type: int
      low: 32
      high: 128
      step: 32

budget:
  max_trials: 30
  timeout_seconds: null
  epochs_per_trial: 50
  max_train_batches: null
  max_validation_batches: null
  run_phase_test: false

engine:
  provider: candidate-provider
  params: {}
  max_concurrent_trials: 1
  resources_per_trial:
    cpu: 4
    gpu: 1
```

约束草案：

- study schema 与 `StochaflowConfig` 分开解析；不得成为其 superset。
- target 使用 JSON Pointer 风格绝对路径，不允许表达式、插值、通配符、`..` 或 Python。
- target 默认必须已存在；patch 后重新运行完整 validation。
- 一个 target 只能由一个 parameter 拥有。
- `/extensions`、output identity、experiment identity 和普通 seed 不得成为 patch target。
- device、worker、GPU、timeout、epoch 上限属于 resource/budget，不属于 model search space。
- component `name` 的 heterogeneous branch 首个切片禁止；复杂 conditional graph 交给窄
  Python extension，而不是通用 `when/then` YAML。
- provider-specific params 不能重复声明 adapter 已拥有的 search space。
- secret 不进入 frozen config、fingerprint 或普通 manifest。

## API 草案

名称和签名都未冻结；它们只说明职责分离：

```python
@dataclass(frozen=True, slots=True)
class StudyRequest:
    identity: str
    base_config: Mapping[str, object]
    parameters: tuple[ParameterDomain, ...]
    objective: ObjectiveSpec
    budget: StudyBudget
    seed: int


@dataclass(frozen=True, slots=True)
class TrialRequest:
    study_identity: str
    trial_identity: str
    assignment: Mapping[str, object]
    run_seed: int
    resource_limit: int


@dataclass(frozen=True, slots=True)
class EpochObservation:
    epoch: int
    canonical_validation: Mapping[str, float]
    checkpoint: str | None


class TrialObserver(Protocol):
    def report(self, observation: EpochObservation) -> TrialDecision: ...


class TuningEngine(Protocol):
    def run(self, request: StudyRequest, executor: TrialExecutor) -> StudyOutcome: ...
```

责任草案：

- `TuningBuilder` 验证完整 collaboration 并返回 immutable plan；不启动 controller。
- `TrialExecutor` 只消费 [Hydra 配置组合迁移计划](../../hydra-configuration-composition-migration-plan.md)
  交付的普通 library-first training invocation；HPO 不另建入口，也不解析 CLI 或日志。
- `TrialObserver` 只消费 canonical validation evidence；不成为 Diagnostic。
- engine 产生 suggestion、拥有 scheduler/pruner 和 engine storage；不构建模型或 loader。
- launcher 分配 process/device；不选择超参数。
- study policy 选择候选；Evaluation runtime 不承担 promotion 决策。

## Frozen study 与 artifact 草案

首次启动时可能冻结：resolved base config、resolved study config、parameter domain、objective、
budget fidelity、extension provenance、Stochaflow/Python/PyTorch/provider 版本和可发现的 VCS
状态。版本号不证明 working tree 没有变化；fingerprint 必须明确包含和排除哪些内容。

```text
study-root/
├── study_manifest.yaml
├── resolved_study.yaml
├── base_config.yaml
├── provider_state/
├── trials/
│   ├── trial-000000/
│   │   ├── trial_manifest.yaml
│   │   ├── resolved_config.yaml
│   │   ├── checkpoints/
│   │   └── metrics.jsonl
│   └── trial-000001/
└── best_trial.yaml
```

`best_trial.yaml` 只保存事实和 pointers，不复制或覆盖 checkpoint。发布另设显式 promotion。
study storage 不替代 trial artifact storage；engine journal 也不成为普通 checkpoint 字段。

Resume 可考虑只允许增加 trial/timeout、改变 launcher resource 和显式 retry policy。以下内容
必须冻结：parameter target/domain、objective、base config、extension identity、study seed 和
fidelity。controller restore、trial retry 和 trial checkpoint resume 必须是三个可区分动作。

## Seed、重复与统计

至少区分：

- study/search seed：控制 suggestion 序列；
- run seed：控制模型初始化、数据顺序和训练随机性；
- replication seed：为同一 assignment 生成独立重复；
- diagnostic seed：固定 sampling/reference comparison，使不同 trial 的观察噪声可比较；它不能
  与 run seed 混用，也不能让 Diagnostic 成为 selection 或 pruning evidence。

单次深度训练 objective 通常有噪声，不能默认声明 deterministic。统计确认候选可以保存
mean、dispersion、置信区间和失败数，但原始 run 与 assignment 仍是审计单位。多目标只有在
Pareto representation、missing/non-finite policy 和 promotion owner 明确后启用。

## Failure、停止、恢复与泄漏

- `pruned` 是正常提前结束，不能归类为 failed。
- pruned trial 的 checkpoint retention 必须由 artifact policy 显式声明；建议默认保留最后一个
  轻量 latest checkpoint 与终止 observation 供审计，但它不得成为 study best 或覆盖 completed
  trial artifact。若 provider 无法给出一致 retention 语义，首个结果应禁用 pruning。
- config error、OOM、I/O、extension、provider、controller 和 user cancellation 必须可区分。
- 未产生合法 objective 的 completed run 仍不得成为成功 trial。
- retry 只能恢复或重跑同一 assignment，不能静默生成新 candidate。
- study timeout 停止新 trial；已运行 trial 的 stop policy 必须显式。
- SIGINT 后先冻结 controller state，再协调 worker；不得发布半写 manifest。
- test split、Diagnostic、非冻结 offline report 和人工挑选结果不得反馈给 search。
- preprocessing、data identity、split、metric implementation 与 protocol identity 必须冻结，
  防止不同 trial 实际优化不同问题。

## Selection、final refit 与正式 Evaluation

study 只发布 best-trial fact 与原 artifact pointer；promotion 不复制或覆盖 trial checkpoint。
选定 subject 后才运行一次独立 formal Evaluation，test 结果不反馈给当前 study。

若用户要求用 train+validation 对选中配置做 final refit，负责该任务数据组合的具体
`DataBuilder` 必须提供新的、明确命名且可解析的 recipe/config。HPO 只绑定其 identity、生成一个
新的 run 并记录从 best trial 到 refit artifact 的 lineage；core 不推断 split、不合并任意
Dataset，也不把原 best-trial checkpoint 伪装成 refit 结果。refit 完成后冻结新的 subject，再
交给 formal Evaluation。

## 并行与外部执行

首个 slice 应保持 `max_concurrent_trials=1`。之后的本地并行使用独立 process，避免 mutable
runtime、CUDA context、Registry 和 extension selection 相互污染。资源调度至少处理 GPU
独占、CPU/worker 预算、临时目录、端口、controller/worker failure 和 orphan cleanup。

Cluster、remote actor 或 managed service 是 launcher/provider 能力。它们不得改变 trial
contract，也不得把普通 artifact authority 交给 dashboard。并行 trial 与 distributed
training 是两个正交维度；未明确二维资源模型时禁止叠加。

## 最小验证矩阵草案

- Config：合法 domain、unknown field、非法 pointer、重复 target、类型和越权 patch。
- Provider：fake adapter、Grid/Random oracle、seed、report、prune、failure 和 restore。
- Execution：独立 runtime、observer 顺序、immutable outcome、artifact publication。
- Resume：相同 fingerprint 接受；objective、space、base config 或 extension 变化拒绝。
- Evidence：validation-only、NaN/Inf、missing objective、test leakage 和 protocol mismatch。
- Resource：并发上限、device collision、worker crash、controller restart 和 cleanup。
- Statistics：replication identity、aggregation、partial failure 和 multi-objective rejection。

## 历史阶段映射

以下 ID 只帮助阅读旧 Git 历史，不驱动当前计划：

| 历史 ID | 当时表达的主题 | 当前归宿 |
| --- | --- | --- |
| E0 | immutable single-run outcome 基础 | 当前事实；不是 HPO task |
| E1 | standalone Evaluation 基础 | 当前事实；future policy 见 Evaluation 记录 |
| T0 | single-run seam | 正文“训练调用面”任务卡 |
| T1 | study config 与 Grid/Random | 正文“Study authority”和“顺序执行”任务卡 |
| T2 | adaptive search 与 scheduler | 正文“自适应搜索与 pruning”任务卡 |
| T3 | local process workers | 正文“本地并行与外部 launcher”任务卡 |
| T4 | replication 与 multi-objective | 正文“Objective、预算与统计”任务卡 |
| T5 | external launcher/provider | 正文“本地并行与外部 launcher”任务卡 |

不要从这些 ID 推断排期、依赖已满足或 API 已获批。

## 上游入口

- [Ray Tune documentation](https://docs.ray.io/en/latest/tune/)
- [Optuna documentation](https://optuna.readthedocs.io/en/stable/)
- [Hydra Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/)
- [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps/)
- [KerasTuner](https://keras.io/keras_tuner/)
- [Ax](https://ax.dev/)
- [BoTorch](https://botorch.org/)
- [Bayesian Optimization](https://bayesian-optimization.github.io/BayesianOptimization/)

这些链接只用于重新调研的起点。实施记录应引用启动时实际审核的版本化文档或源码。
