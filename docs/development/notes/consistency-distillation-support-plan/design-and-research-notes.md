# Consistency Distillation 设计与研究附录

> 本文保存 2026-08-09 主计划重基线前的数学、文献、候选 API、配置、
> 实验矩阵和历史实施清单。它不是当前能力、排期、公共 API 或验收权威；启动时必须按
> 当前 [`SPEC.md`](../../../../SPEC.md)、[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md)
> 和 [`ROADMAP.md`](../../../../ROADMAP.md) 重新验证。可执行结论见
> [`consistency-distillation-support-plan.md`](../../consistency-distillation-support-plan.md)。

## 原始设计资料

- 文档性质：开发草案；不属于当前公开 API 或正式文档导航
- 状态：Later / rebase required，尚未进入实现；不再以退出维护的 extension
  reference project 作为实施入口
- 高层排期：[根级 Roadmap](../../../../ROADMAP.md)；候选依赖与入口见
  [Development Priority Roadmap](../../development-priority-roadmap.md)
- 制定日期：2026-07-23
- 架构复核：2026-07-25；标准 distillation 首版改为独立 target student EMA；
  2026-07-26；质量/性能验收接入独立 Evaluation cases
- 首版范围：离散 VP Gaussian、无条件图像、冻结 diffusion teacher、端点一致性算子
- 实施位置：未来先作为临时、可安装的 extension prototype 与 contract fixture 验证；
  不恢复长期维护的 reference project，也不直接扩大核心算法接口
- 关联决策：
  [默认工作流与可组合任务支持计划](../../default-workflow-pipeline-support-plan.md)、
  [训练后 Evaluation 与 Benchmark 支持计划](../../post-training-evaluation-support-plan.md)

## 1. 目标与结论

目标是从一个已经训练好的 diffusion teacher 中蒸馏出 student operator
\(F_\theta\)。对于 teacher 的同一条确定性生成轨迹上的任意状态
\((x_t,t)\)，student 都应返回同一个 clean endpoint：

\[
F_\theta(x_t,t)
\approx
F_\theta(x_s,s),
\qquad 0 \leq s < t \leq T.
\]

首版把这个想法解释为 **endpoint consistency model**，而不是学习任意
\(t\rightarrow s\) 变换的通用轨迹算子。也就是说：

\[
F_\theta:(x_t,t)\mapsto \hat x_{\mathrm{end}},
\]

所有时间的输出都落到同一个 canonical endpoint。若后续目标变为
\(G_\theta(x_t,t,s)\mapsto x_s\)，则应作为 Consistency Trajectory Model
方向单独设计，不能在首版接口中暗加第二个时间参数。

首版同时实现两种不同的 clean 约束：

1. **硬边界：**
   \(F_\theta(x_0,0)=x_0\)。它由 operator 参数化精确保证，不需要 loss。
2. **clean reconstruction anchor：**
   对非零噪声状态增加
   \(d(F_\theta(x_t,t),x_0)\)。它是可配置、可调度、必须消融的辅助 loss，
   不是 consistency boundary 的另一种写法。

推荐先在临时、可安装的独立 extension prototype 中完成纵向验证。仓库中已有的
frozen-teacher architecture fixture 只能证明通用组合边界；它不承担 consistency
算法支持，也不作为新 workflow 的实现入口。

## 2. 调研结论

### 2.1 与当前想法最接近的方法

[Consistency Models](https://proceedings.mlr.press/v202/song23a.html)
将 consistency function 定义为：把 probability-flow ODE 同一轨迹上的任意点映射到
该轨迹的起点，并要求任意两个时刻的映射相等。论文还要求
\(f(x_\epsilon,\epsilon)=x_\epsilon\) 的 identity boundary，并在蒸馏时用冻结
diffusion model 产生相邻轨迹点，再匹配 online model 与 stop-gradient target model
的输出。这与本计划的主目标一致。

[Consistency Trajectory Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c204d12afa0175285e5aac65188808b4-Abstract-Conference.html)
进一步学习任意 source time 到 destination time 的轨迹跳转。它适合未来的
\(G(x_t,t,s)\) 需求，但会引入 target-time conditioning、更多组合约束和不同的
sampling contract，因此不进入首版。

[Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)
主要研究不依赖 diffusion teacher 的 consistency training，提出移除 CT target EMA、
Pseudo-Huber metric、lognormal noise sampling 和逐步增加离散步数等改进。首版仍是
distillation，因此不能把 CT 的无 EMA 结论直接套到 CD。首版采用独立 target student
EMA，同时采用 Pseudo-Huber 作为可选 distance，并把 timestep 分布和 EMA decay 做成
显式实验变量。

[Consistency Models Made Easy](https://openreview.net/forum?id=xQVxo9dSID&noteId=3skELU0A5Q)
说明从预训练 diffusion model 初始化 student、再逐步加强 consistency 条件，可以显著
降低训练成本。首版因此默认要求 teacher/student 可严格对齐，并提供从 teacher 权重
初始化 student 的路径。

[Elucidating the Preconditioning in Consistency Distillation](https://openreview.net/forum?id=55pCDKiS8B)
强调 preconditioning 对边界、稳定性和 teacher 轨迹对齐的重要性。首版先复用
Stochaflow 已有 Gaussian prediction normalization 构造 clean endpoint，同时把更强的
analytic preconditioning 留作稳定性检查后的独立阶段。

[Progressive Distillation](https://arxiv.org/abs/2202.00512) 的目标是反复把采样步数减半，
不是让所有时刻直接映射到统一端点；它作为速度/质量对照组，而不是首版算法定义。

### 2.2 对 \(x_0\) 额外约束的判断

硬边界与监督 anchor 必须分开：

- \(F(x_0,0)=x_0\) 是无歧义的 identity，可由代码严格保证；
- 对所有 \(t>0\) 强制 \(F(x_t,t)=x_0\) 是从 noisy observation 重建训练样本。

若对后者使用平方损失，则其总体最优解是
\(\mathbb E[x_0\mid x_t]\)，而不是每个输入都可唯一恢复到产生它的训练样本。当
\(t\) 很大、信息被噪声抹去时，过强的 \(x_0\) loss 可能产生均值化、降低多样性，
并与 teacher probability-flow trajectory 的 endpoint target 产生梯度冲突。

因此首版采用以下政策：

- clean anchor 默认开启但权重较小；
- 默认在低噪声/高 SNR 区域权重更大，在接近 terminal noise 时衰减；
- `clean_anchor.weight: 0` 必须恢复纯 consistency distillation；
- 同时记录 consistency gap、clean reconstruction gap 和 teacher-to-clean gap；
- 只有消融证明 clean anchor 改善重建且不明显损伤生成质量后，才提高默认权重。

## 3. 首版数学方案

### 3.1 状态与 teacher

首版要求现有 `DiscreteGaussianDenoisingProcess`：

\[
x_t = \alpha_t x_0 + \sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\qquad t\in\{0,\ldots,T\}.
\]

冻结的 diffusion teacher 记为 \(D_\phi\)。它使用已有
`GaussianModelDynamics` 将 teacher 的 `epsilon`、`x0`、`v` 或 `score` 原始输出
统一为 `GaussianPrediction(clean, epsilon, model_output)`。

Teacher 轨迹 pair 的首版近似为：

1. 从数据取 \(x_0\)；
2. 采样一对 grid 上相邻时间 \(s<t\)；
3. 用 `process.sample_marginal` 得到 \(x_t\)；
4. 用冻结 teacher 预测 `GaussianPrediction`；
5. 调用经 ownership review 后的 Gaussian family-specific deterministic selected-pair
   transition primitive，得到确定性的 \(x_s^\phi\)。现有 sampling-owned
   `DDIMSampler.transition(..., eta=0)` 可作为行为 oracle，但 training 不得反向依赖
   sampling policy。

这里复用 [DDIM](https://arxiv.org/abs/2010.02502) 的确定性 transition 作为离散 VP
首版轨迹近似，不宣称它等价于连续时间高阶 PF-ODE solver。后续只有在实验显示
teacher discretization error 成为瓶颈时，才增加 Heun 或连续时间 Process family。

### 3.2 Student endpoint operator

首版不向 `GenerativeDynamics` 根添加通用 `predict` 或 `operator` 方法，而在 extension
内定义窄能力：

```python
class EndpointConsistencyDynamics(GenerativeDynamics, Protocol):
    @property
    def process(self) -> DiscreteGaussianDenoisingProcess: ...

    def predict_endpoint(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> torch.Tensor: ...
```

运行时 wrapper 组合 student model、Process 和 prediction semantics：

\[
F_\theta(x_t,t)=
\begin{cases}
x_t, & t=0,\\
\operatorname{Clean}\!\left(
  x_t,t,N_\theta(x_t,t-1)
\right), & t>0.
\end{cases}
\]

`Clean(...)` 首版复用 `normalize_gaussian_prediction(...).clean`。这样：

- \(t=0\) 不执行网络，identity boundary 精确成立；
- student 可从 prediction type 相同的 diffusion teacher 严格初始化；
- consistency 与 clean anchor 都在 clean-image 空间计算；
- 训练时默认 `clip_denoised=False`，避免 clamp 截断梯度；
- sampling 输出是否 clamp 由 sampling builder 的私有参数决定。

如果 terminal time 出现除以很小 signal 导致的数值放大，则按顺序评估：

1. teacher/student 改用 `v` 或 `x0` prediction；
2. 增加显式 `c_skip(t)x+c_out(t)N_\theta(x,t)` preconditioner；
3. 最后才研究 analytic preconditioning。

这些策略属于 endpoint operator，不修改 Gaussian Process 或基础 `UNet` 的职责。

### 3.3 Loss

对每个 batch：

\[
\begin{aligned}
y_t &= F_\theta(x_t,t),\\
y_s^- &= \operatorname{stopgrad}
        \left(F_{\bar\theta}(x_s^\phi,s)\right),\\
\mathcal L_\text{cons}
    &= w_\text{cons}(s,t)\,d_\text{cons}(y_t,y_s^-),\\
\mathcal L_{x_0}
    &= w_{x_0}(t)\,d_{x_0}(y_t,x_0),\\
\mathcal L
    &= \mathcal L_\text{cons}
       +\lambda_{x_0}\mathcal L_{x_0}.
\end{aligned}
\]

首版默认：

- `d_cons`：Pseudo-Huber；同时保留 MSE 作为基线；
- `w_cons=1`；
- `w_x0(t)`：按 log-SNR 平滑衰减，高 SNR 权重大、低 SNR 权重小；
- `lambda_x0`：配置项，必须包含 `0` 基线；
- lower branch 全程 `torch.no_grad()`；
- clean anchor 复用 \(y_t\)，不增加 student forward。

建议最小消融矩阵：

| 维度 | 值 |
| --- | --- |
| clean anchor | `weight = 0, 0.01, 0.1, 1.0` |
| anchor schedule | `constant`, `snr_gate` |
| distance | `mse`, `pseudo_huber` |
| pair grid | `18`, `36`, `80` 个离散状态 |
| target policy | `ema_target`；`online_stopgrad` 只作明确命名的实验消融 |

### 3.4 单个训练 step

```text
clean x0
  └─ sample adjacent pair (s, t) from the distillation grid
       └─ process.sample_marginal(x0, t) -> xt
            ├─ frozen diffusion teacher + deterministic DDIM transition -> xs_teacher
            │    └─ stop-gradient target operator -> ys
            └─ online student endpoint operator -> yt
                 ├─ consistency distance(yt, ys)
                 └─ clean anchor metric(yt, x0)
                      └─ one scalar total loss -> existing Trainer
```

一次训练计算需要一个 frozen teacher forward、一个 online student forward 和一个
target student forward；当 \(s=0\) 时 target operator 是 identity，不需要第三次网络
调用。clean anchor 不额外调用模型。

## 4. Stochaflow 组件边界

| 责任 | 组件 | 决策 |
| --- | --- | --- |
| 数据与 batch | 现有 image `DataBuilder` | 返回 clean image；核心不知道 `x0` 字段 |
| Gaussian probability path | 现有 `DiscreteGaussianDenoisingProcess` | 保持 model-free，不加入蒸馏逻辑 |
| Frozen diffusion teacher | `ConsistencyDistillationTrainingBuilder` 构造的 auxiliary module | Builder 加载、冻结、声明 `mode="eval"` |
| Target consistency student | Builder 构造的 frozen auxiliary + core-managed EMA relation | Strategy 只读；成功 optimizer step 后由 core 更新并 checkpoint |
| Teacher transition | 待提取的 Gaussian family-specific primitive | built-in DDIM 与 extension 共用；training 不依赖 sampling policy，也不复制公式 |
| Student endpoint semantics | extension-local `EndpointConsistencyDynamics` | 组合 model、Process、prediction type 和 clean boundary |
| Loss 计算 | `ConsistencyDistillationStrategy` | 解释 batch、调用已注入对象、合成一个 scalar loss |
| Optimization objective | 顶层 Objective + Builder 构造的 clean Objective | consistency Objective 为顶层 Objective；clean Objective 是具名 auxiliary |
| Optimizer/device/mode/checkpoint | 现有 core Trainer | Strategy 不移动、冻结、保存或更新资产 |
| One-step inference | `ConsistencySamplingBuilder` 直接调用 endpoint operator | 单次 student forward，报告 `NFE=1` |
| Few-step sampling loop | extension-local `ConsistencySampler` | 拥有 denoise–renoise lifecycle |
| Sampling composition | `ConsistencySamplingBuilder` | 只加载 student，组合 operator、sampler、writer output |

禁止的实现方式：

- 不新增通用 `KnowledgeDistillationStrategy` 注册表或万能 distillation schema；
- 不在 runner 中根据 consistency 注册名添加分支；
- 不让 Process 持有 teacher、student 或 sampling loop；
- 不把 `x0`、image、condition 字段加入核心 batch contract；
- 不复制 DDIM selected-pair 数学，也不让 training 依赖 sampling-owned Sampler；
- 不让 checkpoint-only sampling 构造 teacher 或 TrainingBuilder；
- 不为首版预埋 CTM 的可选 `target_time`、adversarial loss 或多 optimizer 模式。

## 5. Teacher artifact 与恢复语义

首版使用 extension 自己的、数据化的 teacher bundle，而不是让 extension 依赖
Stochaflow 内部 checkpoint schema。提供
`tools/export_teacher_bundle.py`，从已完成的 teacher run 导出：

```text
format_version
model declaration
model state dict (raw or EMA，导出时已经选定)
process declaration
process state dict / coefficient fingerprint
prediction_type
source checkpoint lineage
```

Builder 的规则：

1. fresh train 时读取 bundle，构造 teacher 并严格加载；
2. teacher `requires_grad_(False)` 且固定 `eval`；
3. 比较 bundle process state 与当前 Process state，schedule 不匹配立即失败；
4. `student_init: teacher` 时对 primary model 严格加载相同权重，不做静默 partial load；
5. student 架构不兼容时必须显式改为 `student_init: config`；
6. teacher、target student 与 clean Objective 以稳定名称作为 auxiliary modules；
7. strict resume 时，当前 checkpoint 的 auxiliary state 覆盖 fresh bootstrap；
8. checkpoint-only sampling 只加载 primary student 与可选 inference EMA，不需要 teacher
   bundle 或 target student 存在。

Teacher bundle 属于本地训练输入，不提交大型权重、数据集或生成 artifact。

## 6. Target network 政策

Consistency distillation 还需要区分：

- **diffusion teacher \(D_\phi\)：** 冻结模型，定义 trajectory pair；
- **consistency target \(F_{\bar\theta}\)：** lower branch 的 stop-gradient endpoint
  predictor。

当前 Trainer 的 EMA 是 primary model 的 inference shadow，不是可在 Strategy 中调用的
target model。标准 consistency distillation 首版采用：

```text
target_policy: ema_target
```

Builder 创建与 online student 架构严格相同的 frozen target student，fresh run 时 exact
copy online state，并把它作为具名 auxiliary 交给 Strategy 只读调用。一个窄的
core-managed EMA relation 声明：

```text
source: primary_model
target: target_student
decay: constant scalar
update: after each successful optimizer step
```

core 负责 update 时序、target state、update counter 和 checkpoint/resume；Strategy
不得在 `training_step()` 内更新 EMA，也不能复用 inference EMA。strict resume 时
checkpoint target 覆盖 fresh bootstrap。

`no_grad()` 不会关闭 dropout，也不会阻止 train-mode buffer 更新。最小首版要求：

- `dropout=0`；
- 不包含 BatchNorm 一类会在 train mode 更新 running state 的模块；
- 使用当前 `UNet` 的 GroupNorm 路径；
- EMA 是否覆盖 buffer、target mode 与 mixed-precision update dtype 在实现前冻结。

若后续支持 non-zero dropout，必须显式同步 online/target forward 的 dropout RNG。
`online_stopgrad` 可作为明确命名的研究消融，但不能代表标准 consistency
distillation，也不能成为默认公开配置。

## 7. Sampling 方案

`EndpointConsistencyDynamics` 不要求 Gaussian Sampler 或 `GenerativeDynamics` 根
新增方法。one-step 是直接 endpoint transform；只有 few-step 使用 extension-local
`ConsistencySampler`。

### One-step

\[
x_T\sim\mathcal N(0,I),
\qquad
\hat x_0=F_\theta(x_T,T).
\]

SamplingBuilder 直接调用 operator，并报告 `NFE=1`；不为了统一形式构造 numerical
Sampler。

### Few-step

给定降序时间 \(T=t_K>\cdots>t_1>0\)：

1. 当前状态调用 endpoint operator 得到 \(\hat x_0\)；
2. 若还有下一步，用 Process 以新的独立噪声把 \(\hat x_0\) re-noise 到 \(t_{k-1}\)；
3. 再次调用 endpoint operator；
4. 最后在 clean time 返回结果。

Sampler 负责时间表、随机数、step count、observer event 和临时状态；
SamplingBuilder 负责 initial prior、模型权重选择、shape、batching 和结果 metadata。

必须输出：

- 实际 NFE / `num_steps`；
- resolved time schedule；
- raw 或 EMA student weights；
- fixed seed 下的 deterministic replay 信息；
- 可选 trajectory preview，但不扩大核心 artifact 容量边界。

## 8. 配置草案

下面只描述 extension 私有参数，不修改顶层 schema：

```yaml
extensions:
  plugins:
    - stochaflow-consistency-distillation

model:
  name: unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 64

process:
  name: discrete_gaussian
  params:
    schedule:
      name: linear_beta
      params:
        num_timesteps: 1000

objective:
  name: stochaflow-consistency-distillation.pseudo-huber
  params:
    delta: 0.03

training:
  name: stochaflow-consistency-distillation.training
  params:
    teacher:
      name: unet
      params:
        in_channels: 1
        out_channels: 1
        base_channels: 64
    teacher_bundle: data/mnist-teacher.pt
    prediction_type: epsilon
    student_init: teacher
    target_policy: ema_target
    target_ema:
      decay: 0.999
      update_after_step: 0
      update_every: 1
    pair_schedule:
      num_steps: 36
      sampling: uniform_adjacent
    clean_anchor:
      objective:
        name: stochaflow-consistency-distillation.pseudo-huber
        params:
          delta: 0.03
      weight: 0.1
      schedule: snr_gate
      params:
        midpoint_log_snr: 0.0
        sharpness: 1.0

ema:
  enabled: true
  decay: 0.9999
```

训练配置不再内嵌 sampling。独立完整 sample invocation：

```yaml
sample:
  sampler:
    name: stochaflow-consistency-distillation.consistency
    params:
      schedule: [1000, 0]
  options:
    weights: ema
    clip_output: true
  shape: [1, 28, 28]
  num_samples: 36
  batch_size: 36
  seed: 42
  writers:
    - name: image
      params:
        grid_nrow: 6
        denormalize: true
```

对应 TrainingBuilder 必须把内部
`stochaflow-consistency-distillation.sampling` recipe 和
`prediction_type: epsilon` fixed contract 写入 `TrainingPlan.inference_recipe`；
sample request 不能选择 Builder 或覆盖 prediction type。最终 recipe option 字段以
实现阶段的严格 parser 和测试为准；未知字段必须报错。

## 9. 历史实施清单（仅用于追溯）

### 算法与实验契约冻结

交付：

- 本计划评审通过；
- 明确首版是 endpoint CM，不是 CTM；
- 明确 teacher checkpoint、prediction type、dataset 和第一条基线；
- 固定 clean anchor 消融矩阵与随机 seed。

退出条件：

- 维护者确认 \(x_0\) anchor 是辅助重建项，而不是宣称 teacher trajectory endpoint
  必然等于产生 noisy sample 的那一张 \(x_0\)。

### Extension skeleton 与 teacher bundle

交付：

- 新建临时、可安装的 extension prototype 与独立 contract fixtures；只有能力达到
  promotion gate 后才决定是否成为 maintained example；
- 注册 namespaced Objective、TrainingBuilder、Sampler、SamplingBuilder；
- 实现 teacher bundle exporter/loader 与 process fingerprint 验证；
- 提供 tiny config 和不含权重的 README 流程。

测试：

- bundle 字段、dtype、key 和版本严格校验；
- model/process 不匹配时 fail fast；
- teacher 冻结、eval、无梯度；
- student strict initialization；
- plugin entry point 安装与注册。

### Endpoint operator 与 pair construction

交付：

- `EndpointConsistencyDynamics`；
- \(t=0\) identity boundary；
- teacher `GaussianModelDynamics`；
- family-specific deterministic selected-pair stepper；
- distillation grid parser 和 pair sampler。

测试：

- boundary bitwise identity；
- 任意 batch shape、dtype、device 与 state time 校验；
- teacher transition 不构建完整 sampling loop；
- fixed seed 得到相同 pair；
- terminal time 输出有限、无 NaN/Inf；
- pair 只允许 \(0\le s<t\le T\)。

### TrainingBuilder、Strategy 与 clean anchor

交付：

- Pseudo-Huber Objective；
- `ConsistencyDistillationTrainingBuilder`；
- `ConsistencyDistillationStrategy`；
- consistency、clean anchor、总 loss 和 diagnostics；
- target student 构造与 strict bootstrap；
- core-managed target EMA relation。

测试：

- 只有 primary student 参数进入 optimizer；
- teacher/target lower branch 没有梯度；
- target 与 online state keys/shape 严格一致；
- target 只在成功 optimizer step 后更新；
- target state、update counter 和 policy 可 strict resume；
- 首版拒绝 non-zero dropout 和 train-mode mutable buffers；
- total loss 是一个 floating scalar；
- `clean_anchor.weight=0` 与纯 consistency 路径等价；
- SNR schedule 单调、有限且范围正确；
- tiny dataset 可过拟合，两个 loss 都按预期下降；
- strict resume 后 teacher、target 与 auxiliary Objective state 由 checkpoint 恢复。

### Student-only sampling

交付：

- one-step direct endpoint path；
- extension-local few-step consistency sampler；
- SamplingBuilder、observer 和 metadata；
- checkpoint-only sampling。

测试：

- one-step 精确执行一次 student forward，NFE 为 1；
- fixed seed deterministic；
- final shape/dtype/device 正确；
- few-step observer 从 terminal 到 clean 只发合法 accepted states；
- sampling 时移除 teacher bundle 仍可运行；
- sampling 不构造 target student；
- raw/EMA student 权重选择与 manifest 一致。

### 稳定性与 target EMA policy gate

实验：

- 比较 target EMA decay、update cadence；`online_stopgrad` 只作命名消融；
- 记录 loss variance、gradient norm、collapse、FID/KID 与 NFE；
- 检查 terminal timestep 的 clean conversion 放大；
- 比较 MSE 与 Pseudo-Huber；
- 比较 18/36/80-step grid。

决策：

- 冻结默认 target EMA decay 与 update policy；
- 若窄 core lifecycle 无法满足正确 checkpoint/step 时序，转为专用
  consistency-distillation loop family，而不是由 Strategy 更新；
- preconditioning 是瓶颈：先在 extension operator 内改进，不修改 Process/UNet。

### 质量评估与文档

最小实验顺序：

1. tiny synthetic/tiny image overfit；
2. MNIST teacher 与 student；
3. AFHQ-v2 作为质量/性能 showcase，但不冒充规模结果；
4. 条件生成、CFG 或高分辨率留到后续提案。

对每个数据集报告：

- teacher 参考：DDIM 1/2/4/8/50-step；
- student：consistency 1/2/4/8-step；
- FID、KID、precision/recall；
- fixed timestep 的 clean MSE/PSNR；
- NFE、batch latency、peak memory；
- `lambda_x0=0` 与 clean anchor 各组消融；
- 相同初始噪声的定性 sample grid。

这些结果通过独立 Evaluation Operation 产生，而不是由 Metric、final sampling 或
TrainingDiagnostic 单独冒充完整 benchmark。student 的 1/2/4/8-step 分别是预先声明的
EvaluationCase，共用 checkpoint/weight variant、sample IDs、seed bank、metric
provider 和硬件 profile；每个 case 同时记录实际 `forward_calls` 与
`effective_model_evaluations`。训练期按 cadence 运行的同类 FID/KID 仍属于
Diagnostic context，正式 test result 不参与 checkpoint 选择。

文档交付：

- extension README；
- teacher export、fresh train、resume、sample 命令；
- 算法公式与配置参考；
- 已知限制和复现实验表；
- 不提交 checkpoint、dataset 或普通 outputs。

## 10. 验收标准

### 功能验收

- 同一 teacher trajectory pair 可计算有限的 consistency loss；
- \(F(x_0,0)=x_0\) 精确成立；
- clean anchor 可开启、关闭和调度；
- teacher 始终冻结，student 可训练；
- checkpoint/resume 可复现训练状态；
- checkpoint-only one/few-step sampling 不依赖 teacher artifact；
- 新 extension 不需要修改 core runner dispatch。

### 架构验收

- Builder 是 teacher、Objective、operator 和 Strategy 的 composition root；
- Strategy 只解释 batch、forward 和 loss；
- Process 仍然 model-free；
- Sampler 拥有完整 few-step loop；
- `GenerativeDynamics` 根没有新增 universal math；
- 新组件通过 Registry/config/extension plugin 进入；
- `uv run ruff check .`、`uv run pyright` 与 focused tests 通过。

### 研究验收

- `lambda_x0=0` 基线和至少两个非零权重完成同 seed 对比；
- clean anchor 对 reconstruction 的收益与 FID/recall 的代价都有记录；
- student 在 1/2/4/8 NFE 上给出速度–质量曲线；
- 与 teacher 同 NFE 和 teacher 高 NFE 两类基线同时比较；
- 若未达到质量目标，能通过 diagnostics 区分 teacher pair error、consistency error、
  clean-anchor conflict 与 preconditioning instability。

不在计划阶段承诺绝对 FID 数值；先由固定 teacher、dataset、硬件和 seed 建立可复现
baseline，再设置 promotion gate。

## 11. 主要风险与缓解

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| clean anchor 与 teacher endpoint 冲突 | consistency loss 降、FID/recall 变差 | 低权重、SNR gate、显式 `weight=0` 消融 |
| trivial/collapsed mapping | 输出接近常数 | hard identity boundary、sample diversity/recall 监控 |
| teacher Process 不一致 | pair 数学错误但仍可运行 | bundle process fingerprint 严格比较 |
| terminal clean conversion 不稳定 | NaN、梯度爆炸 | finite tests、grad clip、v/x0 prediction、preconditioning |
| target EMA lifecycle 错误 | loss variance 大、resume 漂移 | core-managed successful-step update 与 strict state |
| teacher step discretization error | grid 加密仍无法提升 | 比较 grid/solver，必要时增加高阶 teacher stepper |
| 多步 CM 质量不单调 | NFE 增加但质量下降 | 报告完整曲线；若需要任意跳转，转 CTM 提案 |
| 内存占用 | teacher + online + target + inference EMA | 记录 peak memory；按阶段释放无需调用的资产 |
| extension 侵入核心 | runner 出现任务分支 | 以现有 Builder/Strategy/Sampler contract 完成纵向实现 |

## 12. 后续但不属于首版

- `G(x_t,t,s)` Consistency Trajectory Model；
- continuous-time VP/VE Process 与 Heun PF-ODE teacher stepper；
- non-constant target EMA schedule 与可训练 buffer policy；
- analytic preconditioning；
- conditional model、classifier-free guidance 和 guided teacher；
- latent consistency、text-to-image 和多模态 batch；
- adversarial/perceptual loss；
- offline cached teacher pairs；
- distributed teacher/student execution；
- progressive distillation 对照实现。
