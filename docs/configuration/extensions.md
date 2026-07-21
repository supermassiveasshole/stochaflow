# 扩展与 Registry

顶层扩展通过 `extensions.modules` 显式导入。配置不会扫描目录、Python entry point
或工作区；推荐把用户 project 以 editable package 安装，使模块具有稳定 import path。

## RegistryCatalog

| Registry | 扩展契约 | YAML 选择位置 |
| --- | --- | --- |
| `data_builders` | `DataBuilder` 类 | `data.name` |
| `sampling_artifact_writers` | `SamplingArtifactWriter` 类 | `sampling.writers[].name` |
| `models` | `torch.nn.Module` 类 | `model.name` |
| `noise_schedules` | `GaussianNoiseSchedule` 类 | `process.params.schedule` |
| `processes` | `Process` 类 | `process.name` |
| `samplers` | `Sampler` 类 | SamplingBuilder 专属参数 |
| `sampling_builders` | `SamplingBuilder` 类 | `sampling.builder.name` |
| `training_builders` | `TrainingBuilder` 类 | `training.name` |
| `objectives` | `torch.nn.Module` 类 | `objective.name` |
| `optimizers` | `torch.optim.Optimizer` 类 | `optimizer.name` |
| `lr_schedulers` | scheduler 类或 builder | `lr_scheduler.name` |
| `loggers` | `ExperimentLogger` 类 | `logging.backends[].name` |
| `diagnostics` | `TrainingDiagnostic` 类 | `diagnostics[].name` |

Dataset、source、partition、degradation、PyTorch 数据 sampler、collate 和 DataLoader
不拥有全局 Registry。一个自定义 `DataBuilder` 完整拥有数据组合及其兼容性。生成算法的
`Sampler` 则通过 `REGISTRIES.samplers` 注册，由 SamplingBuilder 组合。

## 模块加载

```python
# my_project/extensions.py
from stochaflow.extensions import REGISTRIES


@REGISTRIES.models.register("physics_operator")
class PhysicsOperator(...):
    ...
```

```yaml
extensions:
  modules:
    - my_project.extensions

model:
  name: physics_operator
  params: {}
```

`load_config()` 和 `load_config_dict()` 在 dataclass 构建后导入模块，再执行跨字段校验。
`RegistryCatalog.load_modules()` 保证同一进程内幂等。注册名必须非空且唯一；错误基类、
重复名称和未知名称会抛出 `RegistryError`。

第三方代码应从 `stochaflow.extensions` 导入稳定契约。数据层只导出 `DataBuilder`、
`DataBuilderContext` 和 `DataLoaders`；内置 recipe 的 source、partition、transform、bucket
与 sampler helper 均为私有实现。

## 自定义 Process、Sampler 与 SamplingBuilder

生成扩展分成三层，层间共享的是生命周期，不是万能数学接口：

| 层 | 统一内容 | 不统一内容 |
| --- | --- | --- |
| 框架 | Registry、配置、checkpoint、完整 `Sampler.sample()` | 算法数学兼容性 |
| 算法 family | 自己需要的 Process、Dynamics 与兼容 Sampler | 跨 family 的 `predict/step/drift` API |
| 任务 | 模型适配、condition、guidance、initial state、artifact | 核心 runner 分支 |

`Process` 根类型保存可注册、可迁移和可 checkpoint 的状态，但它是可选算法资产；算法
family 只在确有 model-free probability path 时通过 Process 子类定义内聚数学能力。
`GenerativeDynamics`
同样只是无行为语义根，不拥有 Registry 或 YAML 身份。`Sampler` 只统一完整 `sample()`
生命周期，不强制提供单步 `step()`；`SamplingBuilder` 解释具体任务并装配 initial state、
模型 callable、condition/guidance、family Dynamics、Sampler 和 observer。

当前生产实现只有 Gaussian family：

```text
DiscreteGaussianDenoisingProcess
    + GaussianDenoisingDynamics
    + DDPMAncestralSampler / DDIMSampler
```

未来 flow matching 可以定义自己的 probability path、VectorField Dynamics 和 ODE
Sampler；score SDE 或 sigma-space 方法也可以定义自己的窄契约。它们不需要实现 Gaussian
接口，也不会促使核心为 `GenerativeDynamics` 增加 universal `predict`、`drift`、`score`
或 `denoise` 方法。

`GaussianDenoisingDynamics` 是 DDPM/DDIM 依赖的抽象能力；内置
`GaussianModelDynamics` 是普通 model callable 的具体 adapter。Process 不提供 Dynamics
工厂方法，也不依赖模型 callable、prediction parameterization、clipping policy 或具体
Dynamics 类型。Builder、diagnostic 或用户推理工作流负责显式构造/包装 Dynamics。

`noise_schedules` Registry 只服务 Gaussian Process，不是 Flow Matching interpolant 的
万能 Schedule Registry。离散 Gaussian Process 依赖 `DiscreteVPSchedule` 数学能力；内置
linear/cosine 实现继承 `TabulatedDiscreteVPSchedule`，第三方实现可以用其他存储或解析
方式提供 `GaussianScales` 与 `DiscreteVPCoefficients`，无需暴露 coefficient table。
Schedule 返回与 `state_times` 同形的系数，样本 rank 的 broadcast 由 Process 负责。
`DiscreteGaussianProcess` 在构造时复制完整 coefficient snapshot，运行时不保留 schedule
子模块；检测到 `Parameter`、需要梯度或非法系数会明确失败。marginal、posterior、device
迁移和 checkpoint 因而始终读取同一份 Process-owned state。

内置 `gaussian_denoising` TrainingBuilder、`standard_denoising` SamplingBuilder、DDPM
和 DDIM 都依赖公开的 `DiscreteGaussianDenoisingProcess`。第三方实现该 Process 接口即可
复用这些组件，无需继承 `DiscreteGaussianProcess`；训练 timestep sampling policy 由
Gaussian TrainingStrategy 拥有。

```python
from stochaflow.extensions import (
    DiscreteGaussianDenoisingProcess,
    GaussianModelDynamics,
    REGISTRIES,
    Sampler,
    SamplerResult,
    SamplingBuilder,
    SamplingOutput,
)


@REGISTRIES.samplers.register("project_solver")
class ProjectSolver(Sampler):
    def __init__(self, *, tolerance: float):
        self.tolerance = tolerance

    def sample(self, dynamics, initial_state, *, generator=None, observer=None):
        final_state, steps, diagnostics = solve(
            dynamics,
            initial_state,
            tolerance=self.tolerance,
            generator=generator,
            observer=observer,
        )
        return SamplerResult(final_state, steps, diagnostics)


@REGISTRIES.sampling_builders.register("physics_sampling")
class PhysicsSamplingBuilder(SamplingBuilder):
    def run(self) -> SamplingOutput:
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError("physics_sampling requires discrete Gaussian process")
        dynamics = GaussianModelDynamics(
            process=process,
            predict_fn=build_guided_predict_fn(...),
            prediction_type="epsilon",
            clip_denoised=True,
        )
        # Builder also owns observations, initial state and Sampler execution.
        ...
```

`Sampler.sample()` 不接收 shape、low-res、mask、guidance 或 solver-specific keyword。
shape/initial state 属于 Builder，condition/guidance 通过模型 callable 闭包进入 Dynamics，
solver 参数属于 Sampler 构造配置，trajectory 由 `SamplingObserver` 收集。
Sampler 负责决定 lifecycle 时机、创建完整的 `SamplingObservation` 并直接调用
`observer.observe(observation)`；Observer 只消费事件，决定筛选、复制和记录。
`TrajectoryObserver` 保留 initial、指定间隔和 final observation。Sampler 不提供伪通用
的起止坐标查询；`standard_denoising` 与 image diagnostic 通过真实 observation 验证完整
terminal-to-clean 路径。基于观测或其他中间状态的 partial denoising 应由自定义
SamplingBuilder 构造 initial state，或直接调用 Sampler。

### 复用 Gaussian family 完成 condition 或 physics guidance

如果任务仍满足 Gaussian denoising 行为契约，condition 不进入 Sampler API。Builder 可以
把 low-resolution、mask、CFG 或 physics state 捕获在模型 callable 中，也可以包装
`GaussianDenoisingDynamics`，在每次 Sampler 查询生成方向时修正 prediction：

```python
from stochaflow.extensions import (
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
)


class PhysicsGuidedDynamics(GaussianDenoisingDynamics):
    def __init__(self, process, model, low_res, physics_state):
        self._process = process
        self._delegate = GaussianModelDynamics(
            process,
            lambda state, time: model(state, time, low_res=low_res),
            prediction_type="epsilon",
            clip_denoised=True,
        )
        self.physics_state = physics_state

    @property
    def process(self):
        return self._process

    def predict(self, state, state_times):
        prediction = self._delegate.predict(state, state_times)
        return apply_physics_guidance(
            prediction,
            state=state,
            state_times=state_times,
            process=self.process,
            physics_state=self.physics_state,
        )
```

`apply_physics_guidance()` 必须返回彼此一致的 clean/epsilon prediction。只要这个行为契约
成立，同一个 Dynamics 可以交给 DDPM 或 DDIM。若 DFSR correction 改变的是 reverse
transition、accepted-step 更新或 solver 内部子步，而不能表达为 Gaussian prediction
修正，则应在项目扩展中提供匹配的窄 Dynamics 和自定义 Sampler；不要给内置 Process、
DDPM/DDIM 或顶层 YAML 增加 physics 专用参数。

### 接入新的算法 family

新的 family 仍复用框架 Registry 和完整 sampling lifecycle，但定义自己的数学契约：

```python
from stochaflow.extensions import (
    GenerativeDynamics,
    Process,
    REGISTRIES,
    Sampler,
    SamplingBuilder,
)


@REGISTRIES.processes.register("project_flow_path")
class ProjectFlowPath(Process):
    ...


class ProjectVectorField(GenerativeDynamics):
    def velocity(self, state, time):
        ...


@REGISTRIES.samplers.register("project_ode")
class ProjectODESampler(Sampler):
    def sample(self, dynamics, initial_state, *, generator=None, observer=None):
        if not isinstance(dynamics, ProjectVectorField):
            raise TypeError("project_ode requires ProjectVectorField")
        ...


@REGISTRIES.sampling_builders.register("project_flow_sampling")
class ProjectFlowSamplingBuilder(SamplingBuilder):
    def run(self):
        # Validate ProjectFlowPath, assemble ProjectVectorField, then invoke
        # the registered project ODE sampler and return SamplingOutput.
        ...
```

兼容性检查位于项目 Builder 和 family Sampler 的调用边界。核心不读取 family 名称，不维护
Process/Sampler 组合表，也不注册 Dynamics。

### 不使用 Process 或 Sampler 的生成方法

`SamplingBuilderContext.process` 是可选值。没有 model-free probability path 或 numerical
solver loop 的方法可以直接在 Builder 中调用模型并返回 `SamplingOutput`：

```python
from stochaflow.extensions import (
    REGISTRIES,
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)


@REGISTRIES.sampling_builders.register("direct_transform")
class DirectTransformBuilder(SamplingBuilder):
    def run(self):
        if self.context.process is not None:
            raise TypeError("direct_transform does not use a Process")
        model = self.context.model_provider.get("raw")
        samples = run_exact_transform(model, self.context.params)
        return SamplingOutput((SamplingBatch(samples.cpu()),), {"kind": "direct"})
```

对应配置省略 `process` 或写 `process: null`；resolved config 和 checkpoint 不保存
`process_state_dict`。这条路径不需要占位 Process、Dynamics 或 Sampler，也不需要核心增加
按算法名称分支。训练侧是否需要 Process 由所选 TrainingBuilder 在组合边界校验。

## 自定义 TrainingBuilder 与 TrainingStrategy

训练侧的注册入口是 `TrainingBuilder`，而不是 Strategy。Builder 接收核心预先构建的
primary model、可选 Process/Objective 和辅助 factory，返回完成依赖注入的
`TrainingPlan`。`TrainingStrategy` 只是普通训练计算对象：解释 structured batch、调用
注入的模型和 Objective，并返回 `TrainStepOutput(loss, metrics, diagnostics)`。

```python
from stochaflow.extensions import (
    ManagedTrainingModule,
    REGISTRIES,
    TrainStepOutput,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
)


class ConditionalStrategy(TrainingStrategy):
    def __init__(self, model, objective):
        self.model = model
        self.objective = objective

    def training_step(self, batch):
        high_res, conditions = batch
        prediction = self.model(high_res, low_res=conditions["low_res"])
        return TrainStepOutput(self.objective(prediction, high_res))


@REGISTRIES.training_builders.register("conditional_sr")
class ConditionalSRBuilder(TrainingBuilder):
    def build(self):
        objective = self.context.objective
        if objective is None:
            raise TypeError("conditional_sr requires objective")
        return TrainingPlan(
            strategy=ConditionalStrategy(self.context.primary_model, objective),
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=objective,
        )
```

Strategy 不拥有 `to/train/eval`、optimizer、factory、parameter selection 或 checkpoint
API。Plan 通过稳定名称声明辅助 `nn.Module`，核心统一管理 device、mode、优化和持久化。

### Frozen-teacher 蒸馏

蒸馏不需要 Trainer 的任务分支。自定义 Builder 构建并加载 teacher、关闭其梯度，再用
稳定名称把它交给 Plan；Strategy 只定义 student/teacher forward 和损失组合：

```python
import torch


class DistillationStrategy(TrainingStrategy):
    def __init__(self, student, teacher, task_objective, distill_objective, alpha):
        self.student = student
        self.teacher = teacher
        self.task_objective = task_objective
        self.distill_objective = distill_objective
        self.alpha = alpha

    def training_step(self, batch):
        inputs, targets = batch
        with torch.no_grad():
            teacher_output = self.teacher(inputs)
        student_output = self.student(inputs)
        task_loss = self.task_objective(student_output, targets)
        distill_loss = self.distill_objective(student_output, teacher_output)
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        return TrainStepOutput(
            total_loss,
            metrics={
                "task_loss": task_loss.detach(),
                "distill_loss": distill_loss.detach(),
            },
        )


class DistillationBuilder(TrainingBuilder):
    def build(self):
        if self.context.objective is None:
            raise TypeError("distillation requires a task objective")
        teacher = build_and_load_teacher(self.context)
        teacher.requires_grad_(False)
        distill_objective = build_distillation_objective(self.context)
        strategy = DistillationStrategy(
            self.context.primary_model,
            teacher,
            self.context.objective,
            distill_objective,
            alpha=0.5,
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=self.context.objective,
            auxiliary_modules={
                "teacher": ManagedTrainingModule(teacher, mode="eval"),
                "distill_objective": ManagedTrainingModule(distill_objective),
            },
        )
```

核心会迁移并 checkpoint teacher，保持其 eval mode；`requires_grad=False` 使 teacher 不进入
optimizer。offline distillation 可直接由 DataBuilder 提供 teacher target，因此无需
auxiliary teacher。feature、logit 或 score distillation 也采用同一边界，只需在具体
Strategy 内解释输出并返回单一标量总 loss。

Stage 4 的自动生命周期只支持一个 optimizer 和一次 backward。独立 teacher optimizer、
交替更新、EMA teacher 或 manual backward 属于新的训练 loop family，不通过向 Strategy
增加生命周期开关来模拟。

### 复用 Gaussian diagnostic

`diffusion_quality` 不猜测 primary model 的调用签名。希望复用它的 Gaussian Strategy
必须满足稳定的 `GaussianDiagnosticSemantics` 窄能力：

```python
from stochaflow.extensions import GaussianDiagnosticSemantics


class ConditionalGaussianStrategy(TrainingStrategy, GaussianDiagnosticSemantics):
    @property
    def prediction_type(self):
        return "epsilon"

    def predict_gaussian_model(self, state, model_time):
        return self.model(
            {"state": state, "time": model_time},
            condition=self.diagnostic_condition,
        )
```

这是结构化 Protocol，不要求 Strategy 显式继承它；示例继承用于类型标注和能力发现。
`prediction_type` 说明 Gaussian parameterization，`predict_gaussian_model()` 封装任务自己的
模型签名。diagnostic 用该 callable 与 Process 构造 Dynamics，不直接调用 primary model。

如果条件任务无法为 diagnostic 提供明确的 condition，就不应伪造该 capability；启用
`diffusion_quality` 时会得到清晰的不兼容错误。该能力不让 Strategy 构建 Sampler、运行
diagnostic 或管理 artifact，因此不改变 Strategy 的训练计算职责。

## 自定义 DataBuilder

```python
from torch.utils.data import DataLoader

from stochaflow.extensions import DataBuilder, DataLoaders, REGISTRIES


@REGISTRIES.data_builders.register("physics")
class PhysicsDataBuilder(DataBuilder):
    def build(self) -> DataLoaders:
        params = self.context.params
        train_dataset, validation_dataset = build_physics_datasets(
            params["path"],
            seed=self.context.seed,
        )
        return DataLoaders(
            train=DataLoader(
                train_dataset,
                sampler=PhysicsSampler(train_dataset),
                collate_fn=physics_collate,
            ),
            validation=DataLoader(validation_dataset),
            steps_per_epoch=params.get("steps_per_epoch"),
        )
```

```yaml
extensions:
  modules: [my_project.extensions]
data:
  name: physics
  params:
    path: data/simulation.zarr
    steps_per_epoch: 1000
```

`DataBuilderContext.params` 是调用方配置的深拷贝，`seed` 是实验种子。`build()` 必须
返回一个 `DataLoaders`；所有 loader 必须可重复迭代，不能直接返回一次性的 generator。
train loader 通过自身 `len()` 或显式 `steps_per_epoch` 给出有限 epoch。
validation/test 可以是未知长度的可重复 iterable。

每个 epoch 开始时，Trainer 会对 train loader 的 `sampler` 和 `batch_sampler` 做去重后
的 duck-typed `set_epoch(epoch)` 调用。其他 Dataset 或 DataLoader 属性只用于
best-effort 报告，不属于扩展契约。

## 自定义 sampling artifact writer

```python
from stochaflow.extensions import REGISTRIES, SamplingArtifactWriter


@REGISTRIES.sampling_artifact_writers.register("netcdf")
class NetCDFWriter(SamplingArtifactWriter):
    def __init__(self, *, variable: str):
        self.variable = variable

    def write(self, context):
        path = context.output_dir / "samples.nc"
        write_netcdf(path, context.batches, variable=self.variable)
        return {"netcdf": path}
```

```yaml
sampling:
  shape: [64, 64, 4]
  writers:
    - name: tensor
      params: {}
    - name: netcdf
      params: {variable: velocity}
```

writer 返回的 mapping 不能为空；artifact key 在全部 writer 间必须唯一，路径必须已经
存在。任何 writer 失败都会使采样失败。内置 `tensor` 支持任意 rank Tensor；内置
`image` 才校验 NCHW、1/3 通道，并拥有 `grid_nrow`、`gif_fps` 与 `denormalize` 参数。

## 其他构造约定

组件 `params` 通常作为关键字参数传给注册类。以下参数由运行时注入：

- DataBuilder：`DataBuilderContext`；
- TrainingBuilder：`TrainingBuilderContext`；
- SamplingBuilder：`SamplingBuilderContext`；
- optimizer / LR scheduler：模型参数或 optimizer；
- logger：`output_dir`、`run_name`；
- diagnostic：`logger`、`output_dir`、可选 `sample_shape`。

`diffusion_quality` 的 provider 仍使用其局部 `diagnostics[].params.modules` 机制。
所有内置名称与构造参数见[生成式组件索引](reference.md#registry-组件索引)。
