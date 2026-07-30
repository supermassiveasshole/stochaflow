# 自定义生成算法 family 的采样侧接入

Stochaflow 只统一 Registry、checkpoint、Builder runtime 和完整的
`Sampler.sample()` 生命周期，不要求不同算法 family 共享数学方法。本教程实现一个最小
vector-field family：

```text
VectorPath Process
  + VectorField Dynamics
  + Euler Sampler
  + VectorFlow SamplingBuilder
```

教程随后增加一个 `process: null` 的 direct-transform Builder，说明没有 model-free
probability path 或 numerical solver 时不需要虚构 Process、Dynamics 或 Sampler。

本页聚焦 sampling-side family contract，不重复 DataBuilder、model 和 TrainingBuilder。
开始前需要一个由项目训练侧生成的 checkpoint：其 primary model 对 vector-field 路径实现
`model(state, time) -> Tensor`，所用完整 config 分别声明匹配的 `VectorPath` 或
`process: null`，并已选择本教程的 `generation-demo` 插件。训练资产如何组成 checkpoint
见[扩展手册](../configuration/extensions.md)；
可直接执行 train → resume → sample 的完整包见
[纵向扩展参考项目](../configuration/reference-projects.md)。本教程中的 CLI 从该前置
checkpoint 开始，不把采样侧代码误称为完整训练项目。

## 1. 注册私有 family 契约

目录：

```text
generation-demo/
├── pyproject.toml
└── src/
    └── generation_demo/
        └── stochaflow_ext/
            ├── __init__.py
            └── sampling.py
```

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "generation-demo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "stochaflow @ https://github.com/supermassiveasshole/stochaflow/releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl",
    "torch>=2.2,<3",
]

[project.entry-points."stochaflow.extensions"]
generation-demo = "generation_demo.stochaflow_ext"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

`src/generation_demo/stochaflow_ext/__init__.py`：

```python
from . import sampling

__all__ = ["sampling"]
```

`src/generation_demo/stochaflow_ext/sampling.py`：

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from stochaflow.extensions import (
    GenerativeDynamics,
    Process,
    REGISTRIES,
    Sampler,
    SamplerResult,
    SamplingBatch,
    SamplingBuilder,
    SamplingObservation,
    SamplingObserver,
    SamplingOutput,
)


@REGISTRIES.processes.register("generation-demo.vector-path")
class VectorPath(Process):
    """A model-free path parameter saved in the Process state_dict."""

    rate: torch.Tensor

    def __init__(self, *, rate: float = 1.0) -> None:
        super().__init__()
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise TypeError("vector path rate must be numeric")
        if rate <= 0:
            raise ValueError("vector path rate must be positive")
        self.register_buffer("rate", torch.tensor(float(rate)))


class VectorField(GenerativeDynamics):
    """Project-private behavior required by the project Euler sampler."""

    def __init__(
        self,
        process: VectorPath,
        predict: Callable[[torch.Tensor, torch.Tensor], object],
    ) -> None:
        self.process = process
        self._predict = predict

    def velocity(
        self,
        state: torch.Tensor,
        coordinate: float,
    ) -> torch.Tensor:
        time = torch.full(
            (state.shape[0],),
            coordinate,
            device=state.device,
            dtype=state.dtype,
        )
        output = self._predict(state, time)
        if not isinstance(output, torch.Tensor):
            raise TypeError("vector-field model must return a Tensor")
        if output.shape != state.shape:
            raise ValueError("vector-field model output must match state shape")
        return output * self.process.rate.to(
            device=state.device,
            dtype=state.dtype,
        )


@REGISTRIES.samplers.register("generation-demo.euler")
class EulerSampler(Sampler):
    """Integrate the project VectorField from coordinate 1 to 0."""

    def __init__(self, *, num_steps: int = 20) -> None:
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("Euler num_steps must be a positive integer")
        self.num_steps = num_steps

    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        del generator
        if not isinstance(dynamics, VectorField):
            raise TypeError(
                "generation-demo.euler requires generation-demo VectorField"
            )
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("Euler initial_state must be a Tensor")
        state = initial_state
        if observer is not None:
            observer.observe(
                SamplingObservation(0, 1.0, state, False, {})
            )
        step_size = 1.0 / self.num_steps
        for step_index in range(1, self.num_steps + 1):
            source = 1.0 - (step_index - 1) * step_size
            state = state - step_size * dynamics.velocity(state, source)
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=1.0 - step_index * step_size,
                        state=state,
                        is_final=step_index == self.num_steps,
                        diagnostics={
                            "num_dynamics_evaluations": step_index
                        },
                    )
                )
        return SamplerResult(
            final_state=state,
            num_steps=self.num_steps,
            diagnostics={
                "num_dynamics_evaluations": self.num_steps
            },
        )


def _sampler_from_params(params: dict[str, Any]) -> tuple[Sampler, str]:
    sampler_value = params.get("sampler")
    if not isinstance(sampler_value, dict):
        raise TypeError("sampler must be a component mapping")
    unknown = sorted(set(sampler_value) - {"name", "params"})
    if unknown:
        raise ValueError("unknown sampler field(s): " + ", ".join(unknown))
    name = sampler_value.get("name")
    constructor_params = sampler_value.get("params", {})
    if not isinstance(name, str) or not name.strip():
        raise ValueError("sampler.name must be a non-empty string")
    if not isinstance(constructor_params, dict):
        raise TypeError("sampler.params must be a mapping")
    sampler = cast(
        Sampler,
        REGISTRIES.samplers.create(name, **constructor_params),
    )
    return sampler, name


@REGISTRIES.sampling_builders.register("generation-demo.vector-flow")
class VectorFlowBuilder(SamplingBuilder):
    """Assemble the project Process, model, Dynamics, initial state and solver."""

    def run(self) -> SamplingOutput:
        params = self.context.params
        unknown = sorted(set(params) - {"weights", "initial_value", "sampler"})
        if unknown:
            raise ValueError(
                "unknown vector-flow parameter(s): " + ", ".join(unknown)
            )
        process = self.context.process
        if not isinstance(process, VectorPath):
            raise TypeError("vector-flow builder requires VectorPath")
        shape = self.context.shape
        if shape is None:
            raise ValueError("vector-flow builder requires sampling.shape")
        weights = params.get("weights", "raw")
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("weights must be auto, raw, or ema")
        initial_value = params.get("initial_value", 0.0)
        if isinstance(initial_value, bool) or not isinstance(
            initial_value, (int, float)
        ):
            raise TypeError("initial_value must be numeric")
        sampler, sampler_name = _sampler_from_params(params)
        model, resolved_weights = self.context.model_provider.resolve(
            cast(Any, weights)
        )
        dynamics = VectorField(
            process,
            lambda state, time: model(state, time),
        )

        batches: list[SamplingBatch] = []
        solver_diagnostics: list[dict[str, Any]] = []
        for offset in range(
            0,
            self.context.num_samples,
            self.context.batch_size,
        ):
            count = min(
                self.context.batch_size,
                self.context.num_samples - offset,
            )
            initial = torch.full(
                (count, *shape),
                float(initial_value),
                device=self.context.device,
            )
            result_value = sampler.sample(dynamics, initial)
            if not isinstance(result_value, SamplerResult):
                raise TypeError("Sampler.sample() must return SamplerResult")
            final_state = result_value.final_state
            if not isinstance(final_state, torch.Tensor):
                raise TypeError("Euler final_state must be a Tensor")
            if final_state.shape != initial.shape:
                raise ValueError("Euler sampler must preserve state shape")
            batches.append(
                SamplingBatch(
                    final_state.detach().to(device="cpu", copy=True)
                )
            )
            solver_diagnostics.append(dict(result_value.diagnostics))

        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "family": "generation-demo.vector-field",
                "weights": resolved_weights,
                "sampler": sampler_name,
                "solver_diagnostics": solver_diagnostics,
            },
        )


@REGISTRIES.sampling_builders.register("generation-demo.direct")
class DirectTransformBuilder(SamplingBuilder):
    """Call a checkpointed model without Process, Dynamics or Sampler."""

    def run(self) -> SamplingOutput:
        params = self.context.params
        unknown = sorted(set(params) - {"weights", "input_dim", "offset"})
        if unknown:
            raise ValueError(
                "unknown direct parameter(s): " + ", ".join(unknown)
            )
        if self.context.process is not None:
            raise TypeError("generation-demo.direct requires process: null")
        weights = params.get("weights", "raw")
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("weights must be auto, raw, or ema")
        input_dim = params.get("input_dim", 1)
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim <= 0
        ):
            raise ValueError("input_dim must be a positive integer")
        offset = params.get("offset", 0.0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise TypeError("offset must be numeric")
        model, resolved_weights = self.context.model_provider.resolve(
            cast(Any, weights)
        )

        batches: list[SamplingBatch] = []
        for start in range(
            0,
            self.context.num_samples,
            self.context.batch_size,
        ):
            count = min(
                self.context.batch_size,
                self.context.num_samples - start,
            )
            inputs = torch.ones(
                (count, input_dim),
                device=self.context.device,
            )
            model_time = torch.zeros(
                count,
                device=self.context.device,
                dtype=torch.long,
            )
            output = model(inputs, model_time)
            if not isinstance(output, torch.Tensor):
                raise TypeError("direct model must return a Tensor")
            if output.shape != inputs.shape:
                raise ValueError("direct model output must match input shape")
            samples = output + float(offset)
            batches.append(
                SamplingBatch(
                    samples.detach().to(device="cpu", copy=True)
                )
            )
        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "family": "direct-transform",
                "weights": resolved_weights,
            },
        )
```

`VectorField` 是这个项目的窄 Dynamics capability。它没有 Registry 或 YAML identity；
Builder 直接创建它。`EulerSampler` 在调用边界要求这个 capability，而不是读取 Process
或 Sampler 的注册名称。由此，核心 runtime 不需要 family 分支或兼容矩阵。

## 2. 使用 vector-field family

先安装 distribution：

```bash
python -m pip install -e ./generation-demo
```

产生目标 checkpoint 的完整训练配置必须包含下面的 Process declaration；训练逻辑由该
family 的 TrainingBuilder 定义，且同样应把 `VectorPath` 作为窄依赖：

```yaml
extensions:
  plugins: [generation-demo]

process:
  name: generation-demo.vector-path
  params:
    rate: 0.25
```

该 family 的 TrainingBuilder 还必须把内部 recipe 固化进 checkpoint：

```python
inference_recipe=SamplingRecipe(name="generation-demo.vector-flow")
```

训练完成后，可用下面的 partial sample request 更换 solver 参数。Process、recipe 和
插件 selection 仍来自 checkpoint：

```yaml
sampling:
  shape: [4]
  num_samples: 16
  batch_size: 8
  seed: 11
  sampler:
    name: generation-demo.euler
    params:
      num_steps: 32
  options:
    weights: raw
    initial_value: 0.0
  writers:
    - name: tensor
      params: {}
```

```bash
stochaflow sample \
  --checkpoint outputs/vector-flow/<run>/checkpoints/best.pt \
  --config experiments/vector-flow/sample.yaml \
  --output-dir outputs/vector-flow-samples
```

如果将 `GaussianModelDynamics` 传给 `generation-demo.euler`，或将 `VectorField` 传给
DDPM/DDIM，family Sampler 会在调用边界明确拒绝。Stochaflow 不承诺这两类数学行为兼容。

## 3. 使用 `process: null` direct transform

若生成算法只是一次精确模型变换，保存该 checkpoint 的完整配置应明确写：

```yaml
extensions:
  plugins: [generation-demo]

process: null
```

所选 TrainingBuilder 也必须支持无 Process 训练，并返回：

```python
inference_recipe=SamplingRecipe(name="generation-demo.direct")
```

partial sample request 继续复用 checkpoint 中的 recipe 和插件 selection：

```yaml
sampling:
  shape: null
  num_samples: 12
  batch_size: 4
  seed: 19
  options:
    weights: raw
    input_dim: 8
    offset: 0.5
  writers:
    - name: tensor
      params: {}
```

```bash
stochaflow sample \
  --checkpoint outputs/direct/<run>/checkpoints/best.pt \
  --config experiments/direct/sample.yaml \
  --output-dir outputs/direct-samples
```

该路径仍使用 checkpoint model provider、统一 `SamplingOutput`、writers 和 resolved
sampling manifest，但完全不创建 Process、Dynamics 或 Sampler。`process: null` 也会
保留在 resolved config；checkpoint 不包含 `process_state_dict`。

这两条路径共同说明边界：`Process + Dynamics + Sampler` 是组织有数值生成过程的 family
的方法，不是每种生成算法必须实现的万能三件套。新增 family 可以定义自己的窄数学接口，
但不得向 `GenerativeDynamics` 根类型加入 `predict()`、`velocity()`、`step()`、`drift()`
或 `denoise()`。
