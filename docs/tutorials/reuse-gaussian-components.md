# 在条件任务中复用离散 Gaussian 组件

本教程实现一个最小的条件 inference recipe。扩展保留内置
`DiscreteGaussianDenoisingProcess`、`GaussianModelDynamics` 和 DDPM/DDIM，只在
`SamplingBuilder` 中组装 condition、classifier-free guidance、初始噪声和模型调用；
训练侧同时把 recipe identity 与固定 prediction semantics 写入 v12 checkpoint。

适用前提是 checkpoint 中的模型已经按以下签名训练：

```text
model(state, model_time, condition) -> Tensor
```

示例 condition 与生成 state 同形，便于展示完整边界。超分任务可以让 condition 使用
不同分辨率，并在模型内部编码；此时应在项目 Builder 中替换本示例“condition 与 state
同形”的校验和扩展逻辑。physics 任务也可以在闭包中加载并预处理物理状态。这些变化都
不需要修改 Process 或 Sampler。

## 1. 创建可安装扩展

目录只需要包含一个普通 Python distribution：

```text
conditional-demo/
├── pyproject.toml
└── src/
    └── conditional_demo/
        └── stochaflow_ext/
            ├── __init__.py
            ├── sampling.py
            └── training.py
```

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "conditional-demo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "stochaflow @ https://github.com/supermassiveasshole/stochaflow/releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl",
    "torch>=2.2,<3",
]

[project.entry-points."stochaflow.extensions"]
conditional-demo = "conditional_demo.stochaflow_ext"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

`src/conditional_demo/stochaflow_ext/__init__.py`：

```python
from . import sampling, training

__all__ = ["sampling", "training"]
```

`src/conditional_demo/stochaflow_ext/sampling.py`：

```python
from __future__ import annotations

from typing import Any, cast

import torch

from stochaflow.extensions import (
    DiscreteGaussianDenoisingProcess,
    GaussianModelDynamics,
    PredictionType,
    REGISTRIES,
    Sampler,
    SamplerResult,
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


@REGISTRIES.sampling_builders.register("conditional-demo.gaussian")
class ConditionalGaussianBuilder(SamplingBuilder):
    """Run conditional Gaussian denoising with a registered DDPM/DDIM sampler."""

    def run(self) -> SamplingOutput:
        params = self.context.params
        allowed = {
            "weights",
            "prediction_type",
            "clip_denoised",
            "guidance_scale",
            "condition",
            "sampler",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                "unknown conditional-demo.gaussian parameter(s): "
                + ", ".join(unknown)
            )

        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "conditional-demo.gaussian requires "
                "DiscreteGaussianDenoisingProcess"
            )
        shape = self.context.shape
        if shape is None:
            raise ValueError("conditional-demo.gaussian requires sample.shape")

        weights = params.get("weights", "auto")
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("weights must be auto, raw, or ema")
        prediction_type_value = params.get("prediction_type", "epsilon")
        if prediction_type_value not in {"epsilon", "x0", "v", "score"}:
            raise ValueError("prediction_type must be epsilon, x0, v, or score")
        prediction_type = cast(PredictionType, prediction_type_value)
        clip_denoised = params.get("clip_denoised", True)
        if not isinstance(clip_denoised, bool):
            raise TypeError("clip_denoised must be boolean")
        guidance_scale = _positive_number(
            params.get("guidance_scale", 1.0),
            label="guidance_scale",
        )

        condition = torch.as_tensor(
            params.get("condition"),
            dtype=torch.float32,
            device=self.context.device,
        )
        if tuple(condition.shape) != shape:
            raise ValueError(
                f"condition shape {tuple(condition.shape)} must equal {shape}"
            )
        if not bool(torch.all(torch.isfinite(condition))):
            raise ValueError("condition must contain only finite values")

        sampler_value = params.get("sampler")
        if not isinstance(sampler_value, dict):
            raise TypeError("sampler must be a component mapping")
        sampler_unknown = sorted(set(sampler_value) - {"name", "params"})
        if sampler_unknown:
            raise ValueError(
                "unknown sampler field(s): " + ", ".join(sampler_unknown)
            )
        sampler_name = sampler_value.get("name")
        sampler_params = sampler_value.get("params", {})
        if not isinstance(sampler_name, str) or not sampler_name.strip():
            raise ValueError("sampler.name must be a non-empty string")
        if not isinstance(sampler_params, dict):
            raise TypeError("sampler.params must be a mapping")
        sampler = cast(
            Sampler,
            REGISTRIES.samplers.create(sampler_name, **sampler_params),
        )

        model, resolved_weights = self.context.model_provider.resolve(
            cast(Any, weights)
        )
        generator = torch.Generator(device=self.context.device)
        generator.manual_seed(self.context.seed)
        batches: list[SamplingBatch] = []
        diagnostics: list[dict[str, Any]] = []

        for offset in range(
            0,
            self.context.num_samples,
            self.context.batch_size,
        ):
            count = min(
                self.context.batch_size,
                self.context.num_samples - offset,
            )
            initial = process.sample_terminal_prior(
                (count, *shape),
                device=self.context.device,
                generator=generator,
            )
            condition_batch = condition.unsqueeze(0).expand(count, *shape)
            null_condition = torch.zeros_like(condition_batch)

            def predict(
                state: torch.Tensor,
                model_time: torch.Tensor,
            ) -> torch.Tensor:
                unconditional = model(state, model_time, null_condition)
                conditional = model(state, model_time, condition_batch)
                if not isinstance(unconditional, torch.Tensor):
                    raise TypeError("unconditional model output must be a Tensor")
                if not isinstance(conditional, torch.Tensor):
                    raise TypeError("conditional model output must be a Tensor")
                if unconditional.shape != state.shape:
                    raise ValueError(
                        "unconditional model output must match the state shape"
                    )
                if conditional.shape != state.shape:
                    raise ValueError(
                        "conditional model output must match the state shape"
                    )
                return unconditional + guidance_scale * (
                    conditional - unconditional
                )

            dynamics = GaussianModelDynamics(
                process,
                predict,
                prediction_type=prediction_type,
                clip_denoised=clip_denoised,
            )
            result_value = sampler.sample(
                dynamics,
                initial,
                generator=generator,
            )
            if not isinstance(result_value, SamplerResult):
                raise TypeError("Sampler.sample() must return SamplerResult")
            final_state = result_value.final_state
            if not isinstance(final_state, torch.Tensor):
                raise TypeError("Gaussian sampler final_state must be a Tensor")
            if final_state.shape != initial.shape:
                raise ValueError("Gaussian sampler must preserve the state shape")
            batches.append(
                SamplingBatch(
                    final_state.detach().to(device="cpu", copy=True)
                )
            )
            diagnostics.append(dict(result_value.diagnostics))

        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "recipe": "conditional-demo.gaussian",
                "weights": resolved_weights,
                "prediction_type": prediction_type,
                "guidance_scale": guidance_scale,
                "sampler": {
                    "name": sampler_name,
                    "params": dict(sampler_params),
                },
                "solver_diagnostics": diagnostics,
            },
        )
```

这个 checkpoint recipe 的内部 Builder 做了四件任务级工作：

1. 从 checkpoint 选择 raw 或 EMA 模型；
2. 创建 condition 和 terminal prior；
3. 用闭包把两次模型调用和 CFG 组合为一个 Gaussian prediction callable；
4. 创建 `GaussianModelDynamics`，再把它交给注册的 Sampler。

`guidance_scale: 1.0` 得到普通条件预测，`0.0` 得到 null-condition 预测。实际模型必须按
训练时的方式定义 null condition；这个约定属于项目，不属于 Stochaflow。

## 2. 安装并选择插件

可以使用任意 Python 包管理器；关键是 distribution 必须安装到运行 `stochaflow` 的同一
Python 环境：

```bash
python -m pip install -e ./conditional-demo
```

产生 checkpoint 的完整训练配置必须已经选择这个插件（以及模型、TrainingBuilder 所需的
其他插件），使 checkpoint 保存确定的 selection/provenance：

```yaml
extensions:
  plugins: [conditional-demo]
```

对应 TrainingBuilder 必须在验证模型、Process 和训练参数后返回 fixed recipe；下面只
展示其 `TrainingPlan` 片段：

```python
from stochaflow.extensions import SamplingRecipe, TrainingPlan

return TrainingPlan(
    strategy=strategy,
    primary_model=self.context.primary_model,
    process=self.context.process,
    objective=self.context.objective,
    inference_recipe=SamplingRecipe(
        name="conditional-demo.gaussian",
        contract={"prediction_type": prediction_type},
    ),
)
```

这样 `prediction_type` 由实际训练组合确定。独立 sample config 不能临时选择这个 Builder，
也不能把 contract 改成另一种 prediction parameterization。

下面是一个完整的独立 sample config。model、Process、recipe 与必要插件 provenance
来自 checkpoint；sampler、options、shape、数量、batch、seed 和 writers 都由本次
invocation 完整声明，不从训练配置继承：

```yaml
sample:
  shape: [1, 2, 2]
  num_samples: 8
  batch_size: 4
  seed: 17
  sampler:
    name: ddim
    params:
      num_inference_steps: 50
      eta: 0.0
  options:
    weights: auto
    clip_denoised: true
    guidance_scale: 1.5
    condition:
      - [[0.0, 0.25], [0.5, 0.75]]
  writers:
    - name: tensor
      params: {}
```

sample config 不是第二份 training experiment config。若确实需要另一个 writer/provider
plugin，可另外写顶层 `extensions.plugins`；该列表只追加到 checkpoint 证明的 required
plugins，不能删除产生 model、Process 或 recipe 的插件，也不能用新插件替换 checkpoint
recipe。

运行：

```bash
stochaflow sample \
  --checkpoint outputs/conditional/<run>/checkpoints/best.pt \
  --config experiments/conditional/sample.yaml \
  --output-dir outputs/conditional-samples
```

同一个 recipe 可以改用 DDPM；复制上一份完整 config，并只把其中 `sample.sampler` 的值
替换为：

```yaml
name: ddpm
params:
  start_time: null
  end_time: 0
```

这段 YAML 只是替换值，不是可单独传给 `--config` 的片段；复制后的文件仍须完整声明
`sample` 的所有 invocation fields。

DDPM 与 DDIM 都只依赖 `GaussianDenoisingDynamics` 行为，不读取 condition、guidance 或
模型签名。若扩展只是改变 Gaussian prediction，例如条件模型、CFG 或能归一化为一致
`clean/epsilon` prediction 的 physics guidance，就应复用这条路径。

若任务需要在一次标准 transition 之后再施加 correction，或者改变 accepted-step、
内部子步或 solver history，则应定义项目私有 Dynamics capability 和匹配 Sampler。可以
继续复用 DDPM/DDIM 公开的 family-specific `transition()` 或 `resolve_schedule()`，
但不应把任务参数加入通用 `Sampler.sample()`。

完整的 physics 条件、partial noising、内置 DDPM/DDIM 复用和 guided-DDIM transition
组合可参考 [Physics reconstruction 参考项目](../configuration/reference-projects.md)。
