# 条件 Gaussian 超分辨率

本教程展示一个最小的**组合路径**：内置 `super_resolution` DataBuilder 负责产生
`(high_res, {"low_res": low_res})`，项目扩展负责解释 condition，并复用 Stochaflow
已有的离散 Gaussian Process、训练 target、Objective 和 DDPM/DDIM。

这不是一个经过质量或收敛验证的超分辨率 baseline。下面的网络只用来说明 API 边界；
真实任务仍需自行选择网络、degradation、归一化、损失、数据规模和评估方法。

## 1. 哪些能力需要扩展

内置 `gaussian_denoising` TrainingBuilder 是无条件 recipe：它只接受 Tensor 或
`(Tensor, {})`，遇到非空 condition mapping 会失败。因此条件超分不能只把
`data.name` 改成 `super_resolution`，还需要三个项目组件：

```text
super_resolution DataBuilder
        │ (high_res, {"low_res": low_res})
        ▼
项目 TrainingBuilder + TrainingStrategy
        │ noisy HR + model time + LR condition
        ▼
项目 conditional model

采样时：
LR condition + inference model + Process
        ▼
项目 SamplingBuilder
        ▼
GaussianModelDynamics
        ▼
内置 DDPM 或 DDIM
```

下例假设插件的 entry-point name 为 `my-sr`，所有 Registry component name 也使用
`my-sr.` namespace。聚合模块应导入三个注册模块：

```python
# my_sr/stochaflow_ext/__init__.py
from . import model, sampling, training  # noqa: F401
```

第三方代码只从 `stochaflow.extensions` 导入稳定契约。

## 2. 注册一个条件模型

模型签名属于项目，而不是 Process 或 Sampler。这个最小模型把 LR condition 插值到 HR
空间，与 noisy HR state 和 time embedding 融合：

```python
# my_sr/stochaflow_ext/model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stochaflow.extensions import REGISTRIES


@REGISTRIES.models.register("my-sr.conditional-denoiser")
class ConditionalDenoiser(nn.Module):
    def __init__(self, *, channels: int = 3, hidden_channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.state_input = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.condition_input = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.time = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.body = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        low_res: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 4 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B, C, H, W]")
        if low_res.ndim != 4 or low_res.shape[:2] != state.shape[:2]:
            raise ValueError("low_res must have shape [B, C, h, w]")
        if low_res.device != state.device or low_res.dtype != state.dtype:
            raise ValueError("low_res must share state device and dtype")
        if model_time.ndim != 1 or model_time.shape[0] != state.shape[0]:
            raise ValueError("model_time must be 1D and match the batch")

        condition = F.interpolate(
            low_res,
            size=state.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        time = self.time(model_time.float().unsqueeze(1))
        time = time.to(dtype=state.dtype).reshape(state.shape[0], -1, 1, 1)
        hidden = self.state_input(state) + self.condition_input(condition) + time
        return self.body(hidden)
```

这个模型输出与 HR state 同形的 Gaussian prediction。它不创建 Process、不采样 timestep，
也不运行 solver。

## 3. 定义条件 Gaussian 训练

TrainingStrategy 解释 batch 并定义一次训练计算：

1. 从 Process 的 noisy time range 采样 `state_times`；
2. 用 `process.sample_marginal()` 得到 noisy HR 和实际噪声；
3. 把 LR condition 传给模型；
4. 用 `gaussian_training_target()` 构造 epsilon/x0/v/score target；
5. 用注入的 Objective 计算标量 loss。

```python
# my_sr/stochaflow_ext/training.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn

from stochaflow.extensions import (
    DiscreteGaussianDenoisingProcess,
    PredictionType,
    REGISTRIES,
    TrainStepOutput,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    compute_objective,
    gaussian_training_target,
)

from .model import ConditionalDenoiser


class ConditionalGaussianStrategy(TrainingStrategy):
    def __init__(
        self,
        model: ConditionalDenoiser,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        *,
        prediction_type: PredictionType,
    ) -> None:
        self.model = model
        self.process = process
        self.objective = objective
        self.prediction_type = prediction_type

    def training_step(self, batch: Any) -> TrainStepOutput:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError("SR batch must be (high_res, conditions)")
        high_res, conditions = batch
        if not isinstance(high_res, torch.Tensor):
            raise TypeError("high_res must be a Tensor")
        if not isinstance(conditions, Mapping):
            raise TypeError("SR conditions must be a mapping")
        low_res = conditions.get("low_res")
        if not isinstance(low_res, torch.Tensor):
            raise TypeError("conditions['low_res'] must be a Tensor")

        state_times = torch.randint(
            self.process.clean_time + 1,
            self.process.terminal_time + 1,
            (high_res.shape[0],),
            device=high_res.device,
        )
        noisy, noise = self.process.sample_marginal(high_res, state_times)
        model_time = state_times - self.process.clean_time - 1
        prediction = self.model(noisy, model_time, low_res)
        target = gaussian_training_target(
            self.process,
            clean=high_res,
            noise=noise,
            state_times=state_times,
            prediction_type=self.prediction_type,
        )
        loss, per_sample = compute_objective(
            self.objective,
            prediction,
            target,
        )
        diagnostics: dict[str, Any] = {"state_times": state_times.detach()}
        if per_sample is not None:
            diagnostics["per_sample_loss"] = per_sample.detach()
        return TrainStepOutput(loss=loss, diagnostics=diagnostics)


@REGISTRIES.training_builders.register("my-sr.gaussian-super-resolution")
class ConditionalGaussianBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        params = dict(self.context.params)
        prediction_type_value = params.pop("prediction_type", "epsilon")
        if prediction_type_value not in {"epsilon", "x0", "v", "score"}:
            raise ValueError("prediction_type must be epsilon, x0, v, or score")
        if params:
            raise ValueError(
                "unknown my-sr training parameter(s): "
                + ", ".join(sorted(params))
            )

        model = self.context.primary_model
        if not isinstance(model, ConditionalDenoiser):
            raise TypeError("my-sr training requires ConditionalDenoiser")
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "my-sr training requires DiscreteGaussianDenoisingProcess"
            )
        objective = self.context.objective
        if objective is None:
            raise TypeError("my-sr training requires an Objective")

        strategy = ConditionalGaussianStrategy(
            model,
            process,
            objective,
            prediction_type=cast(PredictionType, prediction_type_value),
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=model,
            process=process,
            objective=objective,
        )
```

Builder 返回的 `primary_model`、`process` 和 `objective` 必须保留 context 注入的对象；
设备、mode、optimizer、EMA 和 checkpoint 生命周期仍由核心管理。Strategy 只定义 batch
解释、forward 和 loss。

## 4. 用 LR closure 复用 DDPM/DDIM

SamplingBuilder 负责取得 condition、选择 checkpoint 权重、创建 terminal prior，并把
模型调用闭包组合进 `GaussianModelDynamics`。DDPM/DDIM 只看到统一的 Gaussian
Dynamics，不需要知道 `low_res`。

下面约定 `low_res_path` 是一个由 `torch.save()` 写出的 `[N, C, h, w]` 浮点 Tensor；
它必须使用与训练 recipe 相同的通道和归一化方式，且 `N == sampling.num_samples`。

```python
# my_sr/stochaflow_ext/sampling.py
from __future__ import annotations

from pathlib import Path
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
    SamplingObservation,
    SamplingOutput,
)

from .model import ConditionalDenoiser


class CompleteDenoisingObserver:
    """Validate the full terminal-to-clean lifecycle without retaining states."""

    def __init__(
        self,
        process: DiscreteGaussianDenoisingProcess,
        expected_shape: torch.Size,
    ) -> None:
        self.process = process
        self.expected_shape = expected_shape
        self.last_step_index = -1
        self.saw_final = False

    def observe(self, observation: SamplingObservation) -> None:
        if self.saw_final:
            raise ValueError("sampler emitted an observation after final")
        if observation.step_index != self.last_step_index + 1:
            raise ValueError("sampler observation steps must be consecutive")
        state = observation.state
        if not isinstance(state, torch.Tensor) or state.shape != self.expected_shape:
            raise ValueError("sampler observation state has the wrong HR shape")
        if observation.step_index == 0:
            if observation.coordinate != self.process.terminal_time:
                raise ValueError("SR sampling must start at terminal_time")
            if observation.is_final:
                raise ValueError("terminal prior cannot be the final SR result")
        if observation.is_final:
            if observation.coordinate != self.process.clean_time:
                raise ValueError("SR sampling must finish at clean_time")
            self.saw_final = True
        elif observation.coordinate == self.process.clean_time:
            raise ValueError("clean-time observation must be final")
        self.last_step_index = observation.step_index

    def validate_result(self, result: SamplerResult) -> None:
        if not self.saw_final:
            raise ValueError("sampler did not emit a final clean-time observation")
        if result.num_steps != self.last_step_index:
            raise ValueError("SamplerResult does not match the observed lifecycle")


@REGISTRIES.sampling_builders.register("my-sr.gaussian-super-resolution")
class ConditionalSRSamplingBuilder(SamplingBuilder):
    def run(self) -> SamplingOutput:
        params = dict(self.context.params)
        weights = params.pop("weights", "auto")
        prediction_type_value = params.pop("prediction_type", "epsilon")
        clip_denoised = params.pop("clip_denoised", True)
        low_res_path = params.pop("low_res_path", None)
        sampler_value = params.pop("sampler", None)
        if params:
            raise ValueError(
                "unknown my-sr sampling parameter(s): "
                + ", ".join(sorted(params))
            )
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("weights must be auto, raw, or ema")
        if prediction_type_value not in {"epsilon", "x0", "v", "score"}:
            raise ValueError("prediction_type must be epsilon, x0, v, or score")
        if not isinstance(clip_denoised, bool):
            raise TypeError("clip_denoised must be boolean")
        if not isinstance(low_res_path, str) or not low_res_path:
            raise ValueError("low_res_path must be a non-empty path")
        if not isinstance(sampler_value, dict):
            raise TypeError("sampler must be a component mapping")
        if set(sampler_value) - {"name", "params"}:
            raise ValueError("sampler supports only name and params")
        sampler_name = sampler_value.get("name")
        sampler_params = sampler_value.get("params", {})
        if not isinstance(sampler_name, str) or sampler_name not in {
            "ddpm",
            "ddim",
        }:
            raise ValueError("this tutorial builder supports ddpm or ddim")
        if not isinstance(sampler_params, dict):
            raise TypeError("sampler.params must be a mapping")

        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "my-sr sampling requires DiscreteGaussianDenoisingProcess"
            )
        shape = self.context.shape
        if shape is None or len(shape) != 3:
            raise ValueError("sampling.shape must be [C, H, W]")

        model_value, resolved_weights = self.context.model_provider.resolve(
            cast(Any, weights)
        )
        if not isinstance(model_value, ConditionalDenoiser):
            raise TypeError("my-sr sampling requires ConditionalDenoiser")
        model = model_value
        if shape[0] != model.channels:
            raise ValueError("sampling.shape channels must match the model")

        loaded = torch.load(
            Path(low_res_path),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, torch.Tensor):
            raise TypeError("low_res_path must contain one Tensor")
        low_res = loaded
        if (
            low_res.ndim != 4
            or low_res.shape[0] != self.context.num_samples
            or low_res.shape[1] != model.channels
        ):
            raise ValueError(
                "low_res must have shape [sampling.num_samples, C, h, w]"
            )
        if not torch.is_floating_point(low_res):
            raise TypeError("low_res must be floating-point")
        if not bool(torch.all(torch.isfinite(low_res))):
            raise ValueError("low_res must contain only finite values")

        sampler = cast(
            Sampler,
            REGISTRIES.samplers.create(sampler_name, **sampler_params),
        )
        prediction_type = cast(PredictionType, prediction_type_value)
        generator = torch.Generator(device=self.context.device)
        generator.manual_seed(self.context.seed)
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
            condition = low_res[offset : offset + count].to(
                device=self.context.device,
                dtype=torch.float32,
            )
            initial = process.sample_terminal_prior(
                (count, *shape),
                device=self.context.device,
                generator=generator,
            )

            def predict(
                state: torch.Tensor,
                model_time: torch.Tensor,
            ) -> torch.Tensor:
                return model(state, model_time, condition)

            dynamics = GaussianModelDynamics(
                process,
                predict,
                prediction_type=prediction_type,
                clip_denoised=clip_denoised,
            )
            lifecycle = CompleteDenoisingObserver(process, initial.shape)
            result_value = sampler.sample(
                dynamics,
                initial,
                generator=generator,
                observer=lifecycle,
            )
            if not isinstance(result_value, SamplerResult):
                raise TypeError("Sampler.sample() must return SamplerResult")
            lifecycle.validate_result(result_value)
            final_state = result_value.final_state
            if not isinstance(final_state, torch.Tensor):
                raise TypeError("Gaussian sampler final_state must be a Tensor")
            if final_state.shape != initial.shape:
                raise ValueError("Gaussian sampler must preserve the HR shape")
            batches.append(
                SamplingBatch(
                    samples=final_state.detach().to(device="cpu", copy=True)
                )
            )
            solver_diagnostics.append(dict(result_value.diagnostics))

        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "builder": "my-sr.gaussian-super-resolution",
                "weights": resolved_weights,
                "prediction_type": prediction_type,
                "sampler": {
                    "name": sampler_name,
                    "params": dict(sampler_params),
                },
                "solver_diagnostics": solver_diagnostics,
            },
        )
```

这里的 `predict()` closure 捕获当前 LR batch。`GaussianModelDynamics` 负责把模型的
epsilon/x0/v/score output 归一化为 `GaussianPrediction`；DDPM/DDIM 继续负责各自的
transition 和完整求解循环。这个 Builder 的产物语义是完整 SR，因此 lifecycle observer
明确要求从 `terminal_time` 到 `clean_time`：DDPM 的 partial `start_time/end_time` 或
DDIM 未覆盖完整区间的显式 schedule 会失败。需要 partial denoising、image-to-image
strength 或中间态输出时，应由另一个 Builder 构造与起始坐标匹配的 initial state，并定义
自己的输出语义，不能继续把 terminal prior 当作任意 `x_start`。

## 5. 训练配置

下面是一份完整的结构示例。路径和容量只是占位值，不代表推荐实验设置：

```yaml
experiment:
  name: conditional-sr
  seed: 42
  output_dir: outputs/conditional-sr

extensions:
  plugins: [my-sr]

data:
  name: super_resolution
  params:
    source:
      kind: image_folder
      path: data/hr
    partition:
      mode: holdout
      validation_size: 0.1
    image:
      high_resolution: [128, 128]
      low_resolution: [32, 32]
      channels: 3
      normalize: true
      random_horizontal_flip: true
    low_resolution:
      kind: bicubic
    loader:
      batch_size: 8
      num_workers: 0
      shuffle: true
      drop_last: true
      pin_memory: false
      persistent_workers: false
      prefetch_factor: null
      steps_per_epoch: auto

model:
  name: my-sr.conditional-denoiser
  params: {channels: 3, hidden_channels: 64}

training:
  name: my-sr.gaussian-super-resolution
  params: {prediction_type: epsilon}

objective:
  name: mse
  params: {reduction: mean}

process:
  name: discrete_gaussian
  params:
    schedule:
      name: linear_beta
      params:
        num_timesteps: 1000
        beta_start: 0.0001
        beta_end: 0.02

optimizer:
  name: torch.optim.Adam
  params: {lr: 0.0002}

lr_scheduler: null

ema:
  enabled: true
  decay: 0.9995
  update_after_step: 100
  update_every: 1
  use_for_sampling: true

sampling:
  builder: null

diagnostics: []

trainer:
  num_epochs: 30
  device: auto
  max_grad_norm: 1.0
  show_progress: true

logging:
  log_every: 100
  backends:
    - name: local
      params: {console: false}
  torch_logs: {}

artifacts:
  checkpoint_every: 5
```

训练命令：

```bash
stochaflow train --config experiments/sr/train.yaml
```

`sampling.builder: null` 避免训练结束时在没有明确 LR condition 的情况下自动采样。

## 6. 条件采样

先用与训练一致的预处理准备一个低分辨率 Tensor，例如
`data/sample-low-res.pt`。它的 batch 数必须等于 `num_samples`；不要把未归一化输入直接
混入使用 `normalize: true` 训练的模型。

sampling overlay：

```yaml
sampling:
  shape: [3, 128, 128]
  num_samples: 4
  batch_size: 2
  seed: 17
  builder:
    name: my-sr.gaussian-super-resolution
    params:
      weights: auto
      low_res_path: data/sample-low-res.pt
      prediction_type: epsilon
      clip_denoised: true
      sampler:
        name: ddim
        params:
          num_inference_steps: 50
          eta: 0.0
  writers:
    - name: tensor
      params: {}
    - name: image
      params: {grid_nrow: 2, denormalize: true}
```

运行：

```bash
stochaflow sample \
  --checkpoint outputs/conditional-sr/<run>/checkpoints/best.pt \
  --config experiments/sr/sample.yaml
```

改用 DDPM 时只替换 Builder 私有的 sampler declaration：

```yaml
sampler:
  name: ddpm
  params:
    start_time: null
    end_time: 0
```

训练和采样必须使用相同的 `prediction_type`、condition 语义及归一化。DDPM/DDIM 不读取
这些任务字段；它们能够复用，是因为 `GaussianModelDynamics` 已把 condition-aware 模型
适配为两者要求的 `GaussianDenoisingDynamics`。

本教程没有宣称该玩具网络达到任何 PSNR、SSIM、感知质量或科学重建精度。应使用独立
validation/test 数据和任务适合的指标完成自己的实验验收。
