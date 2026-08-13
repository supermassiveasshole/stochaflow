# 按类别验证与自定义 Metric

本教程为普通多类别任务增加按类别 recall 和 macro recall。它只依赖 Strategy 明确提供的
class prediction channel，不要求特定图像数据集、模型家族或 batch 字段，也不会让核心
猜测 label。

最终 validation 结果为：

```text
valid/metrics/class_recall/class_0
valid/metrics/class_recall/class_1
valid/metrics/class_recall/class_2
valid/metrics/class_recall/class_3
valid/metrics/class_recall/macro
```

`macro` 可以用于 best checkpoint；逐类别结果用来确认整体均值没有掩盖某一类别的退化。

## 1. 注册按类别 Metric

在 extension distribution 中增加一个模块，例如
`src/class_eval/stochaflow_ext/metrics.py`：

```python
from __future__ import annotations

from collections.abc import Mapping

import torch
from torchmetrics import Metric

from stochaflow.extensions import REGISTRIES


@REGISTRIES.metrics.register("class-eval.per-class-recall")
class PerClassRecallMetric(Metric):
    """Compute recall for every declared class and their macro mean."""

    correct_by_class: torch.Tensor
    total_by_class: torch.Tensor

    def __init__(self, *, num_classes: int) -> None:
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 2
        ):
            raise ValueError("num_classes must be an integer greater than one")
        super().__init__(
            dist_sync_on_step=False,
            sync_on_compute=True,
        )
        self.num_classes = num_classes
        self.add_state(
            "correct_by_class",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_by_class",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N, C]")
        if targets.ndim != 1:
            raise ValueError("targets must have shape [N]")
        if logits.shape[0] != targets.shape[0] or targets.numel() == 0:
            raise ValueError("logits and targets require the same non-zero N")
        if logits.shape[1] != self.num_classes:
            raise ValueError("logits class dimension must equal num_classes")
        if targets.dtype != torch.long:
            raise TypeError("targets must use torch.long class ids")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("logits must be finite")
        if (
            int(targets.min().item()) < 0
            or int(targets.max().item()) >= self.num_classes
        ):
            raise ValueError("targets contain a class id outside num_classes")

        predictions = logits.argmax(dim=1)
        self.total_by_class += torch.bincount(
            targets,
            minlength=self.num_classes,
        )
        matched_targets = targets[predictions == targets]
        self.correct_by_class += torch.bincount(
            matched_targets,
            minlength=self.num_classes,
        )

    def compute(self) -> Mapping[str, torch.Tensor]:
        missing = torch.nonzero(
            self.total_by_class == 0,
            as_tuple=False,
        ).flatten()
        if missing.numel() > 0:
            missing_ids = ", ".join(
                str(int(value))
                for value in missing.detach().cpu().tolist()
            )
            raise RuntimeError(
                "validation observed no samples for class ids: "
                + missing_ids
            )
        recall = (
            self.correct_by_class.float()
            / self.total_by_class.float()
        )
        return {
            **{
                f"class_{index}": recall[index]
                for index in range(self.num_classes)
            },
            "macro": recall.mean(),
        }
```

两个 state 都用 `dist_reduce_fx="sum"` 明确了合并语义，`compute()` 返回 flat mapping；
Stochaflow 会把 mapping key 接到 metric id 后面。教程主动拒绝 validation 中完全缺失的
类别，因为在这种情况下 macro recall 的协议并不完整。数据 recipe 应使用确定性的
stratified validation split，或为自己的任务定义另一种缺失类别政策。

上面的 reduction 声明使 metric contract 清楚，但不会自动选择分布式执行方式。固定 DDP
首版只在 rank 0 的完整 validation view 上运行 validation Metric，并拒绝 train/test phase
Metric；它不消费这里的 reduction 来推断 all-rank merge。FSDP 和分片 validation 仍未实现。

## 2. 由 Strategy 提供 channel

Strategy 解释 batch 并决定传给 metric 的参数。把下面代码放在同一 extension 的
`training.py`：

```python
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stochaflow.extensions import (
    MetricUpdate,
    REGISTRIES,
    TrainStepOutput,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    compute_objective,
)


@REGISTRIES.objectives.register("class-eval.cross-entropy")
class ClassCrossEntropyObjective(nn.CrossEntropyLoss):
    """Cross-entropy objective with the standard mean reduction."""

    def __init__(self) -> None:
        super().__init__(reduction="mean")


class ClassifierStrategy(TrainingStrategy):
    """Interpret ``(inputs, class_ids)`` batches for one classifier."""

    def __init__(self, model: nn.Module, objective: nn.Module) -> None:
        self.model = model
        self.objective = objective

    @property
    def metric_channels(self) -> frozenset[str]:
        """Declare the channel available to configured metrics."""

        return frozenset({"class-eval.prediction_target"})

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Return one scalar loss and one opaque metric update."""

        if (
            not isinstance(batch, (tuple, list))
            or len(batch) != 2
        ):
            raise TypeError(
                "classifier batches must be (inputs, class_ids)"
            )
        inputs, targets = batch
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("classifier inputs must be a Tensor")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("classifier class_ids must be a Tensor")
        logits = self.model(inputs)
        loss, _ = compute_objective(
            self.objective,
            logits,
            targets,
        )
        return TrainStepOutput(
            loss=loss,
            metric_updates={
                "class-eval.prediction_target": MetricUpdate(
                    args=(logits, targets),
                )
            },
            loss_aggregation_weight=targets.numel(),
        )


@REGISTRIES.training_builders.register(
    "class-eval.classification"
)
class ClassificationTrainingBuilder(TrainingBuilder):
    """Compose the classifier Strategy from core-owned assets."""

    def build(self) -> TrainingPlan:
        objective = self.context.objective
        if objective is None:
            raise TypeError(
                "class-eval.classification requires an Objective"
            )
        if self.context.process is not None:
            raise TypeError(
                "class-eval.classification does not use a Process"
            )
        model = self.context.primary_model
        return TrainingPlan(
            strategy=ClassifierStrategy(model, objective),
            primary_model=model,
            process=None,
            objective=objective,
        )
```

`metric_updates` 中的 Tensor 会在进入 metric state 前递归 detach；Strategy 不应自行持有
Metric 实例。`loss_aggregation_weight=targets.numel()` 只让 variable-size batch 的
epoch loss 按样本数报告，不会缩放 backward loss，也不会替 metric 自动添加权重。
`PerClassRecallMetric` 通过自己的计数 state 定义聚合。

如果项目使用 dataclass 或其他自定义 batch 容器并在其中保存 Tensor，还需实现
`DeviceTransferableBatch.to_device(device)`；核心不会反射自定义对象字段。

## 3. 激活插件并配置 validation

聚合注册模块导入 metric 与 training 模块：

```python
# src/class_eval/stochaflow_ext/__init__.py
from . import metrics, training  # noqa: F401
```

distribution 声明稳定 entry point：

```toml
[project.entry-points."stochaflow.extensions"]
class-eval = "class_eval.stochaflow_ext"
```

在完整训练配置中选择插件和 metric。下面只列与本教程相关的部分；`data` 和 `model`
继续由项目自己的 recipe 提供：

```yaml
extensions:
  plugins: [class-eval]

training:
  name: class-eval.classification
  params: {}

objective:
  name: class-eval.cross-entropy
  params: {}

metrics:
  - id: class_recall
    name: class-eval.per-class-recall
    channel: class-eval.prediction_target
    phases: [validation, test]
    params:
      num_classes: 4

trainer:
  early_stopping:
    enabled: true
    monitor: valid/metrics/class_recall/macro
    mode: max
    patience: 8
    min_delta: 0.001
```

模型选择只接受 `valid/loss` 或 `valid/metrics/<id>[/<subkey>]`，因此这个 monitor 会在
训练开始前核对 `class_recall` 已配置到 validation phase。若一个 validation epoch 没有
产生该 key，训练会 fail closed。`test/metrics/class_recall/macro` 会在最终 test
evaluation 中产生，但 test phase result 和 `diagnostics/...` 观测日志都不能控制 best
checkpoint。

运行：

```bash
stochaflow train --config experiments/classification/train.yaml
```

不要用只包含部分类别的 validation batch limit 解释 macro recall。smoke run 若必须限制
validation，应先证明被保留的确定性子集仍覆盖配置中的全部 class ids。

## 4. 聚焦测试 metric contract

在插件测试中直接验证 Registry、channel payload 和 mapping flatten：

```python
import pytest
import torch

from class_eval.stochaflow_ext import metrics as metrics_module
from stochaflow.metrics import (
    MetricEngine,
    MetricSpec,
    MetricUpdate,
)


def test_per_class_recall_contract() -> None:
    del metrics_module  # 导入已执行 Registry registration
    engine = MetricEngine(
        (
            MetricSpec(
                id="class_recall",
                name="class-eval.per-class-recall",
                channel="class-eval.prediction_target",
                params={"num_classes": 4},
            ),
        )
    )
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [5.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    engine.update(
        {
            "class-eval.prediction_target": MetricUpdate(
                args=(logits, targets),
            )
        }
    )
    values = engine.compute(reset=True)

    assert values["class_recall/class_0"] == pytest.approx(1.0)
    assert values["class_recall/class_1"] == pytest.approx(1.0)
    assert values["class_recall/class_2"] == pytest.approx(0.0)
    assert values["class_recall/class_3"] == pytest.approx(1.0)
    assert values["class_recall/macro"] == pytest.approx(0.75)
```

再为实际 TrainingBuilder 增加至少一条 vertical test：用包含全部 class ids 的微型
validation loader 跑一个 epoch，断言 logger、history 和 checkpoint 使用同一个
`valid/metrics/class_recall/...` key。插件 entry-point name、distribution、version 和
target 会写入 provenance；完整 metric declaration 与 channel 保存在 resolved config。
checkpoint 不保存 extension 源码或 Metric runtime state，只保存完成 epoch 的 scalar
mapping；它不保存 metric payload 或逐 key source/provenance metadata。跨环境恢复仍必须
安装同一插件并由项目 lockfile 固定依赖。

相关契约与迁移边界见[扩展公共 API](../api/extensions.md#metrics)和
[Checkpoint、配置权威与可移植性](../configuration/compatibility-and-migration.md)。
