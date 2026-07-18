# 扩展与 Registry

顶层扩展通过 `extensions.modules` 显式导入。配置不会扫描目录、Python entry point
或工作区；推荐把用户 project 以 editable package 安装，使模块具有稳定 import path。

## RegistryCatalog

| Registry | 扩展契约 | YAML 选择位置 |
| --- | --- | --- |
| `data_builders` | `DataBuilder` 类 | `data.name` |
| `sampling_artifact_writers` | `SamplingArtifactWriter` 类 | `sampling.writers[].name` |
| `models` | `torch.nn.Module` 类 | `model.name` |
| `noise_schedules` | `NoiseSchedule` 类 | `diffusion.noise_schedule.name` |
| `diffusions` | `torch.nn.Module` 类 | `diffusion.name`、`sampling.sampler.name` |
| `objectives` | `torch.nn.Module` 类 | `objective.name` |
| `optimizers` | `torch.optim.Optimizer` 类 | `optimizer.name` |
| `lr_schedulers` | scheduler 类或 builder | `lr_scheduler.name` |
| `loggers` | `ExperimentLogger` 类 | `logging.backends[].name` |
| `diagnostics` | `TrainingDiagnostic` 类 | `diagnostics[].name` |

Dataset、source、partition、degradation、Sampler、collate 和 DataLoader 不拥有全局
Registry。一个自定义 `DataBuilder` 完整拥有数据组合及其兼容性。

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
- diffusion：`model`、`noise_schedule`；
- optimizer / LR scheduler：模型参数或 optimizer；
- logger：`output_dir`、`run_name`；
- diagnostic：`logger`、`output_dir`、可选 `sample_shape`。

`diffusion_quality` 的 provider 仍使用其局部 `diagnostics[].params.modules` 机制。
所有内置名称与构造参数见[生成式组件索引](reference.md#registry-组件索引)。
