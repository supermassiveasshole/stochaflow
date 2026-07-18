# 扩展与 Registry

顶层扩展通过 `extensions.modules` 显式导入。配置不会扫描目录、Python entry point
或工作区；推荐把用户 project 以 editable package 安装，使模块具有稳定 import path。

## RegistryCatalog

| Registry | 扩展契约 | YAML 选择位置 |
| --- | --- | --- |
| `data_pipelines` | `DataPipeline` 类 | `data.name` |
| `dataset_factories` | `DatasetFactory` 类 | 内置管线的 dataset factory 字段 |
| `sampling_artifact_writers` | `SamplingArtifactWriter` 类 | `sampling.writers[].name` |
| `models` | `torch.nn.Module` 类 | `model.name` |
| `noise_schedules` | `NoiseSchedule` 类 | `diffusion.noise_schedule.name` |
| `diffusions` | `torch.nn.Module` 类 | `diffusion.name`、`sampling.sampler.name` |
| `objectives` | `torch.nn.Module` 类 | `objective.name` |
| `optimizers` | `torch.optim.Optimizer` 类 | `optimizer.name` |
| `lr_schedulers` | scheduler 类或 builder | `lr_scheduler.name` |
| `loggers` | `ExperimentLogger` 类 | `logging.backends[].name` |
| `diagnostics` | `TrainingDiagnostic` 类 | `diagnostics[].name` |

复杂 split、混合、batch sampler 和 collate 不再拥有独立全局 Registry；自定义
`DataPipeline` 完整拥有这些语义。

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

`load_config()` 和 `load_config_dict()` 在 dataclass 构建后导入这些模块，再执行跨字段
校验。`RegistryCatalog.load_modules()` 保证同一进程内幂等。注册名必须非空且唯一；
错误基类、重复名称和未知名称会抛出 `RegistryError`。

第三方代码应从 `stochaflow.extensions` 导入稳定契约。该入口包括 DataPipeline、
DatasetFactory、sampling writer、Registry、配置组件、NoiseSchedule、logger 和
diagnostic 生命周期类型，不导出内置 split 或图像 bucket 的实现细节。

## 自定义 DataPipeline

```python
from stochaflow.extensions import (
    DataBundle,
    DataPipeline,
    REGISTRIES,
    SplitData,
)


@REGISTRIES.data_pipelines.register("physics")
class PhysicsPipeline(DataPipeline):
    def build(self) -> list[DataBundle]:
        train, valid = build_physics_loaders(
            self.context.params,
            seed=self.context.seed,
        )
        return [
            DataBundle(
                train=SplitData("train", train, num_batches=1000),
                valid=SplitData("valid", valid, num_batches=100),
            )
        ]
```

```yaml
data:
  name: physics
  params:
    mesh: data/mesh.zarr
    steps_per_epoch: 1000
```

`DataPipelineContext.params` 是配置参数的深拷贝，`seed` 是实验种子。`build()` 必须
返回非空 `list[DataBundle]`；训练 split 必须具有有限 epoch 长度。核心不要求
Dataset、sample key 或图像 batch。

## 自定义 DatasetFactory

`DatasetFactory` 适合复用内置 `map` 或 `multi_resolution_image` 的数据读取层：

```python
from stochaflow.extensions import DatasetFactory, DatasetView, REGISTRIES


@REGISTRIES.dataset_factories.register("field_archive")
class FieldArchiveFactory(DatasetFactory):
    def build(self, request):
        dataset = FieldArchive(
            self.context.params["path"],
            split=request.native_split,
        )
        return DatasetView(
            source_id=self.context.source_id,
            dataset=dataset,
            sample_keys=tuple(dataset.stable_ids),
        )
```

Factory context 只含 `source_id` 与 `params`。若图像管线需要逐样本尺寸，Factory 在
`DatasetView.batch_metadata` 中返回等长的 `ImageSampleMetadata` 或包含
`width`、`height` 的 mapping；图像 resize/crop/normalize 由管线处理。

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

writer 返回的 mapping 不能为空；artifact key 在全部 writer 间必须唯一，路径必须
已经存在。任何 writer 失败都会使采样失败。内置 `tensor` 支持任意 rank Tensor；
内置 `image` 才校验 NCHW、1/3 通道，并拥有 `grid_nrow`、`gif_fps` 与
`denormalize` 参数。

## 其他构造约定

组件 `params` 通常作为关键字参数传给注册类。以下参数由运行时注入：

- DataPipeline：`DataPipelineContext`；
- DatasetFactory：`DatasetFactoryContext`；
- diffusion：`model`、`noise_schedule`；
- optimizer / LR scheduler：模型参数或 optimizer；
- logger：`output_dir`、`run_name`；
- diagnostic：`logger`、`output_dir`、可选 `sample_shape`。

`diffusion_quality` 的 provider 仍使用其局部 `diagnostics[].params.modules` 机制。
所有内置名称与构造参数见[生成式组件索引](reference.md#registry-组件索引)。
