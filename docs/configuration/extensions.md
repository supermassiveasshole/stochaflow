# 扩展与 Registry

所有可配置组件都由统一的 `REGISTRIES: RegistryCatalog` 管理。配置不会扫描包、
entry point 或文件系统；`data.modules` 是唯一扩展模块入口。

## RegistryCatalog 一览

| Registry | 扩展契约 | YAML 选择位置 |
| --- | --- | --- |
| `models` | `torch.nn.Module` 类 | `model.name` |
| `dataset_factories` | `DatasetFactory` 类 | `data.datasets[].factory` |
| `split_strategies` | `SplitStrategy` 类 | `data.splits.mode` |
| `noise_schedules` | `NoiseSchedule` 类 | `diffusion.noise_schedule.name` |
| `diffusions` | `torch.nn.Module` 扩散类 | `diffusion.name`、`sampling.sampler.name` |
| `objectives` | `torch.nn.Module` 目标类 | `objective.name` |
| `optimizers` | `torch.optim.Optimizer` 类 | `optimizer.name` |
| `lr_schedulers` | scheduler 类或 builder | `lr_scheduler.name` |
| `loggers` | `ExperimentLogger` 类 | `logging.backends[].name` |
| `diagnostics` | diagnostic 类 | `diagnostics[].name` |

所有内置名称、构造参数和运行时注入参数都在[生成式组件索引](reference.md#registry-组件索引)
中列出。

## 模块注册流程

把装饰器放在可导入模块顶层：

```python
# my_project/components.py
from stochaflow.utils.registry import REGISTRIES


@REGISTRIES.models.register("tiny_model")
class TinyModel(...):
    ...
```

再在 YAML 中显式导入并引用：

```yaml
data:
  modules:
    - my_project.components
  # datasets/image/batching 省略

model:
  name: tiny_model
  params: {}
```

`load_config()` 和 `load_config_dict()` 在 schema 构造后、跨字段校验前调用
`REGISTRIES.load_modules(data.modules)`。导入是幂等的，因此同一进程重复加载配置或
checkpoint 不会重复执行注册。模块必须可由当前 Python 环境导入；推荐把项目以
editable package 安装，而不是依赖工作目录修改 `sys.path`。

注册名必须非空且在对应 Registry 中唯一。重复名称、错误基类、未知名称都会抛出
`RegistryError`；未知名称错误会列出该 Registry 当前所有可用项。

## 自定义 DatasetFactory

下面展示一个最小 map-style Factory。实际 Dataset 应在 `__getitem__` 中输出已经
resize/crop/normalize 的 Tensor。

```python
from torch.utils.data import Dataset

from stochaflow.data import DatasetBuildRequest, DatasetFactory, DatasetView
from stochaflow.utils.registry import REGISTRIES


class ManifestDataset(Dataset):
    def __init__(self, records, *, role, context):
        self.records = records
        self.role = role
        self.context = context

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return load_and_transform(record, role=self.role, context=self.context)


@REGISTRIES.dataset_factories.register("manifest_images")
class ManifestImagesFactory(DatasetFactory):
    def build(self, request: DatasetBuildRequest) -> DatasetView:
        records = read_manifest(
            self.context.params["manifest"],
            split=request.native_split,
        )
        bucket_ids = tuple(
            self.context.buckets.select(record.width, record.height).name
            for record in records
        )
        dataset = ManifestDataset(
            records,
            role=request.role,
            context=self.context,
        )
        return DatasetView(
            source_id=self.context.source_id,
            dataset=dataset,
            sample_keys=tuple(record.id for record in records),
            bucket_ids=bucket_ids,
        )
```

使用配置：

```yaml
data:
  modules: [my_project.datasets]
  datasets:
    - id: portraits
      factory: manifest_images
      params:
        manifest: data/portraits.jsonl
      splits:
        train: train
        test: test
```

不要注册一个预构建 Dataset 实例，也不要写函数式 builder。Factory 的构造器固定
接收 `DatasetFactoryContext`，`build(request)` 是唯一必需扩展点。详细数据约束见
[数据管线](data-pipeline.md#datasetfactory-契约)。

## 自定义 SplitStrategy

公开扩展类型包括 `SplitContext`、`DataPartitions`、`DatasetMaterializer` 和
`SplitStrategy`。策略从 materializer 请求需要的 logical split/role，并返回一个
或多个分区：

```python
from stochaflow.data import (
    DataPartitions,
    DatasetSelection,
    SplitContext,
    SplitStrategy,
)
from stochaflow.utils.registry import REGISTRIES


@REGISTRIES.split_strategies.register("train_tail_validation")
class TrainTailValidation(SplitStrategy):
    def split(self, context: SplitContext) -> list[DataPartitions]:
        train_view = self._required(
            context.datasets.build("train", role="train"),
            logical_split="train",
        )
        eval_view = self._required(
            context.datasets.build("train", role="eval"),
            logical_split="train",
        )
        self._validate_aligned(train_view, eval_view)
        validation_size = context.config.validation_size
        if not isinstance(validation_size, int):
            raise ValueError("train_tail_validation requires an integer size")
        if not 0 < validation_size < len(train_view):
            raise ValueError("validation_size must leave two non-empty sets")
        split_at = len(train_view) - validation_size
        return [
            DataPartitions(
                train=DatasetSelection(train_view, range(split_at)),
                valid=DatasetSelection(
                    eval_view,
                    range(split_at, len(eval_view)),
                ),
                test=context.datasets.build("test", role="eval"),
            )
        ]
```

YAML 只写 Registry 名称：

```yaml
data:
  modules: [my_project.splits]
  splits:
    mode: train_tail_validation
    validation_size: 1000
```

schema 接受任意非空 Registry 名称；未知名称会在 `DataPipeline` 构建策略时报告可用
名称。内置名称仍执行各自的专属配置校验。

## 其他组件的构造约定

组件的 `params` 通常作为关键字参数传给注册类。以下参数由 runner 注入，不允许在
YAML 中覆盖：

- diffusion：`model`、`noise_schedule`；
- optimizer：模型 `parameters`；
- LR scheduler：`optimizer`；
- logger：`output_dir`、`run_name`；
- diagnostic：`logger`、`output_dir`、按需提供的 `sample_shape`；
- DatasetFactory：`DatasetFactoryContext`。

diagnostic 可通过类方法 `context_parameters(context)` 声明额外的运行时参数。
若配置 `params` 与运行时参数重名，构建会失败，避免用户值被静默覆盖。

PyTorch optimizer/scheduler 的透传参数随安装版本变化；本手册只承诺
[字段参考](reference.md)中列出的常用参数，最终可用签名以当前 PyTorch 官方 API
为准。项目自有组件的参数则由生成器与 Python 签名严格比对。

## 扩展模块检查清单

1. 类继承对应基类，装饰器使用正确 Registry。
2. 注册名稳定、非空，且不覆盖内置名称。
3. 模块 import 不执行下载、训练或其他重副作用。
4. 模块路径加入 `data.modules`，训练与 sample 使用同一环境。
5. Factory 的 sample key、bucket metadata 与 Tensor 契约有单元测试。
6. Strategy 对空 split、错位 view、越界参数和多 source 顺序有单元测试。
