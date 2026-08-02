# Sampling Request Config Refactor

- 文档性质：被取代的历史开发决策记录；不属于公开 API
- 状态：Superseded by Hydra C1（2026-08-01）
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)；当前 contract 由
  [Hydra 计划 C1](hydra-configuration-composition-migration-plan.md)拥有
- Historical scope：本文 checkpoint v10、partial request、Physics/KD 示例只描述
  当时实现，不描述当前 runtime；C1 不保留兼容性，retained-example cleanup 也不继续
  维护这些案例
- 日期：2026-07-28
- 历史兼容性：当时只支持 checkpoint v10，不迁移旧 sampling config 或 checkpoint
- 当前结果：checkpoint v12；训练 schema 不含 `sampling`/`use_for_sampling`；sample CLI
  同时要求 checkpoint 与完整 `sample:` config；无 auto final sample 或 partial merge

> 以下章节仅保留 v10 设计的决策沿革。当前配置、CLI 与 checkpoint 行为请以 C1 计划和
> `docs/configuration/` 公开文档为准。

## 1. 背景与问题

旧设计把 `sampling.builder` 暴露在训练配置和独立 sampling config 中。由此产生了两个
相互竞争的 composition root：

- `TrainingBuilder` 决定模型、Process、Objective 和 auxiliary modules 如何共同形成一项
  可训练任务；
- 外部 sampling config 又能任意选择 `SamplingBuilder`，并覆盖训练任务才知道的
  prediction、conditioning、guidance、input adaptation 等语义。

这使 checkpoint 只保存 state、却不能可靠描述如何使用这些 state。一个 YAML overlay
可以组合出与 checkpoint 拓扑或训练语义不兼容的推理任务，运行时只能在深层构造或
state loading 阶段失败。`sample` 同时承担“完整外部实验配置”和“局部覆盖”两种含义，
也导致 config authority、plugin provenance 和输出目录语义含混。

本轮将 checkpoint-backed inference 固定为一条单向责任链：

```text
TrainingBuilder
    -> TrainingPlan.inference_recipe
    -> checkpoint v10
    -> optional sample request
    -> internal SamplingBuilder
    -> SamplingOutput
    -> artifact writers
```

`sample` 是 generation、reconstruction 和 prediction 共用的 checkpoint-backed
inference 入口，不等同于“必须执行一个数值 Sampler”。

## 2. 最终决策

### 2.1 训练配置只保存用户可调默认值

最终公开 schema：

```python
SamplingConfig(
    run_after_training: bool,
    sampler: ComponentConfig | None,
    options: dict[str, object],
    shape: list[int] | None,
    num_samples: int,
    batch_size: int,
    seed: int | None,
    writers: list[ComponentConfig],
)
```

对应 YAML：

```yaml
sampling:
  run_after_training: true
  sampler:
    name: ddim
    params:
      num_steps: 50
      eta: 0.0
  options:
    weights: ema
    trajectory:
      enabled: false
  shape: [3, 32, 32]
  num_samples: 36
  batch_size: 12
  seed: 123
  writers:
    - name: image
      params: {}
```

训练配置不再选择 SamplingBuilder。`run_after_training` 只决定训练结束、恢复 selected
best checkpoint 后是否调用该 checkpoint 的 recipe；CLI
`train --skip-final-sample` 可以仅对当前 invocation 禁止这一步。

### 2.2 TrainingPlan 固化内部 inference recipe

有 checkpoint-backed inference 能力的 `TrainingBuilder` 返回：

```python
TrainingPlan(
    ...,
    inference_recipe=SamplingRecipe(
        name="standard_denoising",
        contract={"prediction_type": "epsilon"},
    ),
)
```

`SamplingRecipe` 只有：

- `name`：内部 `sampling_builders` Registry 名称；
- `contract`：该任务不可由 sample request 覆盖的 JSON-safe 固定参数。

Framework validation 对 contract 做递归 snapshot：mapping 变为只读 mapping、list
变为 tuple，嵌套值也不可再修改；序列化时再显式还原为 JSON list/mapping。外部声明和
checkpoint reader 只接受严格 JSON value，不把 Python tuple 静默规范化成 list。

checkpoint v10 始终保存 `inference_recipe` 字段，值为 `null` 或严格 envelope：

```json
{
  "schema_version": 1,
  "name": "standard_denoising",
  "contract": {"prediction_type": "epsilon"}
}
```

它是 checkpoint 的一部分，而不是另一份通用公开 sampling graph。没有 recipe 的
checkpoint 明确不支持 `sample`。recipe 能自包含推理的组合描述，但 checkpoint
不是自包含代码包：相应内置代码或 extension 仍须安装。

### 2.3 Sample request 是严格的 partial request

`sample --config` 的文件顶层只能包含：

```yaml
sampling:
  num_samples: 36
  batch_size: 12
  options:
    weights: raw
extensions:
  plugins:
    - optional-extra-plugin
```

request 中的 `sampling` 只接受：

- `sampler`
- `options`
- `shape`
- `num_samples`
- `batch_size`
- `seed`
- `writers`

`sampling.run_after_training` 与 `sampling.builder` 非法；训练、模型、Process、Objective、
optimizer、data 或 trainer 等完整 config 字段同样非法。

解析规则固定为：

1. 未提供的字段继承 checkpoint config；
2. `shape`、`num_samples`、`batch_size` 和 `seed` 按字段替换；
3. `options` 只按顶层 key 做一层 merge，嵌套 mapping 不递归合并；
4. 显式 `sampler` 原子替换 checkpoint 默认 sampler，包括显式 `null`；
5. 显式 `writers` 原子替换 checkpoint 默认 writer 列表；
6. `options.sampler` 始终非法；
7. request option 与 recipe fixed contract 同名时 fail closed；
8. recipe 已固定 `sampler` 时，request 不能再提供 sampler。

运行时先解析 checkpoint recipe，再合并 request。最终交给内部 SamplingBuilder 的参数
按以下顺序构造：

```text
checkpoint sampling.options
    <- shallow sample-request options
    <- optional top-level sampler declaration
    <- immutable recipe contract
```

任一 contract collision 都是配置错误，不允许通过覆盖顺序静默解决。

### 2.4 CLI、plugin 与输出契约

`stochaflow sample` 必须显式提供 `--checkpoint`。`--config` 可省略；省略时完全使用
checkpoint 中的 sampling defaults。旧的“只给完整 config、把 checkpoint 当成可选
state”路径已经删除。

sample request 中的 `extensions.plugins` 是 additive activation：

- checkpoint 记录并要求原训练插件身份；
- request 只能追加 writer、sampler 或其他推理期所需插件；
- 空列表不清除 checkpoint 要求的插件；
- 同名 entry point 的身份或版本冲突仍 fail closed。

在处理 additions 之前，runtime 还会检查 checkpoint config 的 plugin selection 与
checkpoint provenance 一致；config-only、未被 provenance 认证的插件不会因为一次
additive request 而被导入。

默认 sample 输出在 checkpoint 所属 run 的 `samples/<timestamp>/` 下创建唯一目录。
显式 `--output-dir` 是这次 invocation 的精确目标目录，不解释为训练 run root，也不
自动增加时间戳层。run manifest 使用 `recipe` 和
`selected_components.sampling_recipe`，不再把内部实现称为用户选择的 builder。
训练 manifest 和 checkpoint metadata 中的 recipe identity 来自已验证
`TrainingPlan`，而不是从 config 或 registry name 推断。

## 3. 公共与内部边界

| 能力 | 公开配置/请求 | checkpoint | framework internal |
| --- | --- | --- | --- |
| 是否训练后执行 | `run_after_training` | resolved default | runner gate |
| 数值求解器 | `sampler` | default | recipe 接收或固定 |
| 任务级可调参数 | `options` | defaults | recipe-specific interpretation |
| shape/count/batch/seed | 独立字段 | defaults | runtime batching/RNG |
| writer selection | `writers` | defaults | writer registry/lifecycle |
| 推理任务组合 | 不可选择 | `inference_recipe` | `SamplingBuilder` |
| prediction/condition 等不变量 | 不可覆盖 | recipe `contract` | Builder validation |
| 模型/Process state | 不可重声明 | checkpoint state/config | runtime reconstruction |

`SamplingBuilder` Registry 继续存在，但它是训练任务写入 checkpoint 的内部 recipe
implementation boundary。它不是 sample request 的公开选择面，也不要求所有任务都
使用数值 `Sampler`。

## 4. Breaking migration

| 旧声明或行为 | v10 替换 |
| --- | --- |
| `sampling.builder.name` | `TrainingPlan.inference_recipe.name` |
| `sampling.builder.params.prediction_type` 等不变量 | `SamplingRecipe.contract` |
| `sampling.builder.params.sampler` | `sampling.sampler` |
| 其他 request-time Builder 参数 | `sampling.options` |
| `sampling.builder: null` 禁止 final sample | `sampling.run_after_training: false` |
| sampling-only overlay | strict partial sample request |
| 完整外部 sampling experiment config | 删除 |
| 不带 checkpoint 的 `sample --config` | 删除；`--checkpoint` 必填 |
| request 替换插件 selection | additive plugin activation |
| checkpoint v8/v9 | 不读取；用当前代码新建 v10 run |

不提供 adapter、双格式解析或自动 migration。旧 checkpoint 不能 resume 或 sample。

## 5. Example 与 scaffold 结果

| 使用方 | Recipe 类型 | 可调 request | 固定 contract |
| --- | --- | --- | --- |
| built-in Gaussian image generation | `standard_denoising` | sampler、weights、trajectory、数量与 writer | prediction type |
| AFHQ-v2 | class-conditional denoising recipe | standalone request 只改 sampler 与 guidance | prediction type |
| Physics reconstruction | reconstruction recipe | sampler 与少量 correction options；real-smoke 另改 count/weights | prediction type |
| Knowledge Distillation | student-only direct prediction | weights、synthetic input distribution 与输出数量；无 standalone request | model input width |
| `stochaflow init` scaffold | direct regression predictions | input range、数量与 writer defaults | model input width |

这些 recipe 都由各自 TrainingBuilder 声明。example YAML 只配置用户可调默认值，独立
sample 文件只表达一次 invocation 的差异。

## 6. 明确拒绝的替代方案

- **保留公开 `sampling.builder`：** 会重新产生与 TrainingBuilder 竞争的 composition
  root。
- **将整个 sampling config 固化且不可覆盖：** 无法满足更换 solver、数量、writer 或
  task input 的正常推理需求。
- **允许 request 覆盖任意 Builder 参数：** 无法区分训练语义不变量与安全的运行时
  参数。
- **递归 merge options：** list、mapping 和 deletion 语义不清晰；一层 merge 更容易
  预测，嵌套对象由 request 原子替换。
- **由 registry name 推断 compatibility：** 将 task-specific 矩阵推回 core，违反
  Builder 边界和 open-closed 原则。
- **把 `sample` 限定为生成模型：** Physics reconstruction、监督 prediction 和 KD
  comparison 都是同一 checkpoint-backed inference lifecycle 的合法实例。

## 7. 实施触点

本轮实现同步修改：

- config schema、strict partial request parser 与 field-wise resolver；
- `SamplingRecipe` validation/serialization 与 `TrainingPlan.inference_recipe`；
- checkpoint v10 writer、loader、strict resume comparison；
- sample runtime、explicit checkpoint CLI、plugin activation 与 run manifest；
- built-in、AFHQ、Physics、KD 和项目 scaffold 的 TrainingBuilder/YAML/request；
- framework、extension API、workflow、configuration reference、migration、
  troubleshooting、教程和 example README。

公开文档只描述最终行为，不链接本开发记录。

## 8. 验收

验收条件：

- training config 精确接受最终 `SamplingConfig` 字段；
- sample request 拒绝完整 experiment config、`run_after_training` 和 `builder`；
- request inheritance、shallow options merge、atomic sampler/writer replacement 受测试；
- recipe contract collision 与无 recipe checkpoint fail closed；
- CLI 强制显式 checkpoint；
- v10 checkpoint round-trip、resume 与 sampling provenance 受测试；
- built-in、AFHQ、Physics、KD 和 scaffold 均通过各自 end-to-end smoke；
- generated configuration reference 与 source metadata 一致；
- repository search 只在 breaking migration 说明中保留旧 schema 名称。

最终验证命令：

```bash
uv run python tools/generate_config_reference.py --check
uv run pytest tests/test_config_reference.py tests/test_project_scaffold.py \
  tests/test_config.py tests/test_sampling_runtime.py tests/test_cli.py \
  tests/test_checkpoint.py
uv run ruff check .
uv run pyright
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
uv run pytest
```
