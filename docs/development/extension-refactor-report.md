# Extension 重构最终报告（草案）

> 状态：Stage 1–7 已形成检查点；Stage 8 的可复现性与文档收口已进入实现，
> 但整分支最终验收尚未执行。本文中的最终验证项均明确标记为 **待执行**，
> 不应据此声称 feature branch 已可合并。
>
> 对比基线：`codex/refactor-diffusion-schedules`
>（merge base `231bfc76340b3a4b81e4b38dd60d228d5b78fb00`）。

## 结论摘要

这次重构把 Stochaflow 从“内置 diffusion 训练程序加若干可配置组件”改为一个具有稳定
组合边界的训练与采样框架：

```text
installed extension distribution
  └─ entry point 激活注册组件
       ├─ DataBuilder -> ready-to-use train/validation/test iterables
       ├─ TrainingBuilder -> TrainingPlan -> TrainingStrategy
       └─ optional Process + family Dynamics + optional Sampler
            └─ SamplingBuilder -> SamplingOutput -> Writer

core
  └─ config / Registry / lifecycle / optimizer / checkpoint / manifest
```

核心只统一真正共享的生命周期，不再尝试统一所有 Dataset、split、condition、概率路径或
数值求解数学。算法 family 可以定义自己的窄 Process/Dynamics/Sampler 契约；任务 Builder
负责组合模型、condition、guidance、initial state 和 artifacts。

两个独立的可安装参考项目已经证明：

- 离散 Gaussian Physics reconstruction 可以复用内置 Process、DDPM/DDIM family primitive，
  同时在项目内增加 exact physics-guided DDIM；
- frozen-teacher distillation 可以通过 TrainingBuilder、TrainingPlan 和无状态 Strategy
  管理 teacher、额外 Objective、checkpoint resume 与 student-only sampling；
- 两者都通过标准 Python entry point 接入，不需要在 runner 中增加任务名称分支。

这不是科学模型效果报告。Stage 7 的真实数据证据是一条 **production-path smoke**：
一次 optimizer update，以及 baseline/guided 各一个样本的完整配置 solver schedule。
它没有执行收敛训练、1272 样本生产任务或论文数值复现。

## 目标、范围与非目标

### 目标

- 用户安装 Stochaflow 后，可以用普通 Python package 编写并发布 extension；
- YAML 只选择框架级组件，组件私有 `params` 由具体 Builder/实现解释；
- 自定义数据、模型签名、condition、训练任务、算法 family、solver 和 artifact writer
  不需要修改核心 dispatch；
- 训练、严格恢复、checkpoint-only sampling 和显式 sampling override 具有明确且唯一的
  配置权威；
- checkpoint 在导入 extension 代码之前即可读取 config/provenance，并为跨环境问题提供
  可操作诊断；
- 通过真实参考项目验证 OCP 边界，而不只依赖内置实现的自我测试。

### 本次范围

- 薄 DataBuilder 与三个内置图像 recipe；
- 可选 Process、family Dynamics、统一完整 `Sampler.sample()` 和 sampling Builder/Writer；
- TrainingBuilder、TrainingPlan、TrainingStrategy、Objective 与托管辅助训练资产；
- PyTorch optimizer/scheduler native-provider；
- entry-point plugin discovery、`stochaflow init`、严格 checkpoint v8；
- sampling artifact 容量证据；
- Physics reconstruction 与 knowledge distillation 纵向参考项目；
- 顶层组件 identity 摘要、缺失插件诊断和发布前文档。

### 明确非目标

- 不建立通用 Dataset/Sampler/DataLoader/Split/Batching Registry；
- 不要求 Gaussian、flow matching、score SDE 和 sigma-space solver 共享一套数学接口；
- 不在 `GenerativeDynamics` 根上增加 universal `predict`、`step`、`drift`、`score`
  或 `denoise`；
- 不建立 Process/Sampler 名称兼容矩阵；
- 不让 Strategy 接管 optimizer、device、checkpoint 或完整 epoch loop；
- 不实现多 optimizer、交替更新和 manual backward；
- 不把 extension 源码、wheel 或完整 Python 环境冻结进 checkpoint；
- 不实现 plugin marketplace、自动安装、目录扫描或 Stochaflow project manifest；
- 不承诺全量 dense trajectory 的内存安全，也不把当前 Writer 描述为 streaming；
- 不迁移未发布的旧 YAML、旧公共 API 或旧 checkpoint。

## Before / After

| 关注点 | 基线或早期草案 | 重构后 |
| --- | --- | --- |
| 扩展激活 | module-path bootstrap，缺少 distribution identity | `stochaflow.extensions` entry point + 显式 `extensions.plugins` + provenance |
| 数据 | 核心统一 DatasetFactory、split、sample key、mixture、bucket 和 loader 配置 | 一个薄 `DataBuilder.build() -> DataLoaders`；复杂拓扑留在普通 PyTorch 代码 |
| 训练 | Trainer 接收 criterion/train-step function，Gaussian 路径拥有特殊桥接 | Builder 组装 Plan；Strategy 只解释 batch 并计算 loss/metrics；Objective 是唯一损失抽象 |
| 生成算法 | `GaussianDiffusion` 持有模型并混合概率路径、prediction 与训练语义 | 可选 Process 保存 model-free path；family Dynamics 适配模型；Sampler 拥有 solver loop |
| 采样接口 | DDPM/DDIM 各自暴露多套 sample/trajectory 方法 | 实际 solver 统一完整 `Sampler.sample()`；trajectory 由 Observer 消费 |
| family primitive | 完整 sampler 内联 transition/schedule 数学，第二 solver 容易复制公式 | 离散 Gaussian 公开 DDPM/DDIM 窄 transition/schedule primitive，完整 sampler 委托它们 |
| condition/guidance | 容易进入通用 sampler 参数或核心分支 | model callable、Dynamics wrapper 和 SamplingBuilder 私有组合 |
| optimizer/scheduler | 框架逐个复制 PyTorch alias、构造参数和默认值 | allowlisted native target，core 注入依赖，用户 kwargs 原样透传 |
| 多模型训练 | 没有稳定的 teacher/auxiliary asset lifecycle | TrainingPlan 具名 managed assets；core 统一管理 mode、device、optimizer 和 checkpoint |
| 项目接入 | 没有标准脚手架或 distribution identity | `stochaflow init` 生成普通可安装 `src` layout 项目；不绑定 uv/pip |
| 恢复 | config/state 边界不完整，扩展代码可能在 provenance 检查前导入 | checkpoint v8、weights-only-safe payload、预检插件身份、strict full resume |
| 容量 | 大 sampling artifact 的驻留与 Writer 峰值未量化 | 公式、可重复 benchmark、参考主机证据和明确的 dense trajectory 非支持边界 |

## Stage 结果

| Stage | 结果 | 检查点 |
| --- | --- | --- |
| 1 | 建立全局扩展选择和稳定 `stochaflow.extensions` 入口；Stage 5 再把未发布的 module bootstrap 收口为 entry point | `2b6f893` |
| 2 | 撤销过重的通用 DataPipeline，落实薄 DataBuilder；提供 `image`、`super_resolution` 和 `multi_resolution_image` recipe | `e11bdaa`、`e6596dc` |
| 3 | 拆分 Process、family Dynamics、Sampler、SamplingBuilder；Process 可选；DDPM/DDIM 使用完整 `sample()` 生命周期；checkpoint 升至 v5 | `1bddd63` |
| 3.1 | 恢复离散 Gaussian family 的公开 DDPM adjacent transition、DDIM selected-pair transition 和 schedule resolver，修复第二 solver 必须复制数学的 OCP 回归 | `63de3db` |
| 4 | 引入 TrainingBuilder、TrainingPlan、单一职责 TrainingStrategy、Objective 和 managed auxiliary assets；checkpoint 升至 v6 | `40ebf16` |
| 4.1 | 标准 PyTorch optimizer/scheduler 改用 allowlisted native-provider 与 kwargs 直传；checkpoint 升至 v7 | `d6688db` |
| 5 | 用 packaging entry point 取代 module bootstrap；加入插件 provenance、版本策略、`stochaflow init`、strict resume 与 weights-only-safe checkpoint v8 | `dfde7a6` |
| 6 | 量化整体物化式 SamplingOutput/Writer 容量，保留现有 API，并明确 full dense trajectory 不受支持 | `abb4a29` |
| 7 | 交付两个独立 reference distribution；完成 tiny E2E、wheel/entry-point、真实 Physics production-path smoke 和复审修复 | `d9f0855`、`879ed0d` |
| 8 | 增加顶层 `selected_components` 审计摘要、包管理器中立的缺失插件诊断、迁移/教程/最终报告；checkpoint 仍为 v8 | **实现中；检查点待创建** |

Stage 2 的第一次实现曾试图把扩展点上移到一个仍然统一 Dataset、split、metadata 和
mixture 的 `DataPipeline`。该方向经架构复审后被明确替换，不保留旧公共类型兼容层。
Stage 3.1 同样是由真实的 guided solver 第二消费者发现并修复的发布前边界缺口；这两次
修正是 OCP 验收的一部分，而不是在错误抽象上继续增加适配层。

## 最终职责边界

| 角色 | 拥有 | 不拥有 |
| --- | --- | --- |
| DataBuilder | Dataset/source、partition、transform、PyTorch sampler、collate、DataLoader | 模型兼容性证明、核心通用 split schema |
| TrainingBuilder | 训练资产组装、teacher/auxiliary 构建和 TrainingPlan | epoch loop、backward、optimizer step |
| TrainingStrategy | structured batch 解释、模型/Objective 调用、单步 loss/metrics | device/mode、参数选择、checkpoint、factory |
| Process | 可选的 model-free probability path 和持久数学状态 | 模型、condition、prediction policy、sampling loop |
| family Dynamics | 该 family 的生成方向、模型适配和 prediction semantics | Registry/YAML 身份、通用跨 family API |
| Sampler | 完整数值算法、solver state、随机增量和 accepted-step lifecycle | 业务模型、任务 condition、artifact I/O |
| SamplingBuilder | 模型/condition/guidance/initial state/family 兼容性和运行组合 | 核心 checkpoint 格式、Writer 序列化策略 |
| Observer | 消费 sampling observation，按需保留 trajectory | solver 何时接受或拒绝 step |
| Writer | 序列化已经形成的 SamplingOutput | 模型数学、PDE metric 计算、增量 backpressure |
| Core | Registry、配置、资产 lifecycle、自动优化、checkpoint、manifest | 按注册名分支的任务/算法数学 |

## Open–Closed 验收矩阵

下表中的“核心 dispatch 修改”指 RegistryCatalog、runner、Trainer 的名称分支或通用 schema
变更，不包括为真实第二消费者发布 family 内窄 primitive 的 Stage 3.1 修正。

| 变化 | 扩展实现 | 复用的核心 | 核心 dispatch 修改 | 证据 |
| --- | --- | --- | --- | --- |
| 新数据组织/split | DataBuilder | Trainer、Strategy | 否 | Physics trajectory partition；distillation labeled data |
| 新 structured batch/模型签名 | Model + Builder/Strategy | Data transfer、Trainer | 否 | Physics three-frame field；distillation student/teacher |
| 新 Gaussian condition | Gaussian Dynamics/model callable | Process、DDPM/DDIM | 否 | baseline DFSR partial-noise reconstruction |
| 改变 accepted transition 的 guidance | 项目窄 Dynamics + custom Sampler | DDIM schedule/transition primitive、Observer、Writer | 否 | physics-guided DDIM |
| 新 probability path/算法 family | family Process/Dynamics/Sampler/Builder | Registry、config、checkpoint、runtime | 否 | 测试私有 Flow/VectorField family |
| 无 Process、无 solver 的直接变换 | SamplingBuilder | model provider、checkpoint、Writer | 否 | scaffold direct-transform 示例与 runtime 测试 |
| 新训练任务/多目标 | TrainingBuilder + Strategy + Objective | 自动 Trainer | 否 | frozen-teacher distillation |
| teacher/额外训练资产 | Builder 返回具名 managed assets | mode/device/optimizer/checkpoint | 否 | teacher resume、extra Objective state |
| 新 artifact 格式 | SamplingArtifactWriter | sampling runtime/manifest | 否 | Physics field/metrics writer |
| 新安装式插件 | entry-point distribution + decorators | discovery、Registry、CLI | 否 | 两个独立 wheel/reference distributions |

新的变化若必须在 runner 中按组件名增加 `if/elif`、给通用数据 schema 增加任务字段，或让
Process/Sampler 接收某个任务的 condition 参数，应视为扩展边界失败并回到设计阶段。

## 关键设计取舍与 rejected alternatives

### 选择薄 Builder，而不是配置化 Python 对象图

Dataset、split、sampler、collate 和模型 batch contract 之间存在任务私有耦合。分别建立
Registry 会把这些耦合搬进 YAML，并制造核心无法可靠验证的组合矩阵。因此核心只注册最终
DataBuilder；内置 holdout/k-fold/bucket 只是具体 recipe 的私有能力。

### 选择 family 内窄契约，而不是 universal Dynamics

Gaussian denoising prediction、vector field、reverse SDE 和 sigma-space denoiser 不需要共享
一个求值方法。`GenerativeDynamics` 只保留无行为语义根。统一的是 Registry、checkpoint
和实际数值 solver 的完整 `Sampler.sample()` 生命周期，不是跨 family 数学。

### 保留完整 `sample()`，同时恢复 family transition primitive

universal `step()` 无法自然表达 multistep、adaptive ODE、rejection 和内部子步；但一个
family 内的第二 solver 确实需要复用 model-free transition 数学。最终边界是：

- `Sampler.sample()` 负责完整循环和 observation lifecycle；
- Gaussian family 公开窄 transition/schedule primitive；
- transition 不调用模型、不发送 observation、不添加任务 correction；
- 内置 DDPM/DDIM 自己也委托这些 primitive，避免两份数学。

### exact physics guidance 使用项目 Sampler，而不是污染 Dynamics 或内置 DDIM

baseline condition 可以封装在 Gaussian model Dynamics 中；DFSR exact correction 则在
每个 accepted DDIM transition 后执行 `x_next = x_next - dx`，改变的是 solver transition。
把它伪装成 prediction 会改变算法含义；给内置 DDIM 增加 physics callback 又违反 OCP。
因此项目 Sampler 复用公开 DDIM transition，再在正确位置应用 correction。

### 选择 Objective 作为唯一损失抽象

`losses` 与 `objective` 不应重复表达同一责任。Strategy 组合模型调用和多个 Objective，
返回一个 scalar loss；Trainer 不理解 MSE、distillation 或 Gaussian target。

### 选择 TrainingBuilder/Plan，而不是万能 Strategy 或训练计算图 YAML

Builder 组装资产，Plan 声明 ownership，Strategy 只定义训练计算。Frozen teacher 是具名
eval auxiliary asset，不是通用 Strategy mode。独立 optimizer、交替更新或 manual backward
会改变 loop family，不能继续增加 Strategy 可选字段。

### 选择 PyTorch native-provider，而不是复制成熟依赖

Stochaflow 只记录 `torch.optim.*`/`torch.optim.lr_scheduler.*` target、显式 kwargs 和
scheduler interval。parameters/optimizer 由 core 注入。框架不复制上游签名、默认值或每个
class 的 Registry alias，也不通过参数名猜测 `total_steps`。

### 选择标准 packaging，而不是 Stochaflow project runtime

entry point 负责可发现性，YAML 负责本次显式激活。`stochaflow init` 只生成普通可安装
项目，不创建环境、不运行包管理器、不增加 `--project`、source-root 扫描或
ProjectManifest。pip、uv、conda、Poetry、PDM 均可；extension 和 CLI 必须位于同一
Python environment。

### 保留整体物化式 Writer，拒绝无证据预建 streaming

Stage 6 证明当前 final-only Physics profile 在参考主机上有足够余量，因此没有为了理论上的
大输出提前增加 event bus 或 `begin/write_batch/finish`。代价是全量 dense trajectory 明确
不支持；一旦真实 final artifact 在单 batch 可容纳时仍超过主机预算，才由证据触发最小
有背压的 streaming lifecycle。

### 选择 metadata + lockfile 责任分离，而不是 checkpoint 冻结源码

checkpoint 保存 resolved config、state、插件 name/distribution/version/target 和顶层组件
identity 摘要；依赖环境由 extension package 和用户 lockfile 恢复。复制 Python class/source
会扩大安全和迁移边界，也无法取代包管理器。

## Breaking changes 与迁移

这是一套尚未发布的新接口，因此采用直接替换，不提供 legacy 迁移层。

| 旧配置/API | 新配置/API | 迁移动作 |
| --- | --- | --- |
| `REGISTRIES.data_pipelines`、`dataset_factories` | `REGISTRIES.data_builders` | 注册一个最终组装 loader 的 DataBuilder |
| `DataPipeline`、`DataBundle`、`DatasetView`、通用 split/mixture | `DataBuilder`、`DataLoaders` | 把 Dataset/split/sampler/collate 组合移入具体 Builder |
| `diffusion: ...`、`REGISTRIES.diffusions` | 可选 `process` + SamplingBuilder 私有 sampler config | 注册 family Process/Sampler/Builder；不要在核心按名称匹配 |
| `stochaflow.diffusion` | `stochaflow.processes`、`stochaflow.sampling`、`stochaflow.extensions` | 从稳定扩展入口导入公开契约 |
| `GaussianDiffusion` 持有 model | `DiscreteGaussianDenoisingProcess` + `GaussianModelDynamics` | Builder/diagnostic 组合 Process 与 model callable |
| `sample_from_noise()`、`sample_trajectory()` 等多套接口 | `Sampler.sample(..., observer=...)` | Builder 创建 initial state；Observer 收集 trajectory |
| 顶层 `sampling.sampler` | `sampling.builder.params.sampler` | solver 参数归具体 SamplingBuilder |
| criterion/loss helper 和 algorithm-specific train function | Objective + TrainingBuilder/Strategy | Strategy 解释 batch 并调用已注入 Objective |
| 框架内 Adam/AdamW/scheduler aliases | `torch.optim.*`、`torch.optim.lr_scheduler.*` | 将显式 constructor kwargs 放入 `params` |
| v7 及更早 checkpoint | checkpoint v8 | 用当前代码重新训练/生成；旧格式直接拒绝 |

迁移时不要把旧的通用对象图逐字段翻译到新 YAML。先确定一次运行最终需要的
DataBuilder、TrainingBuilder 和 SamplingBuilder，再让它们在 Python 中组装私有拓扑。

## Checkpoint、配置与插件可移植性

### 配置权威

| 命令 | 权威配置 | checkpoint 作用 | 允许覆盖 |
| --- | --- | --- | --- |
| 新训练 | 显式完整 config | 无 | 文档化 train CLI runtime flags |
| strict resume | checkpoint 保存的 config | 完整训练 state | 安全 runtime flags；不接受外部完整 config |
| checkpoint-only sampling | checkpoint 保存的 config | 推理 state | sample CLI flags |
| 显式完整 config sampling | 外部 config | 仅提供可加载的推理 state | sample CLI flags |
| sampling-only overlay | checkpoint config 为 base | 推理 state | `sampling`、显式 `extensions.plugins`、sample CLI flags |

sampling 不恢复 optimizer/scheduler/epoch，因此用户可以显式改变样本数、shape、Builder、
Sampler、solver 参数、trajectory、writer 和 raw/EMA 选择。核心不把 external config 与
checkpoint config 做完整 equality 比较；真实兼容性由 model/Process state 和 Builder
capability contract 判断。strict training resume 则必须恢复同一个完整训练状态，不能套用
sampling override 规则。

### v8 保存什么

- resolved `StochaflowConfig`；
- primary model、可选 Process/Objective、EMA；
- optimizer/scheduler、训练进度和具名 auxiliary training assets；
- Python、NumPy、Torch CPU 和可用 CUDA RNG snapshot；
- extension provenance：entry-point name、distribution、version、target；
- lineage、runtime audit，以及 Stage 8 的顶层 `selected_components` 摘要。

`selected_components` 只读取 typed 顶层 role 的 `.name`，可选 role 显式为 `null`，列表
保持声明顺序。它不递归解释 sampler、noise schedule、teacher、source、condition 或其他
Builder 私有 `params`，也不参与 dispatch 或 compatibility 判断。完整 config 仍是重建权威。

### v8 不保存什么

- extension 源码、Python class、wheel 或 lockfile；
- Dataset、DataLoader iterator/worker、PyTorch sampler 和用户私有 generator runtime；
- Strategy state（Strategy 契约无持久状态）；
- SamplingOutput 或未声明的运行时对象；
- 对依赖环境或科学行为兼容性的保证。

payload 限定为 `torch.load(..., weights_only=True)` 默认支持的 Tensor、primitive 和普通
container。它降低读取 metadata 时动态执行扩展代码的风险，但不是恶意 Tensor、DoS 或
内存耗尽的安全沙箱。

### 跨环境规则

- extension distribution 必须安装到错误信息显示的 `sys.executable` 所在环境；
- fresh config 缺少插件时只有 entry-point name 可知，distribution/target 会明确显示为
  unavailable；checkpoint-backed 路径会显示保存的 expected distribution/version/target；
- name/distribution/target 身份变化始终失败；只有 version 差异可经交互确认或专用
  `--force-extension-version-mismatch` 接受；
- version 未变化的 editable source 修改无法由 distribution metadata 检测。若 state 或行为
  不兼容，由严格 state contract 或扩展运行错误暴露；
- 相对 data/output path 按当前进程 cwd 解析。生成模板要求从 repo root 运行；核心不会
  推断项目根或重写任意 Builder 私有路径；
- 依赖版本和上游默认值由用户选择的 lockfile/environment 负责。checkpoint 不是环境快照。

strict resume 每次创建新的 sibling run，不原地覆写父 run。匹配的 inherited `best.pt`
会先验证并原子物化到新 run；如果恢复历史 epoch 时 sibling mutable best 已被未来 epoch
覆盖，框架会拒绝猜测“当时的 best”。

## Stage 6：Sampling 容量证据

当前 `SamplingBuilder.run()` 在 Writer 开始前返回整体物化的 `SamplingOutput`。对
1272 个 `float32 [3,256,256]` final state：

- 一份 raw output 为 `1,000,341,504 B`，约 `954 MiB`；
- Tensor writer 的结构性峰值下界约 `1.8633 GiB`；
- 31 个 retained states 的全量 trajectory raw output 约 `29.81 GiB`；
- 同一路径的 Tensor writer 结构性峰值下界约 `87.57 GiB`。

提交的
[Stage 6 参考主机结果](../../benchmarks/results/stage6-macos-arm64.json)
来自 macOS arm64、16 GiB 主机上的 synthetic writer-ready CPU output。DFSR final-only
profile 经过一次 discarded run 和五次 fresh measured repeat，最大 peak RSS 为
`2.160 GiB`（主机内存的 `13.50%`）；8 样本 sparse trajectory preview 的 Tensor 和
high-entropy image/GIF 最大 peak RSS 分别约 `0.365 GiB` 和 `0.531 GiB`。

这些结果支持“当前 final-only 目标无需先重构 streaming API”，但不能推出：

- 其他 OS、allocator、PyTorch 版本或硬件具有相同峰值；
- 真实模型 activation、PDE residual autograd 或数据 I/O 已包含在 Stage 6 CPU 数字中；
- 1272 样本 dense reverse trajectory 可运行。

因此主 Physics profile 关闭 trajectory；trajectory 仅作为独立 preview，要求
`num_samples <= 8`、`every_steps >= 10`、accepted steps 不超过 40。详细公式、profile 和
复现命令见 [Sampling artifact 容量边界](sampling-capacity.md)。

## Stage 7：真实 Physics production-path smoke

受版本控制的机器可读证据位于
[stage7-physics-macos-arm64.json](../../benchmarks/results/stage7-physics-macos-arm64.json)。

### 输入与环境

- macOS 26.5.2 arm64、Apple M5、16 GiB、MPS；
- Python 3.14.3、PyTorch 2.11.0；
- reference `.npy`：`float32 [40,320,256,256]`，3,355,443,328 bytes，
  SHA-256 `775ca8435d2f2f1887e39d7e302890c434fb5d839e3521d80db4fff19d14cf89`；
- sparse archive：6,711,214,698 bytes，`ZIP_STORED` member `u3232.npy` 为
  `float64 [40,320,256,256]`，archive SHA-256
  `d007da6d934c6882e7f08a43b83520eacd894a6e6471127e899a21e54a14835e`。

prepare tool 以 bounded stream 让标准库验证 ZIP overlap/CRC，再校验 local/central
ZIP64/NPY metadata，最后只 mmap 所需 trajectory。生成的 held-out observation 为
`float32 [4,320,256,256]`，SHA-256
`235be61c4fc975fbf84d3ea33717a6d9143237fec1a5364591bfb6773ef65980`。

### 执行了什么

1. 独立 capacity helper 在真实 `[1,3,256,256]` batch 上执行训练 backward，以及 baseline/
   guided 各两步 sampler；记录一次 lifetime RSS 和 MPS current/driver allocation。
2. 正式 `stochaflow train` CLI 通过 entry point、DataBuilder、TrainingBuilder/Strategy、
   Trainer 和 Adam，在 production train config 上限制为一个 batch，完成：
   - batch size 4；
   - epoch 1、`global_step=1`；
   - checkpoint v8；
   - 70 个 optimizer state，step 均为 1。
3. 该 checkpoint 的 raw weights 分别运行：
   - baseline DDIM：partial-noise time 240，30 个 configured accepted transitions；
   - guided DDIM：partial-noise time 320，40 个 configured accepted transitions；
   - 两者各输出一个 finite `float32 [1,3,256,256]` reconstruction。

baseline 和 guided reconstruction SHA-256 分别为
`cdf78921f9e977e41b04d4b0759e4ae10456af91c3a912b7270bce6ade386053` 与
`35d374b67d8c99551e17369f4c89004f9165ac7b822e9b2ff9403d8c916d81af`。
这些 hash 只用于证据完整性，不代表科学质量。

### 证据限制

- 一次 optimizer update 不是完整训练或收敛训练；
- 每个 solver 一个 reconstruction 不是 1272 样本生产任务；
- 随机初始化权重不能提供论文精度、重建质量或方法优劣证据；
- 两步 capacity helper 与后续 30/40-step CLI smoke 是两项独立证据；
- 当前 sampling manifest 可由固定 schedule 和 dynamics-evaluation diagnostics 审计
  transitions，但没有单独持久化 `SamplerResult.num_steps`；
- prepare 的 CRC/ZIP64 hardening rerun 晚于保留的 train/sample artifacts，但重建了完全
  相同的 observation bytes/hash；该时间关系已写入 evidence；
- evidence 的 `head_at_run` 为 `d9f0855` 且记录工作树为 dirty；报告通过逐文件 SHA-256
  锚定相关实现/config，最终修复收口提交为 `879ed0d`；
- GB 级源数据、checkpoint 和 ignored `outputs/` 未提交到仓库。

因此本节只能声称真实数据走通 production configuration 的核心 extension 生命周期和
完整单样本 solver schedule，不能称为“完整 production training”“full workload”或
“scientific reproduction”。

## 已知限制

- 自动训练 loop 只支持一个 optimizer、一个标量总 loss 和一次 backward；
- EMA 只跟踪 primary inference model；Process、Objective 和 auxiliary modules 保存 raw state；
- `SamplingOutput`/Writer 仍整体物化，不支持全量 dense trajectory 或用户自定义 streaming；
- 一个 Python 进程首次激活后固定 plugin selection，不支持 unload/reload 或 owner-aware
  Registry view；不同 selection 应使用独立进程；
- checkpoint 不冻结 extension 源码；同版本 editable change 无法由 provenance 检测；
- checkpoint v8 只支持当前格式，不迁移 v7 及更早草案；
- strict resume 不是 weights-only warm start；更换 optimizer/scheduler/config 的初始化流程
  尚未提供独立 `--init-from` 入口；
- arbitrary historical resume 依赖仍匹配的 mutable sibling `best.pt`；没有 immutable
  per-epoch best snapshot；
- DataLoader/worker/user sampler runtime state 不进入 checkpoint；可复现性依赖 seed、
  epoch 和扩展自己的 `set_epoch()`；
- learnable/mutable Gaussian schedule 不在固定 Process-owned coefficient snapshot 契约内；
- 自适应、rejection 或 early-stop solver 应在 Builder metadata 中显式记录真实 accepted
  step 数；通用 runtime 尚不强制该字段；
- 顶层 `selected_components` 是审计摘要，不递归覆盖 Builder 私有拓扑；
- Stage 7 未执行 1272 样本 job、收敛训练、论文指标或跨平台 capacity benchmark。

## 证据驱动的后续路线

以下项目不是本 feature 的隐含承诺，只在出现真实第二消费者或容量证据后进入设计：

1. final artifact 在单 batch 可容纳但总输出超过主机预算时，设计同步、有背压并支持
   abort/cleanup/atomic publish 的增量 Writer lifecycle；
2. 需要独立 optimizer、交替更新或 manual backward 时，定义新的 training-loop family，
   不向现有 Strategy 增加控制模式；
3. 需要从旧权重更换训练 config 时，增加独立 weights-only warm-start 入口；
4. 需要从任意历史 epoch 恢复当时 best 时，引入 immutable/versioned best retention；
5. 第二种真实动态 Gaussian Process 出现时，再拆 schedule/marginal/posterior capability；
6. adaptive/rejection solver 出现时，标准化其实际 accepted/rejected/model-evaluation
   diagnostics，而不改变 universal `Sampler.sample()`；
7. 用户确实需要更强代码 provenance 时，再评估 wheel hash、VCS revision 或 lockfile
   artifact；不能把它们误作源码兼容保证。

## 最终验证状态

Stage 1–7 各自执行过聚焦测试、Ruff/Pyright 和相应独立审查；Stage 6/7 另有上述提交的
机器证据。但这不等于当前 Stage 8 工作树已经通过整分支门禁。

下表是合并前必须执行的最终命令。状态栏在真实运行并保存结果前不得改成“通过”。

| 验证 | 状态 | 结果/修复记录 |
| --- | --- | --- |
| `uv run python tools/generate_config_reference.py` | **待执行** | — |
| `uv run python tools/generate_config_reference.py --check` | **待执行** | — |
| `uv run pytest` | **待执行** | — |
| `uv run ruff check .` | **待执行** | — |
| `uv run pyright` | **待执行** | — |
| `uv build` | **待执行** | — |
| `uv run sphinx-build -W --keep-going -b html docs docs/_build/html` | **待执行** | — |
| merge-base 全分支 diff 架构复审 | **待执行** | — |
| 独立最终代码审查及 finding 收口 | **待执行** | — |

最终验收完成后，还应在本表记录实际命令结果、必要修复和最终 commit，而不是仅修改顶部
状态文字。

## 相关资料

- [自定义代码扩展实施计划](../custom-code-extension-support-plan.md)
- [Extension 重构决策记录](extension-refactor-decisions.md)
- [扩展开发手册](../configuration/extensions.md)
- [纵向扩展参考项目](../configuration/reference-projects.md)
- [常用工作流](../configuration/workflows.md)
- [故障排查](../configuration/troubleshooting.md)
- [Sampling artifact 容量边界](sampling-capacity.md)
- [Stage 6 机器结果](../../benchmarks/results/stage6-macos-arm64.json)
- [Stage 7 真实数据 smoke 证据](../../benchmarks/results/stage7-physics-macos-arm64.json)
