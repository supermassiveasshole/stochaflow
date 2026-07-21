# Extension 重构决策记录

本文记录 extension 重构中已经确定的边界、明确删除的方案和仍需评审的技术债。它不重复
[实施计划](../custom-code-extension-support-plan.md)，而是作为每个 Stage 提交检查点的架构
审阅入口。后续批注应优先修改本文的决策或待定项，再更新实施计划和代码。

## Stage 3 检查点

状态：实现、针对性测试和代码审查已完成，等待维护者审阅本记录。

### 最终职责边界

```text
Process   -> 可选的、模型无关的 probability path 与持久状态
Dynamics  -> family 内的生成方向、模型适配和 prediction semantics
Sampler   -> 完整数值求解循环及 accepted-step lifecycle
Builder   -> 任务组合、兼容性验证、初始化和结果规范化
Observer  -> 消费 SamplingObservation
Writer    -> 物化已经形成的 SamplingOutput
```

框架只统一 Registry、配置、checkpoint 和完整 `Sampler.sample()` 生命周期。不同算法
family 不需要共享 `predict()`、`drift()`、`score()`、`denoise()` 或 `step()` 数学接口。

### 取舍与去留

| 问题 | 决策 | 原因 |
| --- | --- | --- |
| Process 是否必需 | 保留为可选资产 | direct transform 或其他方法不应伪造 probability path |
| Process 根接口 | 只保留无数学方法的 `Process` | 数学能力由真实 family 定义，避免 universal API |
| Gaussian Process 契约 | 保留 `DiscreteGaussianDenoisingProcess` | 明确整数时间和 adjacent posterior，不暗示连续 Gaussian 兼容 |
| Dynamics Registry/YAML | 删除 | Dynamics 是 Builder 组装出的运行时对象，没有独立配置身份 |
| Process 工厂化 Dynamics | 删除 | Process 必须保持 model-free；组合责任属于 Builder/diagnostic |
| Sampler 公共接口 | 保留完整 `sample()` | 能表达 multistep、内部子步、自适应和 rejection；不强制单步 `step()` |
| condition/guidance 参数 | 不进入 Process 或 Sampler | 由 model callable、Dynamics wrapper 和 Builder 拥有 |
| trajectory API | 保留 Observer 事件流 | 避免复制 sample/trajectory 两套求解循环 |
| Gaussian schedule 状态 | 构造期生成 Process-owned snapshot | 消除可变 schedule 与缓存 posterior 之间的双重权威状态 |
| learnable schedule | 本 Stage 拒绝 | 需要动态 coefficient capability，不能静默套用固定快照语义 |
| checkpoint 缺失 Process | 省略 `process_state_dict` | 空 mapping 是合法 Process state，不能同时表示“不存在” |
| 旧 diffusion API/v4 checkpoint | 删除且不迁移 | Stage 3 尚未发布，保留兼容层会固化错误边界 |

### 已验证的 OCP 场景

- Gaussian family：同一 `DiscreteGaussianDenoisingProcess` 可复用 DDPM、DDIM 和自定义
  `GaussianDenoisingDynamics` wrapper。
- 新算法 family：测试私有 Flow Process、VectorField Dynamics、Sampler 和 Builder 可经过
  既有 Registry、checkpoint 与 sampling runtime，不修改核心 dispatch。
- direct transform：测试私有 Builder 在 `process: null` 下不创建 Dynamics 或 Sampler，
  仍可经过 checkpoint-only sampling、writer 和 manifest。
- task variation：condition、CFG、physics guidance 和 partial initialization 留在自定义
  Builder/Dynamics；核心 Gaussian Process 和 Sampler 不增加任务字段。

### 当前刻意保留的临时限制

以下内容不是 Stage 3 的最终架构承诺，应由后续 Stage 处理：

- 训练入口仍是 Gaussian epsilon bridge，并按当前 `objective` 选择 train step；正式
  `TrainingStrategy`、Loss 组合和 structured batch 解释留给 Stage 4。
- 顶层 `model`、`data` 和 `objective` 仍是必填项，即使某个 sampling-only Builder 理论上
  不需要全部资产；只有 `process` 在本 Stage 变为可选。
- EMA 只跟踪 primary inference model；多模型、教师模型和 Strategy auxiliary state 尚无
  checkpoint 契约。
- diffusion-quality diagnostic 仍是 Gaussian/image 专用能力，不代表通用 diagnostic 必须
  理解 Process 或 Sampler。
- `SamplingOutput` 仍整体驻留内存；streaming/chunked artifact 边界留给容量 Stage。

## Stage 4 审阅问题

开始训练层重构前，需要先确定以下问题：

1. 顶层训练选择是否收敛为单个 `training: {name, params}` Strategy 声明，并移除通用
   `objective` 字段？
2. Loss 是否值得拥有独立 Registry，还是默认作为 Strategy 私有实现，只在存在真实跨
   Strategy 复用时公开？
3. primary model 应继续由核心构建后注入 Strategy，还是由 Strategy 完整拥有单模型、
   多模型和冻结模型的构建？
4. optimizer、scheduler、EMA 和 gradient clipping 是 Strategy 组装的训练资产，还是继续
  由核心统一构建并通过窄 context 注入？
5. Strategy 的 auxiliary modules/state 应如何进入 checkpoint，同时保持 checkpoint-only
   sampling 只加载 Builder 需要的资产？

上述问题确定前，不应通过给现有 Trainer、Process 或 objective 增加更多可选字段来提前
兼容知识蒸馏、physics condition 或其他具体任务。
