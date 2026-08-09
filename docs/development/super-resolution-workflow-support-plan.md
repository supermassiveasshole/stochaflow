# 内置超分辨率工作流支持计划

> 工作状态：候选
>
> 当前结论：仓库已有超分辨率数据组合和教程，但没有内置、可独立运行且可被
> 其他操作组合的训练、恢复、artifact 发布和正式 Evaluation 工作流。
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 最后核对：2026-08-09

## 完成后用户能做什么

用户可以训练一个超分辨率模型，选择 checkpoint 对低分辨率图像执行恢复，发布
可追溯的高分辨率图像 artifact，并用任务自有的正式 Evaluation 比较 checkpoint。
同一恢复操作既能独立读取输入，也能消费上游 sampling 操作发布的图像 artifact。

首个可维护结果使用确定性前向模型验证完整产品路径。条件 Gaussian 超分辨率在基础路径稳定后
作为后续候选，不把 diffusion 作为任务契约的前提。blind SR、
Real-ESRGAN 风格二阶 degradation、perceptual/adversarial training 和 general restoration
继续保留为独立重审方向，不进入首版完成条件。

## 当前仓库已经支持什么

- `super_resolution` DataBuilder 能组合低分辨率输入和高分辨率参考数据。
- 公开教程展示了相关数据路径和条件生成基础。
- `TrainingBuilder`/`TrainingStrategy` 可以实现任务特定训练，而不要求 core 理解 batch 字段。
- `SamplingBuilder` 可以解释任务输入、完成初始化和 batching、组合任务特定 inference adapter，
  并返回 writer-ready output；sampling runtime 负责验证 output、运行 writers 和原子发布。
- EvaluationBuilder 可以为任务声明 metric provider、预处理、sample identity 和结果。
- sampling 和 Evaluation 已有各自独立的完整配置，并能投影 checkpoint inference assets。

当前基础没有提供内置 SR model、objective、SamplingBuilder、writer、正式 profile 或
从 sampling artifact 读取输入的产品工作流。

## 还没有支持什么

- 没有第一方确定性 SR extension 和维护配置。
- 没有统一但足够窄的低分辨率输入与恢复输出 artifact contract。
- 没有面向大图的 tiling/overlap/blending 实现和边缘正确性测试。
- 没有任务自有的 PSNR/SSIM 等正式 Evaluation protocol。
- 没有把 sampling artifact 显式绑定为 SR 输入的组合示例。
- 没有条件 Gaussian SR 的完整训练、checkpoint sampling 和质量证据。
- 没有 unknown/mixed degradation 的 blind SR 数据、训练与正式 Evaluation protocol。
- 没有 Real-ESRGAN 风格二阶 degradation producer、参数 provenance 或合成/真实域验证。
- 当前 automatic loop 不支持 adversarial 多 optimizer lifecycle，也没有获批的 perceptual/
  adversarial SR training family。
- 没有跨 denoising、deblurring、deblocking 等任务证据支持 general restoration abstraction。

## 什么时候可以开始或重新审查

路线图选择内置超分辨率为 `Next` 后即可开始确定性基线。开始前必须指定：

- 一个可在 CI 中有界运行的小数据配置；
- 一个独立 extension 负责人，不向 core 添加图像字段；
- 输入/输出 artifact 的身份和完整性规则；
- 正式 Evaluation 使用的数据 split、预处理和 metric provider；
- 首版是否必须支持 tiling，或先明确限制最大输入尺寸。

条件 Gaussian SR 只有在确定性路径的训练、恢复、writer 和 Evaluation 全部通过后
才能开始。

后续方向只在下面各自条件成立后重审，且不会自动扩大确定性首版：

| 后续方向 | 重审触发 | Owner |
| --- | --- | --- |
| Conditional Gaussian SR | 确定性训练、恢复、writer 和正式 Evaluation 已验收，并且被选择的产品结果确实需要随机生成或感知/失真权衡 | SR task extension 与 Gaussian family 负责人 |
| Blind SR | 至少一个真实输入域无法提供可信 paired degradation，且确定性 paired 基线已成为正确性对照 | 独立 blind-SR task/extension 负责人 |
| Real-ESRGAN 风格二阶 degradation | blind-SR 方向已被选择，并证明一次 degradation 无法覆盖目标域；完整参数分布、顺序和随机性可版本化 | blind-SR extension 的 DataBuilder/degradation provider 负责人 |
| Perceptual/adversarial loop | 确定性像素基线已验收，但正式 perception 证据证明仅重建 loss 不足；训练 loop family 与资源预算另获批准 | SR TrainingBuilder/Strategy 负责人；多 optimizer lifecycle 由独立 training-loop 负责人 |
| General restoration | 至少两个非 scale-only 的持续维护任务重复同一输入/输出、artifact 和 Evaluation 约定 | 新的 general-restoration 计划与 task-extension 负责人，不由 SR core 预建 |
| 任意或非整数倍率 | 固定整数倍率首版已验收，至少两个真实任务需要不同或连续倍率，并能定义训练配对、geometry 与 Evaluation | SR DataBuilder、model adapter 与 Evaluation 负责人 |
| Alpha 或 16-bit 图像 | 真实输入需要透明度或高位深，codec、范围、writer、artifact 与 metric 规则已经固定 | SR input/output adapter、writer 与 Evaluation provider 负责人 |
| Video SR | 一个视频恢复产品被单独选择，帧序、时间身份、数据、时序指标和资源预算已明确 | 独立 video-SR task extension 负责人 |
| Diffusion upscaler | 确定性首版已验收，正式证据表明随机 upscaler 对目标质量有必要，并能与确定性基线同协议比较 | 独立 diffusion-SR task、Gaussian family 与 Evaluation 负责人 |

## 要完成哪些工作

### 固定任务输入和输出

- 动作：由 SR extension 定义低分辨率输入、高分辨率参考、scale factor 和 sample
  identity；core 继续把 batch 当作结构化 `Any`。
- 原因：训练、恢复和 Evaluation 必须解释相同的图像对应关系。
- 影响范围：DataBuilder、artifact manifest 和任务文档。
- 交付物：训练数据、inference 输入和恢复输出的明确约定。
- 验证方法：尺寸、scale、配对、重复 sample ID 和不完整 artifact 的失败测试。
- 完成条件：替换另一种兼容数据来源不需要修改 core runtime。

### 交付确定性训练基线

- 动作：在 extension 中实现 model、objective、TrainingBuilder 和 Strategy，并使用
  当前 automatic training loop。
- 原因：先验证产品生命周期，再增加随机生成方法的复杂度。
- 影响范围：第一方 extension、配置、示例和 checkpoint。
- 交付物：有界 train config、严格 resume config 和选择 checkpoint 的规则。
- 验证方法：短训练、resume、loss/metric 记录和 checkpoint portability tests。
- 完成条件：训练路径不引入任务名称分支或第二套 optimizer lifecycle。

### 提供独立恢复操作

- 动作：实现任务 SamplingBuilder 或窄 inference adapter，接受显式低分辨率输入并
  返回恢复后的图像与 sample identity。
- 原因：恢复必须可独立使用，不能只作为训练后的隐式 hook。
- 影响范围：sampling composition、input loader 和 output writer。
- 交付物：checkpoint-backed restore config、writer 和 manifest。
- 验证方法：batch size、顺序、输出尺寸、raw/EMA 选择和重复运行测试。
- 完成条件：SamplingBuilder 负责任务输入、初始化和按 `num_samples`/`batch_size` batching；
  sampling runtime 只负责外层执行、完整 output 校验、writers 和原子发布。

### 决定并验证大图切片策略

- 动作：若首版纳入 tiling，明确 tile size、overlap、padding、blend 和边界裁剪规则；
  否则在配置和文档中拒绝超出限制的输入。
- 原因：隐式切片会改变像素、显存需求和 Evaluation 结果。
- 影响范围：任务 inference adapter 和 manifest protocol facts。
- 交付物：确定的切片策略或明确的输入限制。
- 验证方法：奇数尺寸、边缘 tile、overlap 拼接接缝、不同 batch 行为一致性和显存上限测试。
- 完成条件：相同配置和输入得到确定的空间拼接结果。

### 建立任务自有的正式 Evaluation

- 动作：定义配对 reference 数据、codec/quantization、预处理、PSNR/SSIM provider
  identity，以及需要时的 perceptual metric。
- 原因：训练 scalar 或普通 sampling output 不能替代正式质量证据。
- 影响范围：EvaluationBuilder、metric provider、profile 和公开教程。
- 交付物：checkpoint Evaluation 和 prediction-artifact offline replay 配置。
- 验证方法：sample completeness、live/offline 行为一致性、provider identity 和 reference
  mismatch tests。
- 完成条件：两个 checkpoint 只有在相同 protocol digest 下才进入普通比较。

### 允许消费上游 sampling artifact

- 动作：为完整图像 sampling artifact 提供显式 SR input binding，不扫描任意目录。
- 原因：生图后超分必须保留上游 sample identity 和来源。
- 影响范围：operation result binding、SR input adapter 和组合示例。
- 交付物：`sample -> super resolve -> evaluate` 示例。
- 验证方法：digest、顺序、缺失样本、重复样本和不兼容 codec 的失败测试。
- 完成条件：同一 SR operation 可独立运行或消费受治理的上游 artifact。

## 如何证明已经完成

- 有界 train、strict resume、checkpoint restore 和 writer tests。
- independent compatible DataSource/DataBuilder substitution tests。
- 确定性恢复的尺寸、顺序、identity 和不同 batch 一致性测试。
- 若支持 tiling，包含边缘、overlap 和非整除尺寸 tests。
- 正式 Evaluation 的 live/offline 一致性和完整性测试。
- sampling artifact 到 SR artifact 的 end-to-end composition test。
- 用户文档给出独立恢复和生图后超分两个可运行例子。
- 后续方向不作为首版验收；若单独重开，必须满足自己的触发证据并由对应负责人验收。

## 明确不包含什么

- 不把图像 condition、scale、tile 或 reference 字段加入通用 core config。
- 不建立 universal image pipeline 或按 `super_resolution` 名称分支的 runner。
- 不把训练后的自动恢复作为默认行为。
- 不在确定性基线完成前承诺 Gaussian、video SR、diffusion upscaler、任意倍率、alpha 或 16-bit。
- 不在首版实现 blind SR、Real-ESRGAN 二阶 degradation、perceptual/adversarial loop 或 general
  restoration；暂停不表示删除这些未来构想。
- 不用 loss flag 把 adversarial 多 optimizer lifecycle 偷塞进当前 automatic Trainer。
- 不把普通 PSNR/SSIM 数字写成跨 protocol 可比较的证据。

## 详细设计和研究资料在哪里

本节不属于首版“要完成哪些工作”。上面的触发表决定何时重审以及由谁负责；只有被路线图单独
选择的方向才执行相应工作。

### 条件 Gaussian SR

- 动作：复用现有 Gaussian family 的窄能力，增加条件 model adapter、objective、
  `SamplingBuilder` 和质量证据。
- 原因：随机方法不应改变通用 SR 输入和输出约定。
- 交付物：条件 Gaussian train/sample/evaluate 配置与对照结果。
- 验证方法：condition 使用、checkpoint restore、sampler compatibility 和质量测试。
- 完成条件：确定性与 Gaussian 方法共享任务 artifact 和 Evaluation 边界，core 不按方法名称
  分支。

### Blind、二阶 degradation、感知训练与 general restoration

- 动作：只按触发表分别建立 blind SR、Real-ESRGAN 风格二阶 degradation、perceptual/
  adversarial loop 或 general restoration 的短期提案。
- 原因：degradation uncertainty、合成数据分布、GAN optimization 和跨任务抽象是四种不同职责，
  不能打包成“高级 SR”开关。
- 交付物：每个被重开的方向各自拥有真实数据/使用方、版本化协议、负责人、资源预算、失败模型
  和独立验收；未触发方向继续保留。
- 验证方法：按实际获批方向选择执行 blind synthetic/real domain separation、degradation
  parameter/seed replay、perceptual provider identity、adversarial optimizer/step ordering，或至少
  两个 general-restoration task substitution tests。
- 完成条件：任何方向都不修改确定性 SR artifact/Evaluation 约定，不把多 optimizer 塞入当前
  automatic loop，也不把任务字段提升为通用 core schema。

### 任意倍率、Alpha/16-bit、Video 与 diffusion upscaler

- 动作：只在触发表对应证据成立后，为被选择的输入表示或模型 family 制定独立短期计划。
- 原因：倍率、像素表示、时间维度和随机生成分别改变不同的数据、writer 与 Evaluation 规则。
- 交付物：对应方向的输入输出约定、数据身份、writer、正式 Evaluation、资源预算和失败测试。
- 验证方法：由对应 Owner 在短期计划中固定，不复用确定性 x4 首版的隐含假设。

### 相关资料

- [原工作流文档中的 SR 详细工程方案和测试矩阵](notes/default-workflow-pipeline-support-plan/design-archive.md)
- [内置操作与工作流组合计划](default-workflow-pipeline-support-plan.md)
- [当前 Data 组合边界](../configuration/data-pipeline.md)
- [正式 Evaluation 后续计划](post-training-evaluation-support-plan.md)
