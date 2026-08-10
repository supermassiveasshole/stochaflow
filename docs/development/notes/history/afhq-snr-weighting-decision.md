# AFHQ-v2 Gaussian SNR loss-weighting 实验结论

> 文档性质：已完成实验的历史决策，不是开发计划
>
> 实验关闭：2026-08-06

本记录保存 `P2` 这个旧简称背后的实际实验结论，避免四份历史删除候选获批后只能从 Git
猜测当时为什么退休该 recipe。这里的 `P2` 指 SNR loss weighting，不是优先级。

## 做了什么对照

- 在当前 canonical ADM 上运行了 gamma 1 的 SNR weighting 实验；
- 保留了 cosine + epsilon 和 linear + epsilon 的训练记录；
- 运行了匹配的 linear + epsilon + gamma 0 对照；
- 用相同的一轮正式评估比较 gamma 0 与 gamma 1。

受控比较中，gamma 1 在总体和每个类别上都略差，但差异没有统计显著性。cosine 实验学到
了动物结构，同时持续出现色彩噪声；linear 两组也没有给出足以维护第二套公开训练 recipe
的质量证据。这些结果只能说明“没有验证出可靠收益”，不能证明所有 SNR weighting 方法
永远无效。

## 从旧计划保留的研究思想

下面四点解释当时为什么把工作拆开。它们是历史研究方法，不是恢复 P2 支持的实施说明。

1. **先修模型拓扑，再比较 loss weighting。** 旧 ADM 实现与 canonical ADM 的 skip 和
   decoder 结构不同。若同时改拓扑和 weighting，就无法知道结果来自哪一项。因此拓扑修正
   作为正确性工作独立完成；这一决定已经进入当前实现。P2 实验随后单独结束并退休。
2. **论文复刻与产品实验不是同一条证据。** 旧设想把单域、无条件的 AFHQ-Dog 论文对照，
   与三类、带条件和 CFG 的 AFHQ-v2 产品实验分开。两者的数据、模型、采样和要回答的问题
   都不同，不能用一条结果冒充另一条结果。这两条 P2 实验线现在都不再继续。
3. **无法冻结全部历史事实时，要降低结果名称的强度。** 当时无法从公开资料确认精确数据
   版本、文件清单、metric 实现、seed 和 checkpoint 选择规则，所以最多只能计划发布
   “P2-compatible AFHQ-v2 Dog reproduction”，不能声称精确复现论文 AFHQ-D 数值。
4. **P2 weighting 属于 Gaussian 训练语义。** 旧设计只让它作用于 epsilon prediction 的
   simple loss，不作用于 learned-range variance 的 variational-bound 项；也不把它包装成
   通用 Objective、通用 weighting registry 或所有生成方法都要实现的接口。

这些思想保留下来，是为了说明实验归因和框架边界。它们不构成 future-support 承诺；任何
重新实现仍须满足下文的重新讨论条件。

## 最终决定

- 不支持 AFHQ-v2 Gaussian SNR loss-weighting recipe；
- 不继续这些实验，也不把普通 diagnostics 当成正式 Evaluation；
- Metrics 保持任务无关，checkpoint 选择继续使用完整 validation Evaluation；
- 维护路线继续使用 fresh canonical ADM graph、`[1,2,3,4]` / 16×16 scale layout、cosine、
  v-prediction 和 learned-range variance。

最后一条是拓扑、variance、sampling 设置和训练预算共同形成的质量候选，不是一次只改变
单个因素的 ablation，不能把最终质量结果单独归因给 learned-range variance。

## 什么时候才允许重新讨论

重新引入任何 SNR weighting 方法，都必须先有新的路线图决定、单独命名的具体 recipe，以及
与当前基线匹配的正式 validation。这个历史记录本身不授权重新实施。

更完整的机器本地 manifest、metrics、diagnostics 和 Evaluation bundle 仍属于当时的实验
记录；它们不是当前仓库能力，也不应成为日常 checkpoint 保留要求。
