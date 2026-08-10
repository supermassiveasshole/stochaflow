# Sampling 调用方式：Hydra 完成后的复查备忘

> 文档类型：条件性复查，不是开发计划
>
> 排期状态：不参与排期
>
> 当前可用性：Sampling 现在可以正常使用；Hydra 完成后只复查调用方式是否仍然清楚。

这份文件不表示 sampling 重构尚未完成。它只保存一个以后要重新检查的问题：
[Hydra 配置迁移](hydra-configuration-composition-migration-plan.md)改变配置组织方式以后，sampling
的 YAML 和 Python 调用是否仍然好用。现在没有理由提前改接口，未来复查也完全可以得出
“保持现状”的结论。

## 用户现在怎样生成样本

用户准备一份完整 sample 配置，配置中写明 checkpoint、任务、采样器、样本数量、条件和
输出方式，然后运行：

```bash
stochaflow sample \
  --config path/to/sample.yaml \
  --device cuda \
  --output-dir outputs/samples/run-a
```

采样外层代码负责读取配置、加载 checkpoint 中声明的推理组件、准备输入并发布结果。
任务自己的 `SamplingBuilder.run()` 只负责生成样本并返回内存结果。这两层责任已经确定，
不属于复查范围。

完整用法见[配置与工作流文档](../configuration/workflows.md)。

## 为什么现在不改

Hydra 迁移尚未完成。现在设计新的“采样调用参数对象”或“供 Python 程序直接调用的稳定
函数”，可能只是围绕旧配置布局解决暂时问题。当前 checkpoint 与采样方式已有代码和测试
支撑，没有证据要求立即替换它。

## Hydra 完成后具体检查什么

维护者会拿真实的 sample 配置、CLI 调用和 Python 宿主代码回答这些普通问题：

1. 用户是否仍能在一份文件中看清 checkpoint、数量、条件和输出？
2. 任务自己的选项是否容易找到，还是被大量通用字段淹没？
3. Python 程序是否反复手写同一套“读取配置并运行 sampling”的重复调用代码？
4. 只返回内存对象、不写文件的结果是否真的有两个以上的使用者？
5. 一个 checkpoint 保存多个推理组件时，当前选择方式是否仍然清楚？

只收集实际失败、重复代码和用户操作记录；不因为旧计划曾写过某个类型名就默认实现它。

## 复查可能得到什么结果

- 保持现有 CLI 和内部 `run_sampling()`，不增加公共 API；
- 只改善 sample 配置的组织和文档；
- 如果多个 Python 使用者确实重复同一调用，再提出一个稳定的 Python 调用函数；
- 如果任务选项难以发现，只调整任务配置边界；
- 如果证据互相冲突，继续保留现状并记录原因。

任何新接口都必须先写成一个完整使用过程：调用者传入什么、得到什么、错误怎样报告。
候选接口在获批和实现前都不是公共契约。

## 不会重新打开的旧设计

- 不恢复把 sample 配置当作 training config 的不完整补丁；
- 不恢复训练结束后自动 sampling；
- 不让 SamplingBuilder 负责 checkpoint、配置权威或外层文件发布；
- 不让框架通用代码按任务名称增加分支；
- 不因复查而改变当前 checkpoint 文件格式和采样结果文件的含义。

旧文件记录的是已经完成的“不完整采样请求”重构。它明确禁止恢复的旧行为和后来新增的
复查问题都保存在
[Sampling 复查资料](notes/sampling-request-config-refactor/review-notes.md)。
