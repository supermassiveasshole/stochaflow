# Stochaflow

`Stochaflow` 是一个面向扩散模型实验的 Python 项目骨架，当前阶段只完成目录初始化、配置文件占位和文档整理，便于后续按模块逐步实现 DDPM / DDIM 训练、采样与评估流程。

## 项目目标

- 基于 `src` 布局组织代码，便于包管理与测试
- 按功能拆分 `data / models / diffusion / training / sampling / utils`
- 预留 `configs`、`scripts`、`tests`、`notebooks`、`outputs`、`assets` 等常用目录
- 支持后续扩展到 MNIST、CIFAR-10 等数据集上的扩散模型实验

## 当前状态

当前仓库是“项目初始化版本”：

- 已创建目录结构
- 已创建配置文件占位
- 已创建脚本与模块文件占位
- 已创建测试文件占位
- 尚未实现具体训练、采样、评估逻辑

如果你希望后续自己补代码，现在可以直接在对应模块内继续实现。

## 目录结构

```text
stochaflow/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── ddpm_mnist.yaml
│   ├── ddpm_cifar10.yaml
│   └── ddim_cifar10.yaml
├── scripts/
│   ├── train.py
│   ├── sample.py
│   └── eval.py
├── src/
│   └── stochaflow/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── datasets.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── unet.py
│       │   ├── blocks.py
│       │   └── embeddings.py
│       ├── diffusion/
│       │   ├── __init__.py
│       │   ├── schedules.py
│       │   ├── ddpm.py
│       │   ├── ddim.py
│       │   └── objectives.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── ema.py
│       ├── sampling/
│       │   ├── __init__.py
│       │   ├── sampler.py
│       │   └── grid.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── seed.py
│           ├── checkpoint.py
│           └── logging.py
├── tests/
│   ├── test_schedules.py
│   ├── test_ddpm_shapes.py
│   └── test_unet_shapes.py
├── notebooks/
│   └── ddpm_sanity_check.ipynb
├── outputs/
│   └── .gitkeep
└── assets/
    └── .gitkeep
```

## 模块说明

### `configs/`

存放实验配置文件，占位内容按实验类型区分：

- `ddpm_mnist.yaml`
- `ddpm_cifar10.yaml`
- `ddim_cifar10.yaml`

后续可以把数据集参数、模型参数、训练超参、采样参数都统一放在这里。

### `scripts/`

命令行入口脚本目录：

- `train.py`：训练入口
- `sample.py`：采样入口
- `eval.py`：评估入口

当前仅为占位文件，后续可接入参数解析和配置加载。

### `src/stochaflow/data/`

负责数据集相关逻辑，例如：

- 数据集下载与加载
- 训练/验证数据预处理
- dataloader 构建

### `src/stochaflow/models/`

负责模型结构定义，例如：

- U-Net 主干
- 残差块、注意力块
- 时间步嵌入

### `src/stochaflow/diffusion/`

负责扩散过程本身，例如：

- beta schedule
- DDPM 正向与反向过程
- DDIM 采样过程
- 训练目标定义

### `src/stochaflow/training/`

负责训练流程，例如：

- trainer 主循环
- 损失函数
- EMA 权重更新

### `src/stochaflow/sampling/`

负责采样与结果可视化，例如：

- 采样调度器
- 图像网格拼接与保存

### `src/stochaflow/utils/`

负责公共工具，例如：

- 配置读取
- 随机种子设置
- checkpoint 管理
- 日志管理

### `tests/`

预留单元测试目录。建议后续优先补这些测试：

- schedule 输出长度与范围
- DDPM / DDIM 张量 shape
- U-Net 前向输出 shape

### `notebooks/`

预留实验验证 notebook，可用于：

- schedule 可视化
- 单步前向 sanity check
- 采样结果观察

## 开发建议

推荐按下面顺序推进实现：

1. 先完成 `utils/config.py` 与配置读取
2. 再完成 `data/datasets.py`
3. 实现 `models/` 中的 U-Net 与基础模块
4. 实现 `diffusion/schedules.py` 与 `ddpm.py`
5. 接入 `training/trainer.py`
6. 最后补 `sample.py`、`eval.py` 和测试

## 本地开发

当前项目使用 `pyproject.toml` 管理元数据，并采用 `src` 布局。

可选的初始化步骤示例：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## License

本项目使用仓库中的 `MIT License`。
