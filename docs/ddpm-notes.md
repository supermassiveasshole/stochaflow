# DDPM 学习笔记：从 Motivation 到 Loss 与 Reverse

这份笔记按一条主线整理 DDPM（Denoising Diffusion Probabilistic Models）：先说明为什么要把数据逐步加噪到标准高斯，再引入 ELBO，接着说明 DDPM 中的观测变量、隐变量、proposal distribution、Markov trick 和高斯假设，最后推到训练 loss 与 reverse 采样。

## Motivation

生成建模的目标是学习数据分布 $q(x_0)$，然后从中采样。直接建模复杂高维数据分布很难，DDPM 采用一个绕路思路：

1. 设计一个固定的 forward process，把真实数据 $x_0$ 逐步破坏成接近标准高斯的 $x_T$。
2. 学习一个 reverse process，从 $x_T \sim \mathcal{N}(0, I)$ 一步步去噪回 $x_0$。
3. 用变分推断的 ELBO 把反向去噪分布训练出来。

核心直觉：

$$
x_0
\xrightarrow{\text{add noise}}
x_1
\xrightarrow{\text{add noise}}
\cdots
\xrightarrow{\text{add noise}}
x_T \approx \mathcal{N}(0, I)
$$

采样时走反方向：

$$
x_T
\xrightarrow{\text{denoise}}
x_{T-1}
\xrightarrow{\text{denoise}}
\cdots
\xrightarrow{\text{denoise}}
x_0
$$

## 符号约定

符号尽量沿用 DDPM 原文：

- 数据：$x_0 \sim q(x_0)$
- 隐变量链：$x_1, x_2, \dots, x_T$
- forward process：$q(x_{1:T} \mid x_0)$
- reverse process：$p_\theta(x_{0:T})$
- 噪声方差表：$\beta_1, \dots, \beta_T$
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
- 标准高斯噪声：$\epsilon \sim \mathcal{N}(0, I)$
- 网络预测噪声：$\epsilon_\theta(x_t, t)$

## Forward Process：逐步加噪到标准高斯

DDPM 先手工定义 forward process。它是一个固定的 Markov chain：

$$
q(x_{1:T} \mid x_0)
= \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

每一步加一小段高斯噪声：

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}\!\left(
  x_t;
  \sqrt{1 - \beta_t}x_{t-1},
  \beta_t I
\right)
$$

用 $\alpha_t = 1 - \beta_t$ 写得更紧凑：

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}\!\left(
  x_t;
  \sqrt{\alpha_t}x_{t-1},
  (1 - \alpha_t)I
\right)
$$

重参数化形式为：

$$
x_t
= \sqrt{\alpha_t}x_{t-1}
  + \sqrt{1 - \alpha_t}\epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0, I)
$$

### 为什么是根号加权

根号来自方差控制。假设 $x_{t-1}$ 的每个维度已经大致标准化，$\epsilon_t \sim \mathcal{N}(0, I)$，且二者独立。如果写成：

$$
x_t
= \sqrt{\alpha_t}x_{t-1}
  + \sqrt{1 - \alpha_t}\epsilon_t
$$

那么方差会按平方系数相加：

$$
\operatorname{Var}(x_t)
\approx
\alpha_t \operatorname{Var}(x_{t-1})
+ (1 - \alpha_t)I
\approx I
$$

所以 $\sqrt{\alpha_t}$ 控制 signal amplitude，$\sqrt{1-\alpha_t}$ 控制 noise amplitude；对应的 signal power 和 noise power 分别是 $\alpha_t$ 与 $1-\alpha_t$。这种写法让每一步噪声量由 $\beta_t = 1-\alpha_t$ 直接控制，同时让整体尺度稳定。

### 从 $x_0$ 直接采样到 $x_t$

逐步展开两步：

$$
x_1
= \sqrt{\alpha_1}x_0
  + \sqrt{1 - \alpha_1}\epsilon_1
$$

$$
\begin{aligned}
x_2
&= \sqrt{\alpha_2}x_1
   + \sqrt{1 - \alpha_2}\epsilon_2 \\
&= \sqrt{\alpha_2\alpha_1}x_0
   + \sqrt{\alpha_2(1-\alpha_1)}\epsilon_1
   + \sqrt{1-\alpha_2}\epsilon_2
\end{aligned}
$$

噪声项是独立高斯的线性组合，总方差为：

$$
\begin{aligned}
\alpha_2(1-\alpha_1) + (1-\alpha_2)
&= 1 - \alpha_1\alpha_2 \\
&= 1 - \bar{\alpha}_2
\end{aligned}
$$

归纳得到原文中的 closed form：

$$
q(x_t \mid x_0)
=
\mathcal{N}\!\left(
  x_t;
  \sqrt{\bar{\alpha}_t}x_0,
  (1-\bar{\alpha}_t)I
\right)
$$

也就是：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$

当 $t$ 变大，$\bar{\alpha}_t$ 逐渐接近 $0$，signal 部分 $\sqrt{\bar{\alpha}_t}x_0$ 被压小，noise 部分 $\sqrt{1-\bar{\alpha}_t}\epsilon$ 变主导。合理设置 $\beta_t$ 后：

$$
q(x_T \mid x_0) \approx \mathcal{N}(0, I)
$$

这解释了为什么 reverse 采样可以从标准高斯开始。

## Reverse Process：要学习的生成模型

Forward process 是固定的，真正需要学习的是 reverse process：

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

其中：

$$
p(x_T) = \mathcal{N}(0, I)
$$

DDPM 把每一步反向转移建模成高斯：

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}\!\left(
  x_{t-1};
  \mu_\theta(x_t, t),
  \Sigma_\theta(x_t, t)
\right)
$$

这个高斯形式是一处 heuristic 建模假设。理由是 forward 每一步只加入很小的高斯噪声，当 $\beta_t$ 足够小时，反向一步的条件分布可以用高斯近似。实践中常固定方差：

$$
\Sigma_\theta(x_t, t) = \sigma_t^2 I
$$

常见取法：

$$
\sigma_t^2 = \beta_t
\qquad \text{or} \qquad
\sigma_t^2 = \tilde{\beta}_t
$$

于是主要学习对象变成反向均值 $\mu_\theta(x_t, t)$。

## ELBO：先写通用形式

先看一般 latent variable model。观测变量为 $x$，隐变量为 $z$，联合分布为 $p_\theta(x, z)$。目标是最大化：

$$
\log p_\theta(x)
=
\log \int p_\theta(x, z)\,dz
$$

引入 proposal distribution $q(z \mid x)$：

$$
\begin{aligned}
\log p_\theta(x)
&=
\log
\int
q(z \mid x)
\frac{p_\theta(x, z)}{q(z \mid x)}
\,dz \\
&\ge
\mathbb{E}_{q(z \mid x)}
\left[
  \log p_\theta(x, z)
  - \log q(z \mid x)
\right]
\end{aligned}
$$

右侧是 ELBO：

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=
\mathbb{E}_{q(z \mid x)}
\left[
  \log p_\theta(x, z)
  - \log q(z \mid x)
\right]
$$

训练时常最小化 negative ELBO：

$$
-\mathcal{L}_{\mathrm{ELBO}}(x)
=
\mathbb{E}_{q(z \mid x)}
\left[
  \log q(z \mid x)
  - \log p_\theta(x, z)
\right]
$$

## DDPM 语境下的 ELBO

DDPM 中的变量对应关系：

$$
x \leftrightarrow x_0,
\qquad
z \leftrightarrow x_{1:T}
$$

观测变量是干净数据 $x_0$。隐变量是整条 noisy trajectory：

$$
x_{1:T} = (x_1, x_2, \dots, x_T)
$$

所以边缘似然为：

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
$$

把通用 ELBO 代入 DDPM 的变量：

$$
\mathcal{L}_{\mathrm{ELBO}}(x_0)
=
\mathbb{E}_{q(x_{1:T} \mid x_0)}
\left[
  \log p_\theta(x_{0:T})
  - \log q(x_{1:T} \mid x_0)
\right]
$$

对应 negative ELBO：

$$
L_{\mathrm{vlb}}
=
\mathbb{E}_{q(x_{1:T} \mid x_0)}
\left[
  \log q(x_{1:T} \mid x_0)
  - \log p_\theta(x_{0:T})
\right]
$$

这就是 DDPM 后续 loss 分解的起点。

## 为什么引入这个 Proposal Distribution

这里的 proposal distribution 取 fixed forward process：

$$
q(x_{1:T} \mid x_0)
=
\prod_{t=1}^T q(x_t \mid x_{t-1})
$$

选择它的原因直接服务后续推导和训练：

- 容易采样：用 closed form 可以直接得到任意 $x_t$。
- 密度可计算：每一步都是高斯，$\log q(x_t \mid x_{t-1})$ 有解析形式。
- 终点简单：$q(x_T \mid x_0)$ 接近 $\mathcal{N}(0, I)$，与先验 $p(x_T)$ 对齐。
- 后验可解析：$q(x_{t-1} \mid x_t, x_0)$ 可以写成高斯闭式。
- 训练局部化：ELBO 分解后，每个时间步对应一个局部去噪 KL。

关键点是第四条。DDPM 后面能推到均值匹配和噪声预测 loss，依赖这个后验闭式。

## Markov Trick：把 ELBO 拆成逐步 KL

先展开 negative ELBO：

$$
L_{\mathrm{vlb}}
=
\mathbb{E}_q
\left[
  \log q(x_{1:T} \mid x_0)
  - \log p_\theta(x_{0:T})
\right]
$$

代入两条 Markov chain：

$$
q(x_{1:T} \mid x_0)
=
\prod_{t=1}^T q(x_t \mid x_{t-1})
$$

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

得到：

$$
L_{\mathrm{vlb}}
=
\mathbb{E}_q
\left[
  \sum_{t=1}^T \log q(x_t \mid x_{t-1})
  - \log p(x_T)
  - \sum_{t=1}^T \log p_\theta(x_{t-1} \mid x_t)
\right]
$$

接下来用一个 Bayes rewrite。由于 forward chain 是 Markov chain：

$$
q(x_{t-1} \mid x_t, x_0)
=
\frac{
  q(x_t \mid x_{t-1})q(x_{t-1} \mid x_0)
}{
  q(x_t \mid x_0)
}
$$

这个式子把单步 forward likelihood $q(x_t \mid x_{t-1})$ 改写成后验项：

$$
q(x_t \mid x_{t-1})
=
q(x_{t-1} \mid x_t, x_0)
\cdot
\frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

把这些项代回 $L_{\mathrm{vlb}}$，中间的边缘分布会望远镜式相消，得到 DDPM 原文中的分解：

$$
L_{\mathrm{vlb}}
=
\mathbb{E}_q
\left[
  L_T
  + \sum_{t>1} L_{t-1}
  + L_0
\right]
$$

其中：

$$
L_T
=
D_{\mathrm{KL}}
\left(
  q(x_T \mid x_0)
  \,\|\, p(x_T)
\right)
$$

$$
L_{t-1}
=
D_{\mathrm{KL}}
\left(
  q(x_{t-1} \mid x_t, x_0)
  \,\|\, p_\theta(x_{t-1} \mid x_t)
\right)
$$

$$
L_0
=
-\log p_\theta(x_0 \mid x_1)
$$

这一步是 DDPM 推导里最重要的结构化结果：训练目标被拆成一个终点 prior KL、一组逐步 denoising KL、一个最后重建项。

## Forward Posterior 的闭式

中间 KL 需要真实后验：

$$
q(x_{t-1} \mid x_t, x_0)
$$

由于：

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}\!\left(
  x_t;
  \sqrt{\alpha_t}x_{t-1},
  (1-\alpha_t)I
\right)
$$

并且：

$$
q(x_{t-1} \mid x_0)
=
\mathcal{N}\!\left(
  x_{t-1};
  \sqrt{\bar{\alpha}_{t-1}}x_0,
  (1-\bar{\alpha}_{t-1})I
\right)
$$

两项相乘后仍然给出高斯后验：

$$
q(x_{t-1} \mid x_t, x_0)
=
\mathcal{N}\!\left(
  x_{t-1};
  \tilde{\mu}_t(x_t, x_0),
  \tilde{\beta}_t I
\right)
$$

原文中的方差：

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

原文中的均值：

$$
\tilde{\mu}_t(x_t, x_0)
=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
$$

这个闭式后验提供了监督信号：模型的 $p_\theta(x_{t-1} \mid x_t)$ 应该贴近它。

## 从逐步 KL 到均值匹配

考虑 $t > 1$ 的项：

$$
L_{t-1}
=
D_{\mathrm{KL}}
\left(
  q(x_{t-1} \mid x_t, x_0)
  \,\|\, p_\theta(x_{t-1} \mid x_t)
\right)
$$

两边都是高斯：

$$
q(x_{t-1} \mid x_t, x_0)
=
\mathcal{N}\!\left(
  x_{t-1};
  \tilde{\mu}_t(x_t, x_0),
  \tilde{\beta}_t I
\right)
$$

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}\!\left(
  x_{t-1};
  \mu_\theta(x_t, t),
  \sigma_t^2 I
\right)
$$

当 $\sigma_t^2$ 固定时，和 $\theta$ 相关的部分落在均值误差：

$$
L_{t-1}
=
\mathbb{E}_q
\left[
  \frac{1}{2\sigma_t^2}
  \left\|
    \tilde{\mu}_t(x_t, x_0)
    - \mu_\theta(x_t, t)
  \right\|^2
\right]
+ C
$$

$C$ 与 $\theta$ 无关。于是训练反向均值即可。

## 从均值匹配到噪声预测 Loss

DDPM 使用噪声预测参数化。先从 closed form 反解 $x_0$：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon
$$

所以：

$$
x_0
=
\frac{
  x_t - \sqrt{1-\bar{\alpha}_t}\epsilon
}{
  \sqrt{\bar{\alpha}_t}
}
$$

让网络预测噪声 $\epsilon_\theta(x_t, t)$，得到：

$$
\hat{x}_0
=
\frac{
  x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)
}{
  \sqrt{\bar{\alpha}_t}
}
$$

把 $\hat{x}_0$ 代入 $\tilde{\mu}_t(x_t, x_0)$ 的公式，可以得到原文常用的均值参数化：

$$
\mu_\theta(x_t, t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
  x_t
  -
  \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
  \epsilon_\theta(x_t, t)
\right)
$$

真实后验均值也可以用真实噪声 $\epsilon$ 写成：

$$
\tilde{\mu}_t(x_t, x_0)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
  x_t
  -
  \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
  \epsilon
\right)
$$

二者相减：

$$
\tilde{\mu}_t(x_t, x_0)
- \mu_\theta(x_t, t)
=
\frac{\beta_t}
     {\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}
\left(
  \epsilon_\theta(x_t, t) - \epsilon
\right)
$$

代回均值匹配项：

$$
L_{t-1}
=
\mathbb{E}_q
\left[
  \frac{\beta_t^2}
       {2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}
  \left\|
    \epsilon - \epsilon_\theta(x_t, t)
  \right\|^2
\right]
+ C
$$

原文进一步使用简化目标，去掉时间相关的权重，直接优化噪声预测 MSE：

$$
L_{\mathrm{simple}}(\theta)
=
\mathbb{E}_{t, x_0, \epsilon}
\left[
  \left\|
    \epsilon
    -
    \epsilon_\theta
    \left(
      \sqrt{\bar{\alpha}_t}x_0
      + \sqrt{1-\bar{\alpha}_t}\epsilon,
      t
    \right)
  \right\|^2
\right]
$$

这个式子仍然是一个期望，真实训练时用 Monte Carlo estimate 来近似它。每一步 SGD 不会枚举所有数据、所有时间步、所有噪声；训练会随机采样一批 $(x_0, t, \epsilon)$：

$$
x_0 \sim q(x_0),
\qquad
t \sim \mathrm{Uniform}(\{1,\dots,T\}),
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$

对单个样本，Monte Carlo 估计量是：

$$
\hat{L}_{\mathrm{simple}}(\theta)
=
\left\|
  \epsilon
  -
  \epsilon_\theta
  \left(
    \sqrt{\bar{\alpha}_t}x_0
    + \sqrt{1-\bar{\alpha}_t}\epsilon,
    t
  \right)
\right\|^2
$$

对 mini-batch $\{(x_0^{(i)}, t^{(i)}, \epsilon^{(i)})\}_{i=1}^B$，实际优化的 batch loss 是：

$$
\hat{L}_{\mathrm{batch}}(\theta)
=
\frac{1}{B}
\sum_{i=1}^B
\left\|
  \epsilon^{(i)}
  -
  \epsilon_\theta
  \left(
    \sqrt{\bar{\alpha}_{t^{(i)}}}x_0^{(i)}
    + \sqrt{1-\bar{\alpha}_{t^{(i)}}}\epsilon^{(i)},
    t^{(i)}
  \right)
\right\|^2
$$

这个 batch loss 是 $L_{\mathrm{simple}}(\theta)$ 的 Monte Carlo 近似。随着 batch size 增大，它对期望的估计方差会下降；随着训练迭代推进，随机采样会覆盖不同数据、时间步和噪声。

训练时可以按下面的逻辑理解：

1. 采样真实数据 $x_0 \sim q(x_0)$。
2. 采样时间步 $t \sim \mathrm{Uniform}(\{1,\dots,T\})$。
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I)$。
4. 构造 noisy sample：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon
$$

5. 训练网络预测这份噪声：

$$
\left\|
  \epsilon - \epsilon_\theta(x_t, t)
\right\|^2
$$

这样，原本复杂的变分下界训练，最终落到一个噪声回归问题。

## Reverse：采样时怎么做

训练完成后，采样来自学习到的生成联合分布：

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

把这个联合分布按时间顺序展开：

$$
\begin{aligned}
p_\theta(x_{0:T})
=
&p(x_T)
p_\theta(x_{T-1} \mid x_T)
p_\theta(x_{T-2} \mid x_{T-1})
\cdots \\
&\cdot
p_\theta(x_1 \mid x_2)
p_\theta(x_0 \mid x_1)
\end{aligned}
$$

所以采样流程也按这个分解来做。第一步从先验采样：

$$
x_T \sim \mathcal{N}(0, I)
$$

然后依次采样：

$$
x_{T-1} \sim p_\theta(x_{T-1} \mid x_T)
$$

$$
x_{T-2} \sim p_\theta(x_{T-2} \mid x_{T-1})
$$

一直到：

$$
x_0 \sim p_\theta(x_0 \mid x_1)
$$

每一步的条件分布设为高斯：

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}\!\left(
  x_{t-1};
  \mu_\theta(x_t, t),
  \sigma_t^2 I
\right)
$$

因此从这个高斯里采样可以写成：

$$
x_{t-1}
=
\mu_\theta(x_t, t)
+ \sigma_t z
$$

其中：

$$
z \sim \mathcal{N}(0, I)
\quad \text{if } t > 1,
\qquad
z = 0
\quad \text{if } t = 1
$$

这里 $t=1$ 时取 $z=0$，最后一步直接给出 $x_0$，避免在最终输出里额外注入噪声。

### 方法一：直接预测反向均值

第一种做法是让网络直接输出：

$$
\mu_\theta(x_t, t)
$$

采样时直接代入：

$$
x_{t-1}
=
\mu_\theta(x_t, t) + \sigma_t z
$$

这种写法最贴近反向高斯分布本身：

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}\!\left(
  x_{t-1};
  \mu_\theta(x_t, t),
  \sigma_t^2 I
\right)
$$

训练时，中间 KL 项会推动 $\mu_\theta(x_t,t)$ 贴近真实后验均值 $\tilde{\mu}_t(x_t,x_0)$。

### 方法二：预测噪声，再换算均值

DDPM 原文更常用噪声预测参数化。网络输出：

$$
\epsilon_\theta(x_t, t)
$$

先用它估计干净样本：

$$
\hat{x}_0
=
\frac{
  x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)
}{
  \sqrt{\bar{\alpha}_t}
}
$$

再把 $\hat{x}_0$ 放进真实后验均值公式：

$$
\tilde{\mu}_t(x_t, x_0)
=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
$$

得到用于采样的模型均值：

$$
\mu_\theta(x_t, t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
  x_t
  -
  \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
  \epsilon_\theta(x_t, t)
\right)
$$

然后仍然从同一个反向高斯里采样：

$$
x_{t-1}
=
\mu_\theta(x_t, t) + \sigma_t z
$$

直观上，方法一直接学习“下一步去哪里”；方法二先学习“当前样本里有多少噪声”，再由噪声估计换算出去噪方向。DDPM 的简化 loss 对应方法二，因为训练目标直接是：

$$
\left\|
  \epsilon - \epsilon_\theta(x_t,t)
\right\|^2
$$

完整采样算法可以写成：

1. 采样 $x_T \sim \mathcal{N}(0,I)$。
2. 对 $t=T,T-1,\dots,1$：
3. 用网络得到 $\epsilon_\theta(x_t,t)$。
4. 计算 $\mu_\theta(x_t,t)$。
5. 采样 $z \sim \mathcal{N}(0,I)$；当 $t=1$ 时取 $z=0$。
6. 更新 $x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z$。

最后得到的 $x_0$ 就是生成样本。

## 一页总结

- Motivation：把复杂数据分布逐步加噪到简单高斯，再学习反向去噪链。
- Forward：$q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)$。
- 根号权重：控制 signal/noise 的方差比例，让尺度稳定。
- Closed form：$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$。
- ELBO 中观测变量是 $x_0$，隐变量是 $x_{1:T}$。
- Proposal distribution 是 fixed forward process $q(x_{1:T} \mid x_0)$。
- Markov trick 把 negative ELBO 拆成 $L_T + \sum_{t>1}L_{t-1} + L_0$。
- 高斯假设让逐步 KL 变成反向均值匹配。
- 噪声预测参数化把均值匹配化成 $L_{\mathrm{simple}}$。
- Reverse 采样从 $x_T \sim \mathcal{N}(0,I)$ 开始，按 $T \to 1$ 逐步生成 $x_0$；反向均值可以直接预测，也可以由噪声预测 $\epsilon_\theta(x_t,t)$ 换算得到。
