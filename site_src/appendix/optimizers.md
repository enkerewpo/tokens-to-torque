---
title: "附录 B · 优化器速查：从梯度下降到 AdamW"
---

看懂 [Day 00 §2.4](../days/day00.md) 的显存账需要知道 Adam 每个参数存了什么、为什么要存。这一页从最朴素的梯度下降开始，每加一个部件都先说它解决什么问题，最后用一个能手算的例子把 Adam 走两步。

## B.1 梯度下降

训练就是找一组参数 $\theta$ 让损失 $\mathcal{L}(\theta)$ 尽量小。梯度 $g=\nabla_\theta\mathcal{L}$ 指向损失**上升最快**的方向，所以往反方向挪一小步：

$$
\theta_{t}=\theta_{t-1}-\eta\,g_t
$$

$\eta$ 是学习率，决定步子多大。每一步只需要当前梯度，**不需要记住任何过去的东西**——这是它显存最省的原因，也是它的全部问题的来源。

**例。** $\mathcal{L}(\theta)=\theta^2$，梯度 $g=2\theta$。从 $\theta_0=1$、$\eta=0.1$ 出发：$\theta_1=1-0.1\times2=0.8$，$\theta_2=0.8-0.1\times1.6=0.64$，一路往 0 走。

## B.2 问题一：梯度是噪声，要平均——动量

真实训练里每步的梯度是在一个小 batch 上算的，方向抖得厉害：这一步指东，下一步指东北，再下一步指东南。直接跟着走就是来回晃，平均下来才是往东。

**动量**（momentum）[^polyak] 不直接用 $g_t$，而是维护一个梯度的**指数滑动平均** $m_t$：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t,
\qquad
\theta_t=\theta_{t-1}-\eta\,m_t
$$

$\beta_1$ 通常取 0.9。把递推展开就能看到它在做什么：

$$
m_t=(1-\beta_1)\big(g_t+\beta_1 g_{t-1}+\beta_1^2 g_{t-2}+\cdots\big)
$$

越近的梯度权重越大，越远的按 $\beta_1^k$ 衰减——是一个“记性有限”的平均，大约记住最近 $1/(1-\beta_1)=10$ 步。抖动的分量互相抵消，一致的分量累积起来。

代价：**每个参数多存一个数 $m$**，训练全程保留。

## B.3 问题二：不同参数的梯度尺度差几个量级——逐参数归一化

一个网络里，embedding 的梯度可能是 $10^{-5}$ 量级，某个 bias 的梯度可能是 $10^{-1}$。用同一个 $\eta$，前者纹丝不动，后者一步跳飞。想给每个参数配自己的学习率，但几十亿个参数不可能手调。

RMSProp[^rmsprop] 的办法：再维护一个**梯度平方的**指数滑动平均 $v_t$，用它的平方根去除梯度：

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^2,
\qquad
\theta_t=\theta_{t-1}-\eta\,\frac{g_t}{\sqrt{v_t}+\epsilon}
$$

$\beta_2$ 通常取 0.999（记住最近约 1000 步），$\epsilon$ 是防止除零的小常数（$10^{-8}$）。$\sqrt{v_t}$ 估计的是这个参数梯度的**典型大小**，除掉它之后 $g_t/\sqrt{v_t}$ 大约在 $\pm1$ 附近——**不管原始梯度是 $10^{-5}$ 还是 $10^{-1}$，每一步的实际位移都是 $\eta$ 量级**。这就是“逐参数自适应学习率”的含义。

代价：**每个参数再多存一个数 $v$**。

## B.4 Adam：两者合起来，再修一个偏差

Adam[^adam] 就是同时用 $m$（方向）和 $v$（尺度）：

$$
\begin{aligned}
m_t&=\beta_1 m_{t-1}+(1-\beta_1)\,g_t\\
v_t&=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^2\\
\hat m_t&=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}\\
\theta_t&=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$

第三行是**偏差修正**。$m$、$v$ 都从 0 起步，前几步的滑动平均被那个 0 严重拉低（$m_1=(1-\beta_1)g_1=0.1\,g_1$，只有真实梯度的十分之一）。除以 $1-\beta^t$ 把这个起步偏差补回来；$t$ 大了以后 $\beta^t\to0$，修正项趋于 1，自动失效。

### 手算两步

$\mathcal{L}(\theta)=\theta^2$，$g=2\theta$，$\theta_0=1$，$\eta=0.1$，$\beta_1=0.9$，$\beta_2=0.999$，忽略 $\epsilon$。

**第 1 步。** $g_1=2$。

| 量 | 计算 | 值 |
|---|---|---|
| $m_1$ | $0.9\times0+0.1\times2$ | $0.2$ |
| $v_1$ | $0.999\times0+0.001\times4$ | $0.004$ |
| $\hat m_1$ | $0.2/(1-0.9)$ | $2$ |
| $\hat v_1$ | $0.004/(1-0.999)$ | $4$ |
| 步长 | $0.1\times2/\sqrt4$ | $0.1$ |
| $\theta_1$ | $1-0.1$ | $0.9$ |

注意如果**不做**偏差修正，步长是 $0.1\times0.2/\sqrt{0.004}=0.316$——三倍于应有的值，第一步就冲过头。这就是第三行存在的理由。

**第 2 步。** $g_2=2\times0.9=1.8$。

| 量 | 计算 | 值 |
|---|---|---|
| $m_2$ | $0.9\times0.2+0.1\times1.8$ | $0.36$ |
| $v_2$ | $0.999\times0.004+0.001\times3.24$ | $0.007236$ |
| $\hat m_2$ | $0.36/(1-0.81)$ | $1.895$ |
| $\hat v_2$ | $0.007236/(1-0.998001)$ | $3.620$ |
| 步长 | $0.1\times1.895/\sqrt{3.620}$ | $0.0996$ |
| $\theta_2$ | $0.9-0.0996$ | $0.800$ |

两步的步长都是 $\approx\eta=0.1$，虽然梯度从 2 变成了 1.8。这就是 B.3 说的：**Adam 的步长由 $\eta$ 定，和梯度的绝对大小基本无关**。梯度方向一致时 $\hat m/\sqrt{\hat v}\approx\pm1$；方向来回变时 $\hat m$ 被抵消、步长自动变小。

## B.5 AdamW：把 weight decay 拿出来

Weight decay 是让参数每步往 0 缩一点（$\theta\leftarrow\theta-\eta\lambda\theta$），防止过拟合。老写法是把 $\lambda\theta$ 加进梯度里再交给 Adam，但这样它也会被 $\sqrt{\hat v}$ 除一遍，对梯度大的参数几乎不起作用。AdamW[^adamw] 把它拆出来单独做：

$$
\theta_t=\theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\Big)
$$

“W”就是 decoupled **w**eight decay。现在几乎所有 LLM 训练用的都是它。状态和 Adam 完全一样：每个参数一个 $m$、一个 $v$。

## B.6 回到显存

上面每加一个部件都说了“每个参数多存一个数”。汇总：

| 优化器 | 每个可训练参数额外存 | 说明 |
|---|---|---|
| SGD | 0 | 只要当前梯度 |
| 动量 SGD | 1（$m$） | |
| RMSProp | 1（$v$） | |
| **Adam / AdamW** | **2（$m$ 和 $v$）** | 都是 fp32，各 4 字节 |

$m$、$v$ 必须是 fp32：它们每步只变一点点（$v$ 每步加 $0.001\,g^2$），而 bf16 只有两三位有效数字，比当前值的 $0.8\%$ 还小的增量会被舍入掉，$m$、$v$ 就卡住不动了（为什么是 $0.8\%$，见[附录 C](numeric-formats.md)）。这就是 Day 00 §2.4 那张表里 Adam $m$、Adam $v$ 各占 4 字节的来源；加上 fp32 主副本 4 字节和 bf16 的权重、梯度各 2 字节，每个可训练参数 16 字节。**冻结的参数一个 $m$、$v$ 都不需要**——LoRA 省显存的账就是这么算的。

## 自测

1. $\beta_1=0.9$ 时动量“记住”最近多少步？$\beta_2=0.999$ 呢？（提示：$1/(1-\beta)$。）
2. 某参数的梯度稳定在 $10^{-6}$，用 SGD 和 Adam 各走一步，位移分别是多少量级？
3. 为什么偏差修正在训练后期可以忽略？

<!-- 参考文献用脚注 [^key] 写在这里，站点会自动汇总到文末的「参考文献」区 -->

[^polyak]: Polyak, B. T. "Some methods of speeding up the convergence of iteration methods." *USSR Computational Mathematics and Mathematical Physics* 4(5):1–17, 1964. [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5) 深度学习里的用法见 [Sutskever et al., *ICML* 2013](https://proceedings.mlr.press/v28/sutskever13.html)。
[^rmsprop]: Tieleman, T. & Hinton, G. ["Lecture 6.5 — RMSProp."](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) Coursera *Neural Networks for Machine Learning*, 2012.
[^adam]: Kingma, D. P. & Ba, J. "Adam: A Method for Stochastic Optimization." [*ICLR* 2015](https://openreview.net/forum?id=8gmWwjFyLj). [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). 算法见 Algorithm 1，偏差修正的推导见 §3。
[^adamw]: Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." [*ICLR* 2019](https://openreview.net/forum?id=Bkg6RiCqY7). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
