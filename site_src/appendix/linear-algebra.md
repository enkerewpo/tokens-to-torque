---
title: "附录 A · 线性代数速查：从向量到秩"
---

> 看懂 [Day 00 §2.2](../days/day00.md) 需要的全部线性代数，按依赖顺序排：线性组合 → 张成 → 线性无关 → 基与维数 → 列空间 → 秩 → 零空间 → 正交 → 奇异值。每个概念先给定义，再给一个能手算的例子。教材引用一律指向 Strang[^strang]，章节号标在旁边。

## A.1 向量与线性组合

$\mathbb{R}^n$ 是所有 $n$ 个实数组成的列向量的集合。给定向量 $\mathbf{v}_1,\dots,\mathbf{v}_k\in\mathbb{R}^n$ 和标量 $c_1,\dots,c_k\in\mathbb{R}$，

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k
$$

叫这些向量的一个**线性组合**。整本线性代数只做两件事：加向量、乘标量，线性组合就是这两件事的全部。

**例。** $\mathbf{v}_1 = \begin{pmatrix}1\\0\end{pmatrix}$，$\mathbf{v}_2 = \begin{pmatrix}0\\1\end{pmatrix}$，则 $3\mathbf{v}_1 - 2\mathbf{v}_2 = \begin{pmatrix}3\\-2\end{pmatrix}$。

**矩阵乘向量就是线性组合。** 把 $A\in\mathbb{R}^{m\times n}$ 的列记作 $\mathbf{a}_1,\dots,\mathbf{a}_n\in\mathbb{R}^m$，则对 $\mathbf{x}=(x_1,\dots,x_n)^{\top}$，

$$
A\mathbf{x} = x_1\mathbf{a}_1 + x_2\mathbf{a}_2 + \cdots + x_n\mathbf{a}_n
$$

也就是说，$A\mathbf{x}$ 是 $A$ 的各列以 $\mathbf{x}$ 的分量为系数的线性组合（Strang §2.1）。这一条是后面所有东西的基础，请确认自己能用下面的例子验证：

$$
\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix}
= x_1\begin{pmatrix}1\\3\\5\end{pmatrix} + x_2\begin{pmatrix}2\\4\\6\end{pmatrix}
$$

## A.2 张成（span）与子空间

一组向量的所有线性组合构成的集合叫它们的**张成**：

$$
\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_k\} = \{\,c_1\mathbf{v}_1+\cdots+c_k\mathbf{v}_k : c_i\in\mathbb{R}\,\}
$$

$\mathbb{R}^n$ 的一个子集 $S$ 叫**子空间**，如果它对加法和数乘封闭：$\mathbf{u},\mathbf{v}\in S \Rightarrow \mathbf{u}+\mathbf{v}\in S$，$c\mathbf{v}\in S$。任何张成都是子空间，反过来任何子空间都是某组向量的张成（Strang §3.1）。

**几何直觉（直观上）。** $\mathbb{R}^3$ 里，一个非零向量的张成是过原点的一条直线；两个不共线向量的张成是过原点的一个平面；三个不共面向量的张成是整个 $\mathbb{R}^3$。子空间必过原点，因为 $0\cdot\mathbf{v}=\mathbf{0}$。

**例。** $\mathbf{v}_1=\begin{pmatrix}1\\0\\0\end{pmatrix}$，$\mathbf{v}_2=\begin{pmatrix}0\\1\\0\end{pmatrix}$，$\operatorname{span}\{\mathbf{v}_1,\mathbf{v}_2\}$ 是 $xy$ 平面 $\{(a,b,0)^{\top}\}$。向量 $(1,2,3)^{\top}$ 不在里面，因为第三个分量非零。

## A.3 线性无关

向量组 $\mathbf{v}_1,\dots,\mathbf{v}_k$ 叫**线性无关**，如果

$$
c_1\mathbf{v}_1+\cdots+c_k\mathbf{v}_k=\mathbf{0}\quad\Longrightarrow\quad c_1=\cdots=c_k=0
$$

即只有全零系数才能组合出零向量。否则叫**线性相关**，此时至少有一个向量能写成其余向量的线性组合——它是多余的，去掉它张成不变（Strang §3.4）。

**例。** $\begin{pmatrix}1\\2\end{pmatrix}$ 和 $\begin{pmatrix}2\\4\end{pmatrix}$ 线性相关：$2\cdot\begin{pmatrix}1\\2\end{pmatrix}-1\cdot\begin{pmatrix}2\\4\end{pmatrix}=\mathbf{0}$，系数不全为零。它们的张成只是一条直线，虽然有两个向量。

**例。** $\begin{pmatrix}1\\0\end{pmatrix}$ 和 $\begin{pmatrix}1\\1\end{pmatrix}$ 线性无关：$c_1\begin{pmatrix}1\\0\end{pmatrix}+c_2\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}c_1+c_2\\c_2\end{pmatrix}=\mathbf{0}$ 迫使 $c_2=0$ 进而 $c_1=0$。

## A.4 基与维数

子空间 $S$ 的一组**基**是 $S$ 里一组线性无关、且张成恰好等于 $S$ 的向量。两个条件缺一不可：线性无关保证没有多余的，张成等于 $S$ 保证没有缺的。

::: {.callout-important title="定理（维数良定义）"}
同一个子空间的任意两组基包含的向量个数相同。这个数叫 $S$ 的**维数** $\dim S$。（Strang §3.5）
:::

有了基，$S$ 里每个向量都能**唯一地**写成基的线性组合——唯一性正是线性无关给的：若 $\sum c_i\mathbf{b}_i=\sum c_i'\mathbf{b}_i$，则 $\sum(c_i-c_i')\mathbf{b}_i=\mathbf{0}$，由线性无关得 $c_i=c_i'$。Day 00 §2.2.2 证明里“唯一地写成基的线性组合”用的就是这一条。

**例。** $\mathbb{R}^3$ 的标准基是 $\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3$，所以 $\dim\mathbb{R}^3=3$。$xy$ 平面的一组基是 $\{\mathbf{e}_1,\mathbf{e}_2\}$，另一组基是 $\{(1,1,0)^{\top},(1,-1,0)^{\top}\}$，都是两个向量，所以 $xy$ 平面维数是 2。

## A.5 列空间与秩

矩阵 $A\in\mathbb{R}^{m\times n}$ 的**列空间**是它各列的张成：

$$
\operatorname{col}(A)=\operatorname{span}\{\mathbf{a}_1,\dots,\mathbf{a}_n\}=\{\,A\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\,\}\subseteq\mathbb{R}^m
$$

第二个等号就是 A.1 的“矩阵乘向量是列的线性组合”。所以 $\operatorname{col}(A)$ 是**$A$ 作为一个映射所有可能的输出**——不管输入什么，$A\mathbf{x}$ 永远落在这个子空间里。

**秩**是列空间的维数：

$$
\operatorname{rank}(A)=\dim\operatorname{col}(A)
$$

等价地，秩是 $A$ 的列中线性无关的最大个数（Strang §3.5）。

**例。** $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$，两列 $(1,2)^{\top}$ 和 $(2,4)^{\top}$ 线性相关（A.3 的例子），列空间是直线 $\operatorname{span}\{(1,2)^{\top}\}$，$\operatorname{rank}(A)=1$。虽然 $A$ 是 $2\times2$ 的，它的输出永远在一条直线上。

**例。** $B=\begin{pmatrix}1&0\\0&1\end{pmatrix}$ 两列线性无关，$\operatorname{col}(B)=\mathbb{R}^2$，$\operatorname{rank}(B)=2$。

::: {.callout-note title="秩的上界"}
$\operatorname{rank}(A)\le\min(m,n)$。列空间是 $\mathbb{R}^m$ 的子空间所以维数 $\le m$；它由 $n$ 个向量张成所以维数 $\le n$。Day 00 里 $B\in\mathbb{R}^{4096\times16}$ 的秩 $\le16$ 就是这一条。
:::

## A.6 行秩等于列秩

$A$ 的**行空间**是各行（视为 $\mathbb{R}^n$ 中的向量）的张成，即 $\operatorname{col}(A^{\top})$。

::: {.callout-important title="定理"}
$\dim\operatorname{col}(A)=\dim\operatorname{col}(A^{\top})$。即行空间和列空间维数相同，都等于秩。（Strang §3.5）
:::

证明思路：对 $A$ 做行变换化成行阶梯形 $R$。行变换不改变行空间（每一行都是原来行的线性组合，且可逆）；行变换会改变列空间，但**不改变列之间的线性关系**（$A\mathbf{x}=\mathbf{0}\Leftrightarrow R\mathbf{x}=\mathbf{0}$），所以不改变列中线性无关的最大个数。而 $R$ 里主元的个数既是无关行数也是无关列数。完整证明见 Strang §3.5。

## A.7 零空间与秩–零化度定理

$A$ 的**零空间**是被映到零的所有输入：

$$
\operatorname{null}(A)=\{\,\mathbf{x}\in\mathbb{R}^n : A\mathbf{x}=\mathbf{0}\,\}\subseteq\mathbb{R}^n
$$

它是子空间（两个解相加还是解，解乘标量还是解）。

::: {.callout-important title="定理（秩–零化度）"}
对 $A\in\mathbb{R}^{m\times n}$，$\operatorname{rank}(A)+\dim\operatorname{null}(A)=n$。（Strang §3.6）
:::

直观上：输入空间 $\mathbb{R}^n$ 有 $n$ 个维度，其中 $\dim\operatorname{null}(A)$ 个维度被 $A$ 压没了（映到零），剩下的 $\operatorname{rank}(A)$ 个维度活着走到了输出端。

**例。** $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$，$\operatorname{rank}=1$，所以 $\dim\operatorname{null}(A)=2-1=1$。验证：$A\mathbf{x}=\mathbf{0}$ 即 $x_1+2x_2=0$，解集是直线 $\operatorname{span}\{(2,-1)^{\top}\}$，确实一维。

**在 Day 00 里的用法。** $A\in\mathbb{R}^{16\times4096}$，$\operatorname{rank}(A)\le16$，所以 $\dim\operatorname{null}(A)\ge4096-16=4080$。输入 $\mathbf{x}$ 里落在零空间的分量被 $A$ 直接扔掉——这就是“4080 个方向被忽略”的来源。

## A.8 矩阵乘积与列空间的包含

对 $B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$，乘积 $BA$ 的列空间满足

$$
\operatorname{col}(BA)\subseteq\operatorname{col}(B)
$$

**证明。** 任取 $\mathbf{y}\in\operatorname{col}(BA)$，则 $\mathbf{y}=(BA)\mathbf{x}=B(A\mathbf{x})$，令 $\mathbf{z}=A\mathbf{x}\in\mathbb{R}^r$，则 $\mathbf{y}=B\mathbf{z}\in\operatorname{col}(B)$。$\blacksquare$

推论：$\operatorname{rank}(BA)\le\operatorname{rank}(B)\le r$。这就是 Day 00 §2.2.2 中（$\Leftarrow$）方向的全部内容：只要能写成 $BA$ 且 $B$ 只有 $r$ 列，秩就不可能超过 $r$。

## A.9 长度、正交、标准正交

前面的概念都只用了加法和数乘。从这里开始多用一样东西：**长度**，它由内积定义。

两个向量的**内积**是 $\mathbf{u}^{\top}\mathbf{v}=\sum_i u_iv_i$。向量的**长度**是 $\|\mathbf{x}\|=\sqrt{\mathbf{x}^{\top}\mathbf{x}}=\sqrt{\sum_i x_i^2}$，就是勾股定理推广到 $n$ 维。两个向量**正交**指内积为零：$\mathbf{u}^{\top}\mathbf{v}=0$，直观上就是互相垂直（Strang §1.2）。

**例。** $(1,1)^{\top}$ 和 $(1,-1)^{\top}$ 的内积是 $1\cdot1+1\cdot(-1)=0$，正交。$(1,1)^{\top}$ 的长度是 $\sqrt2$。

一组向量 $\mathbf{q}_1,\dots,\mathbf{q}_k$ 叫**标准正交**（orthonormal），如果两两正交且每个长度为 1：

$$
\mathbf{q}_i^{\top}\mathbf{q}_j=\begin{cases}1 & i=j\\ 0 & i\ne j\end{cases}
$$

“标准”指长度归一，“正交”指两两垂直。$\mathbb{R}^n$ 的标准基 $\mathbf{e}_1,\dots,\mathbf{e}_n$ 就是一组标准正交向量；把它们整体旋转一下，还是标准正交。

**例。** $\mathbf{q}_1=\frac{1}{\sqrt2}(1,1)^{\top}$，$\mathbf{q}_2=\frac{1}{\sqrt2}(1,-1)^{\top}$。检查：$\mathbf{q}_1^{\top}\mathbf{q}_1=\frac12(1+1)=1$，$\mathbf{q}_1^{\top}\mathbf{q}_2=\frac12(1-1)=0$。它们就是标准基逆时针转 $45°$。

把标准正交向量排成矩阵 $Q=[\mathbf{q}_1\ \cdots\ \mathbf{q}_k]\in\mathbb{R}^{n\times k}$，上面的定义可以一次写完：

$$
Q^{\top}Q=I_k
$$

因为 $(Q^{\top}Q)_{ij}=\mathbf{q}_i^{\top}\mathbf{q}_j$。若 $Q$ 是方阵（$k=n$），$Q^{\top}Q=I$ 意味着 $Q^{\top}$ 就是 $Q$ 的逆：

::: {.callout-important title="标准正交方阵的两个性质"}
（Strang §4.4）

1. $Q^{-1}=Q^{\top}$。求逆不用算，转置就行。
2. $Q$ 不改变长度和夹角：$\|Q\mathbf{x}\|^2=(Q\mathbf{x})^{\top}(Q\mathbf{x})=\mathbf{x}^{\top}Q^{\top}Q\mathbf{x}=\mathbf{x}^{\top}\mathbf{x}=\|\mathbf{x}\|^2$，同理 $(Q\mathbf{u})^{\top}(Q\mathbf{v})=\mathbf{u}^{\top}\mathbf{v}$。
:::

不改变长度和夹角的线性映射，直观上就是**旋转**（可能再加一次镜像）。所以下一节里“$U$、$V$ 的列标准正交”读成“$U$、$V$ 是旋转”就对了。也正因为 $U^{-1}=U^{\top}$、$V^{-1}=V^{\top}$ 都可逆，SVD 里 $\operatorname{rank}(A)=\operatorname{rank}(\Sigma)$ 才成立。

## A.10 奇异值与 SVD

**先看对角矩阵。** $\Sigma=\begin{pmatrix}3&0\\0&1\end{pmatrix}$ 把 $\mathbf{e}_1$ 拉长 3 倍、把 $\mathbf{e}_2$ 保持不变：单位圆被映成一个半轴长 3 和 1 的椭圆。这两个数就是 $\Sigma$ 的奇异值。对角矩阵一眼能看出来，一般矩阵需要先“转正”。

::: {.callout-important title="定理（奇异值分解，SVD）"}
任意 $A\in\mathbb{R}^{m\times n}$ 都可以写成 $A=U\Sigma V^{\top}$，其中 $U\in\mathbb{R}^{m\times m}$、$V\in\mathbb{R}^{n\times n}$ 的列都是标准正交的（A.9，即 $U^{\top}U=I$、$V^{\top}V=I$），$\Sigma\in\mathbb{R}^{m\times n}$ 只有对角线非零，且 $\sigma_1\ge\sigma_2\ge\cdots\ge0$。对角线上的 $\sigma_i$ 叫 $A$ 的**奇异值**。（Strang §7.1–7.2）
:::

读法：$V^{\top}$ 先把输入旋转到一组合适的正交方向上，$\Sigma$ 在每个方向上独立拉伸（拉伸倍数就是 $\sigma_i$），$U$ 再把结果旋转到输出空间。所以**任何线性映射都是“旋转 → 各方向拉伸 → 旋转”**，奇异值就是各方向的拉伸倍数。直观上：$A$ 把单位球映成一个椭球，$\sigma_i$ 是椭球的半轴长。

**奇异值从哪来。** $A^{\top}A$ 是对称矩阵，且对任意 $\mathbf{x}$ 有 $\mathbf{x}^{\top}A^{\top}A\mathbf{x}=\|A\mathbf{x}\|^2\ge0$，所以它的特征值 $\lambda_i\ge0$。定义 $\sigma_i=\sqrt{\lambda_i}$，$V$ 的列取 $A^{\top}A$ 的标准正交特征向量，$U$ 的列取 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$（对 $\sigma_i>0$）。这就是 SVD 的构造，验证 $A\mathbf{v}_i=\sigma_i\mathbf{u}_i$ 即得 $AV=U\Sigma$（Strang §7.2）。

**例。** $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$（A.5 里秩为 1 的那个）。$A^{\top}A=\begin{pmatrix}5&10\\10&20\end{pmatrix}$，特征值 $\lambda_1=25$、$\lambda_2=0$，所以奇异值 $\sigma_1=5$、$\sigma_2=0$。只有一个非零奇异值，对应秩 1：$A$ 把整个平面压到一条直线上，另一个方向被压成零。

::: {.callout-note title="非零奇异值的个数等于秩"}
$U$、$V$ 可逆（A.9：标准正交方阵的逆是转置），所以 $\operatorname{rank}(A)=\operatorname{rank}(\Sigma)$，而对角矩阵的秩就是非零对角元的个数。
:::

**为什么它是“低秩近似”的正确工具。** 把 $A=U\Sigma V^{\top}$ 按列展开，得到

$$
A=\sum_{i=1}^{\operatorname{rank}(A)}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^{\top}
$$

每一项 $\mathbf{u}_i\mathbf{v}_i^{\top}$ 都是一个秩 1 矩阵，权重是 $\sigma_i$，且按大小排好了序。只保留前 $r$ 项就得到一个秩 $r$ 的矩阵 $A_r$，丢掉的部分大小恰好是 $\sqrt{\sum_{i>r}\sigma_i^2}$（Frobenius 范数）。Eckart–Young–Mirsky 定理说这已经是所有秩 $\le r$ 矩阵里最好的了[^eckart]。所以看一个矩阵能不能被低秩近似，只要画它的奇异值曲线：**衰减快 → 可以，尾巴长 → 不行。** Day 00 §2.2.3 和 day 31 用的就是这个判据。

## A.11 一张总表

| 概念 | 定义 | 是什么空间的子集 | Day 00 里的角色 |
|---|---|---|---|
| 线性组合 | $\sum c_i\mathbf{v}_i$ | — | $A\mathbf{x}$ 是 $A$ 各列的线性组合 |
| 张成 | 所有线性组合的集合 | 输入/输出空间 | 定义列空间 |
| 线性无关 | 只有零系数组合出零 | — | 保证基的表示唯一 |
| 基 | 线性无关且张成整个子空间 | — | 秩分解证明里取列空间的基 |
| 维数 | 基的向量个数 | — | 秩就是列空间的维数 |
| 列空间 $\operatorname{col}(A)$ | 所有输出 $A\mathbf{x}$ | $\mathbb{R}^m$ | $\operatorname{col}(B)$ 是修正量能到的 16 维子空间 |
| 秩 | $\dim\operatorname{col}(A)$ | — | $r$ |
| 零空间 $\operatorname{null}(A)$ | 所有 $A\mathbf{x}=\mathbf{0}$ 的输入 | $\mathbb{R}^n$ | 被忽略的 4080 个输入方向 |
| 秩–零化度 | $\operatorname{rank}+\dim\operatorname{null}=n$ | — | 算出 4080 |
| 标准正交 | 两两内积 0、长度 1；$Q^{\top}Q=I$，方阵时 $Q^{-1}=Q^{\top}$ | — | SVD 里的 $U$、$V$ 是“旋转” |
| 奇异值 $\sigma_i$ | $A$ 在各正交方向上的拉伸倍数，$\sqrt{\lambda_i(A^{\top}A)}$ | — | 非零个数 = 秩；衰减快 ⇒ 可低秩近似 |

## A.12 秩分解定理与 SVD

Day 00 §2.2 的选读，跑通 LoRA 不需要这一节。它回答三个问题：为什么秩 $\le r$ 就一定能写成 $BA$；SVD 怎么给出一组显式的 $B$、$A$；“近似低秩”怎么用奇异值曲线量化。

### A.12.1 秩分解定理

::: {.callout-important title="定理（秩分解）"}
对 $M\in\mathbb{R}^{m\times n}$，$\operatorname{rank}(M)\le r$ 当且仅当存在 $B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$ 使 $M = BA$。
:::

**证明。**

（$\Rightarrow$）设 $\operatorname{rank}(M)=k\le r$。取列空间的一组基 $\mathbf{b}_1,\dots,\mathbf{b}_k\in\mathbb{R}^m$，排成 $B_0=[\mathbf{b}_1\ \cdots\ \mathbf{b}_k]\in\mathbb{R}^{m\times k}$。$M$ 的第 $j$ 列 $\mathbf{m}_j$ 属于列空间，所以能唯一地写成基的线性组合：

$$
\mathbf{m}_j=\sum_{i=1}^{k} a_{ij}\,\mathbf{b}_i
\qquad\Longleftrightarrow\qquad
M = B_0 A_0,\quad A_0=(a_{ij})\in\mathbb{R}^{k\times n}
$$

若 $k<r$，给 $B_0$ 补 $r-k$ 个零列、给 $A_0$ 补 $r-k$ 个零行，得到 $B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$，仍有 $M=BA$。

（$\Leftarrow$）若 $M=BA$，则对任意 $\mathbf{x}$，$M\mathbf{x}=B(A\mathbf{x})\in\operatorname{col}(B)$，所以 $\operatorname{col}(M)\subseteq\operatorname{col}(B)$，于是 $\operatorname{rank}(M)\le\dim\operatorname{col}(B)\le r$（$B$ 只有 $r$ 列，列空间维数不可能超过 $r$）。$\blacksquare$

所以 Day 00 §2.2 那句“任何秩 $\le r$ 的矩阵都能写成 $BA$”就是这个定理。分解不唯一（对任意可逆 $G\in\mathbb{R}^{r\times r}$，$(BG)(G^{-1}A)$ 也是一组），但训练只需要存在性。

### A.12.2 SVD 给出显式分解；「近似低秩」怎么量化

上面的证明是存在性的。**奇异值分解**（A.9–A.10）给出一个具体构造：任意 $M$ 可写成 $M=U\Sigma V^{\top}$，其中 $\Sigma$ 的对角线是奇异值 $\sigma_1\ge\sigma_2\ge\cdots\ge 0$，非零奇异值的个数恰好等于 $\operatorname{rank}(M)$[^strang]。取前 $r$ 个：

$$
M_r = U_r\,\Sigma_r\,V_r^{\top},
\qquad
B = U_r\Sigma_r\in\mathbb{R}^{m\times r},\quad
A = V_r^{\top}\in\mathbb{R}^{r\times n}
$$

若 $\operatorname{rank}(M)\le r$，则 $M_r = M$，这就是一组显式的 $BA$。

更重要的是 $\operatorname{rank}(M)>r$ 的情况。**Eckart–Young–Mirsky 定理**[^eckart]说：在所有秩 $\le r$ 的矩阵里，$M_r$ 是离 $M$ 最近的那个，且

$$
\min_{\operatorname{rank}(X)\le r}\ \|M-X\|_F \;=\; \|M-M_r\|_F \;=\; \sqrt{\textstyle\sum_{i>r}\sigma_i^2}
$$

这句话给了「近似低秩」一个可测的定义：**如果奇异值衰减得快，尾部 $\sum_{i>r}\sigma_i^2$ 就小，秩 $r$ 的近似就好。** 所以 LoRA 的假设可以被检验——把全参微调得到的 $\Delta W$ 做 SVD，看奇异值曲线。Day 31 会真的做这个实验。

### A.12.3 $BA$ 的几何：修正量落在哪

这一小节回答“$r=16$ 到底限制了什么”。注意：**下面说的都只关于修正支路 $\Delta W = BA$；主路 $W_0\mathbf{x}$ 照常看全部输入、写全部输出，LoRA 没有让模型忽略任何方向。** 回到 $\Delta W\mathbf{x} = B(A\mathbf{x})$，两端各有一个子空间：

**输出端。** $B\in\mathbb{R}^{d_{\text{out}}\times 16}$ 只有 16 列，其列空间 $\operatorname{col}(B)$ 是 $\mathbb{R}^{d_{\text{out}}}=\mathbb{R}^{4096}$ 里一个维数 $\le 16$ 的子空间。由 2.2.2 的（$\Leftarrow$）方向，对**任何**输入 $\mathbf{x}$，修正量 $\Delta W\mathbf{x}$ 都落在这同一个子空间里。即：**不管输入是什么，修正支路只能往 16 个固定方向上改输出。**

**输入端。** $A\in\mathbb{R}^{16\times d_{\text{in}}}$ 的零空间 $\operatorname{null}(A)=\{\mathbf{x}:A\mathbf{x}=\mathbf{0}\}$ 维数 $\ge d_{\text{in}}-16 = 4080$（秩–零化度定理[^strang]）。把 $\mathbf{x}$ 分解成行空间分量加零空间分量 $\mathbf{x}=\mathbf{x}_{\parallel}+\mathbf{x}_{\perp}$，则 $A\mathbf{x}=A\mathbf{x}_{\parallel}$：**修正支路对输入的 4080 个方向没有反应，只有落在 $A$ 的行空间（维数 $\le 16$）里的分量会产生修正。**

所以 $r$ 是修正支路的宽度：它从输入里读 $r$ 个数，往输出里写 $r$ 个方向。

## 自测

做完能答出这三题，§2.2 就没有障碍了：

1. $A=\begin{pmatrix}1&1&2\\0&1&1\\1&0&1\end{pmatrix}$ 的秩是多少？给出列空间的一组基。（提示：第三列等于前两列之和。）
2. 若 $\operatorname{rank}(A)=r$，$A\in\mathbb{R}^{m\times n}$，零空间维数是多少？
3. 为什么 $B\in\mathbb{R}^{4096\times16}$ 不管乘什么，输出都在同一个至多 16 维的子空间里？用 A.5 和 A.8 各说一遍。
4. $\begin{pmatrix}3&0\\0&0\end{pmatrix}$ 的奇异值是什么？秩是多少？它把单位圆映成什么？

<!-- 参考文献用脚注 [^key] 写在这里，站点会自动汇总到文末的「参考文献」区 -->

[^eckart]: Eckart, C. & Young, G. "The approximation of one matrix by another of lower rank." *Psychometrika* 1(3):211–218, 1936.
[^lora]: Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR* 2022. arXiv:2106.09685.
[^strang]: Strang, G. *Introduction to Linear Algebra*, 5th ed. Wellesley-Cambridge Press, 2016. 线性组合与 $A\mathbf{x}$：§2.1；子空间与张成：§3.1；线性无关：§3.4；基、维数、秩、行秩=列秩：§3.5；零空间与秩–零化度：§3.6；内积与长度：§1.2；标准正交矩阵：§4.4；SVD：§7.1–7.2。中文可用清华大学出版社影印版；MIT OCW 18.06 是配套公开课。
