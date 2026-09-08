# Day 00 · 用 LoRA 微调一个 9B 模型

> **Phase** 0 · Quickstart
> **日期** 2026-09-__ · **机器** Jetson AGX Thor · **耗时** ~2h

在一块 Jetson 上用 LoRA 微调 Qwen3.5-9B，让它换一套说话方式。全程约两小时，其中训练只占 4 分钟。

你会做这几件事：

- 准备数据集。仓库自带一份 169 条的演示数据
- 读一遍训练脚本：LoRA 挂在这个模型的哪些矩阵上，一段对话怎么变成 token，哪些位置计算 loss
- 运行一次 LoRA 微调
- 用一个可统计的指标验证微调生效
- 与微调后的模型对话，随时切回原模型对比

结束时手上会有：一个 166 MB 的 adapter、一张训练曲线、一组 base 与微调后的对比数字。

§2 是概念，§3 是操作。只想先运行的读者可以从 §3 开始。Phase 3（day 25 到 36）会把训练这条线从头讲一遍。

## 1. 为什么要学这个

微调是把通用模型改成自己需要的样子最直接的手段。读文章能知道 LoRA 是什么，但只有真正训练一次才能得到这些数字：显存涨到多少，loss 降到多少算正常，学习率（learning rate）高一点会发生什么，多少条数据才够。

## 2. 背景

### 2.1 线性层与 `d_in`、`d_out`

Transformer 中绝大部分参数都在**线性层**里。线性层就是一次矩阵乘法：

$$
\mathbf{y} = W\mathbf{x},
\qquad
\mathbf{x}\in\mathbb{R}^{d_{\text{in}}},\quad
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}},\quad
\mathbf{y}\in\mathbb{R}^{d_{\text{out}}}
$$

$\mathbf{x}$ 是一个 token 的向量表示，$W$ 把它从 $d_{\text{in}}$ 维映射到 $d_{\text{out}}$ 维。

以 hidden size $d=4096$ 的模型为例，注意力中的 `q_proj`：

$$
W_q \in \mathbb{R}^{4096\times 4096}
\quad\Longrightarrow\quad
4096^2 = 16\,777\,216 \ \text{个参数}
$$

一个矩阵有 1678 万个参数。一层里有 `q/k/v/o` 四个这样的矩阵，再乘几十层，这就是“9B”的来源。

微调就是修改这些 $W$。记改动量为 $\Delta W$，最终权重是 $W + \Delta W$。

### 2.2 低秩分解 $\Delta W = BA$

微调要学的是改动量 $\Delta W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$。全参微调把它的每个元素都当作自由参数，一个 $4096\times4096$ 的矩阵就是 1678 万个可训练参数。LoRA[^lora] 不直接学 $\Delta W$，而是把它约束成两个小矩阵的乘积（论文 §4.1）：

$$
W = W_0 + \Delta W = W_0 + BA,
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r},\quad
A\in\mathbb{R}^{r\times d_{\text{in}}},\quad
r\ll\min(d_{\text{in}},d_{\text{out}})
$$

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../site_src/assets/fig-lora-arch-dark.svg">
  <img class="fig" alt="LoRA 在一个线性层里的结构" src="../../site_src/assets/fig-lora-arch-light.svg">
</picture>

$W_0$ 冻结不动，只训练 $A$ 和 $B$。前向变成 $\mathbf{y} = W_0\mathbf{x} + \frac{\alpha}{r}B(A\mathbf{x})$。原来的路径照常计算，旁边多一条支路：先把 $\mathbf{x}$ 压到 $r$ 维，再展开回 $d_{\text{out}}$ 维。两路输出都是 $d_{\text{out}}$ 维，因此可以逐元素相加。$\alpha/r$ 是缩放约定，见 §2.5。

| | 参数量 | 训练时每参数占 |
|---|---|---|
| $W_0$ 冻结 | $4096^2$ = 16.8 M | 2 字节（只存权重） |
| $A$ + $B$ 可训练 | $r(d_{\text{in}}+d_{\text{out}})$ = 131 072 | 16 字节（另有梯度、正本、$m$、$v$） |

图中省略了 bias。实际训练时挂载 LoRA 的线性层是可配置的，不是每层都挂。

这个写法的代价如下。全参微调时 $\Delta W$ 的每个元素都能独立取值，训练结束后它可以是任意一个 $4096\times4096$ 的矩阵。LoRA 只更新 $A$ 和 $B$，因此无论这两个小矩阵学成什么样，乘积 $\Delta W$ 的秩都不超过 $r$（$B$ 只有 $r$ 列，见[附录 A.8](../../appendix/linear-algebra.md#a.8-矩阵乘积与列空间的包含)）。反过来，秩不超过 $r$ 的矩阵都能写成 $BA$（[附录 A.12](../../appendix/linear-algebra.md#a.12-秩分解定理与-svd) 有证明）。

也就是说，秩高于 $r$ 的那些 $\Delta W$，LoRA 无法表示，训练再久也到不了。这与“近似”不同。LoRA 不是先算出真正的 $\Delta W$ 再用低秩（low-rank）矩阵逼近它，而是从头到尾只更新 $A$ 和 $B$，高秩的解从未被触及。

这个限制为什么不损害效果，目前没有定理，只有经验证据。Aghajanyan 等[^aghajanyan] 发现预训练模型微调时“有效的更新方向”远少于参数总数。LoRA 论文 §7 在 GPT-3 175B 上实测 $r=1,2$ 已接近 $r=64$。对一个新任务，$r$ 该取多少要自己测量（day 31）。

“秩不超过 $r$ 就一定能拆成 $BA$”的证明、SVD 给出的显式分解、以及用奇异值曲线量化“近似低秩”，见[附录 A.12](../../appendix/linear-algebra.md#a.12-秩分解定理与-svd)。运行 LoRA 不需要这些，day 31 做 $\Delta W$ 的 SVD 实验时会用到。

### 2.3 参数量的节省与临界点

| | 参数量 | $4096\times4096,\ r=16$ |
|---|---|---|
| 全参 $\Delta W$ | $d_{\text{out}}d_{\text{in}}$ | $16\,777\,216$ |
| LoRA $A + B$ | $r(d_{\text{in}} + d_{\text{out}})$ | $16\times 8192 = \mathbf{131\,072}$ |
| 比值 | $\dfrac{r(d_{\text{in}}+d_{\text{out}})}{d_{\text{in}}d_{\text{out}}}$ | $\mathbf{0.78\%}$ |

$r$ 必须小，原因可以从不等式直接解出。要真正节省参数，需要

$$
r\,(d_{\text{in}} + d_{\text{out}}) \;<\; d_{\text{in}}\,d_{\text{out}}
\quad\Longleftrightarrow\quad
r \;<\; \frac{d_{\text{in}}\,d_{\text{out}}}{d_{\text{in}} + d_{\text{out}}}
$$

方阵时 $d_{\text{in}}=d_{\text{out}}=d$，右边就是 $d/2$。$r$ 一旦达到 $d/2$（这里是 2048），LoRA 一个参数都不省。实践中取 $r\in[8,64]$，相对 4096 是 $1/512$ 到 $1/64$。

反过来，$r$ 太小时 $A$、$B$ 容纳不了足够的信息，效果会下降。因此 $r$ 是在“能学到多少”和“占多少显存”之间的取舍，不是越小越好。day 31 用实验找这个平衡点。

### 2.4 优化器状态与显存占用

参数量省 99% 只是表面。训练时显存的大头不是参数，而是每个可训练参数背后的一组状态。下面先说明这组状态是什么。

AdamW 每个参数要存两个动量。AdamW[^adam] 的更新不直接使用梯度 $g_t$，而是使用梯度的两个指数滑动平均（各项含义和手算例子见[附录 B · 优化器速查](../../appendix/optimizers.md)）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,
\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

$m$ 是梯度的平均（方向），$v$ 是梯度平方的平均（尺度）。它们是逐参数的：每个参数都有自己的 $m$ 和 $v$，训练全程保留。因此模型有多少可训练参数，就要多存两倍这么多的数。

混合精度还需要一份 fp32 正本。前向和反向用 bf16 计算速度快，但 bf16 只有 7 位尾数，两三位有效数字（fp32、fp16、bf16 的定义见[附录 C](../../appendix/numeric-formats.md)）。一次更新量 $\eta\,m/\sqrt v$ 常常小到加到 bf16 权重上直接被舍入成零。

因此标准做法（ZeRO 论文[^zero] §3）是同一个参数存两份：

- **fp32 正本**（master weights）：训练期间它是这个参数的权威值，所有更新都累加在它身上，全程不降精度。
- **bf16 工作副本**：从正本复制出的低精度版本，只用于前向和反向，因为 bf16 矩阵乘更快。每更新一步，就从正本重新复制一份覆盖它。

bf16 副本可以随时从正本重新生成，fp32 正本才是被训练的对象。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../site_src/assets/fig-train-step-dark.svg">
  <img class="fig" alt="一步训练里 bf16 工作副本与 fp32 正本各自何时被用到" src="../../site_src/assets/fig-train-step-light.svg">
</picture>

一步训练中这些量各自何时用到。设某个可训练参数是 $\theta$：

1. **前向**：用 $\theta$ 的 bf16 副本参与矩阵乘，算出 loss。bf16 走 Tensor Core，比 fp32 快得多。
2. **反向**：算出梯度 $g$，也是 bf16。
3. **更新**：把 $g$ 转成 fp32，更新 $m$ 和 $v$，算出更新量 $\eta\,\hat m/\sqrt{\hat v}$，加到 $\theta$ 的 fp32 正本上。
4. **同步**：把更新后的 fp32 正本转成 bf16，覆盖第 1 步用的工作副本，供下一轮前向。

第 3 步必须在 fp32 正本上做，因为更新量常常只有 $10^{-4}$ 量级，直接加到 bf16 上会被舍入成零（[附录 C](../../appendix/numeric-formats.md)）。

精度换算：bf16 是 16 位，即 2 字节；fp32 是 32 位，即 4 字节。逐项相加：

| 存什么 | 什么时候用 | 精度 | 每个参数占 |
|---|---|---|---|
| $\theta$ 的 bf16 工作副本 | 步骤 1、2 算前向反向 | bf16 | 2 |
| 梯度 $g$ | 步骤 2 产出，步骤 3 消费 | bf16 | 2 |
| $\theta$ 的 fp32 正本 | 步骤 3 累加更新 | fp32 | 4 |
| Adam $m$ | 步骤 3，全程保留 | fp32 | 4 |
| Adam $v$ | 步骤 3，全程保留 | fp32 | 4 |
| **合计** | | | **16 字节** |

冻结的参数只需要第一行的 2 字节。它不参与步骤 2 到 4：没有梯度，没有动量，不需要 fp32 正本。它不更新，bf16 那一份就是全部。

对 9B 模型，可训练参数 $N$：

| | 全参微调 $N = 9\times10^9$ | LoRA $N\approx2\times10^7$ |
|---|---|---|
| 冻结参数 × 2 B | 0 | $9\times10^9\times2$ = 18 GB |
| 可训练参数 × 16 B | $9\times10^9\times16$ = **144 GB** | $2\times10^7\times16$ = 0.3 GB |
| **合计（不含激活值）** | **~144 GB** | **~18.3 GB** |

一块 128 GB 统一内存的 Jetson AGX Thor 可用约 115 GB。全参微调 9B 放不下，LoRA 则剩下一大半空间给激活值。24 GB 独显只有 LoRA 或 QLoRA 一条路。LoRA 省的就是这一块：90 亿个参数全部冻结，每个只占权重的 2 字节；额外的 14 字节只落在 4300 万个可训练参数上。

> [!CAUTION]
> **LoRA 不减少反向传播的计算量**
>
> 梯度仍然要穿过整个网络才能到达 $A$、$B$，中间激活值照样要保存。因此 `gradient_checkpointing` 仍需开启。LoRA 省的是状态，不是算力。

### 2.5 两个实现细节：初始化与 alpha

初始化必须一零一随机：$A\sim\mathcal{N}(0,\sigma^2)$，$B = 0$。

- 为什么 $B=0$：这样 $\Delta W = BA = \mathbf{0}$，训练开始那一刻模型和原来完全一致，预训练能力不会被随机噪声破坏。
- 为什么不能都取 0：梯度为

    $$
    \frac{\partial \mathcal{L}}{\partial B} = \boldsymbol{\delta}\,(A\mathbf{x})^{\!\top},
    \qquad
    \frac{\partial \mathcal{L}}{\partial A} = B^{\!\top}\boldsymbol{\delta}\,\mathbf{x}^{\!\top}
    $$

    两者会同时为零，参数永远不会更新。取 $B=0$、$A$ 随机时，第一步 $\partial\mathcal{L}/\partial B \neq 0$，$B$ 先动。$B$ 一旦非零，$A$ 也开始收到梯度。

`alpha` 的作用。§2.2 写的 $\Delta W = BA$ 是简化版。实际实现中还会乘一个系数：

$$
\Delta W = \frac{\alpha}{r}\,BA
$$

$r$ 变大时 $BA$ 的数值幅度大致随之变大，除以 $r$ 把尺度稳住。这样改 $r$ 之后不必重新调整学习率（LoRA 论文的原话是 reduces the need to retune，不是完全免除）。$\alpha$ 才是真正的强度旋钮，习惯取 $\alpha = 2r$。本节用 $r=16,\ \alpha=32$。

### 2.6 合并回权重：推理零开销

训练完成后可以把 adapter 合并回去：

$$
W' = W + \frac{\alpha}{r}BA
$$

得到一个和原模型形状完全一样的权重矩阵。因此 $A$、$B$ 只在训练时存在，部署时没有额外的矩阵乘法，也没有额外延迟。adapter 文件只有几十 MB，因为它只包含 $A$ 和 $B$。

### 2.7 SFT 数据的形式

SFT 数据是一组（user 说什么，assistant 回什么）对。关键在于：风格是从 assistant 一侧学到的。因此 response 必须是作者的原文，instruction 只提供上下文。`build_sft.py` 的工作就是反推一个提示词放在原文前面。

另一个细节本节默认开启，day 33 详细讲：`assistant_only_loss=True` 表示只对 assistant 的 token 计算 loss。否则模型会去学那些合成的 instruction 模板。

### 2.8 所需的数据量

这里不是在教新知识（那需要几十万条），只是在改说话风格：用词习惯、句子长度、语气词、起头收尾。风格是很浅的模式。

而且由 §2.3，可训练参数只有约 $2\times10^7$ 个。参数少，需要的数据也少。几百条不至于欠拟合（underfitting），也便于逐条检查。

在 Thor 上，122 GB 统一内存是优势：4B 模型的 LoRA 微调在 24 到 32 GB 独显上要精打细算，在 Thor 上可以放开 batch。受 120 W 功耗墙限制，速度慢，但容量足够。

### 2.9 这个模型的结构

前面的 $d_{\text{in}}$、$d_{\text{out}}$ 一直是抽象符号。这一节把它们落到这个模型上，只回答两个问题：LoRA 挂在哪几个矩阵上，以及为什么加起来正好是 43.3 M 个参数。

下面会出现注意力、GQA、RoPE、GatedDeltaNet 这些名字。本节不需要理解它们，它们是 Phase 1（day 02 到 04）的内容。想现在就弄清 Q、K、V 在做什么，见[附录 D](../../appendix/transformer.md)，那里有一个三个 token、能手算完的例子。本节只需要接受一件事：这个模型是一叠 32 个结构相同的“块”，每个块里有十来个矩阵。LoRA 给其中每个矩阵挂一对小矩阵 $A$、$B$。因此这一节要看的是一个块里有几个矩阵、各自多大。矩阵的名字和它在注意力中的角色，本节可以跳过。

一个块做两件事：先让不同位置的 token 交换信息（这部分叫 mixer），再对每个位置各自做一次变换（这部分叫前馈网络，feed-forward network，简称 FFN）。两件事都靠矩阵乘法完成，而矩阵乘法就是 `nn.Linear`。LoRA 能挂载的位置，就是这些 `nn.Linear`。

下面每个数字都来自模型自己的 `config.json`，用仓库里的工具几秒钟就能复现。它在 `meta` 设备上搭建模型，只建模块不读权重，因此不占显存也不用等待加载：

```bash
python code/peek_model.py --model Qwen/Qwen3.5-9B --layer 3
```

| | 值 |
|---|---|
| decoder 层数 | 32 |
| 隐藏维度 $d_{\text{model}}$ | 4096 |
| FFN 中间维度 | 12288（SwiGLU，因此有 `gate` / `up` / `down` 三个矩阵） |
| 全注意力层 | 每 4 层一个，共 8 层（`full_attention_interval: 4`） |
| 线性注意力层 | 其余 24 层 |
| 全注意力的头 | 16 个查询头 / 4 个 KV 头（GQA），`head_dim` 256 |
| 词表大小 | 248320 |
| 最大位置数 | 262144 |

两种块的区别在本节只影响一件事：它们的矩阵个数和名字不一样（8 个 vs 7 个），数参数时要分开数。它们各自怎么计算，day 03 讲 KV cache 时才需要：

- **全注意力层**（full attention）：标准 Transformer 的形式。每个 token 要和它前面所有 token 计算一次注意力，计算量随序列长度平方增长。推理时要把每个历史 token 的 K、V 保存下来，这就是 KV cache，day 03 计算它的占用。
- **线性注意力层**（linear attention）：把注意力改写成一个随时间递推的状态更新，计算量随长度线性增长，也不需要逐 token 保存 K、V。代价是表达能力比全注意力弱。

Qwen3.5 混合使用两种层：每 3 个线性注意力层配 1 个全注意力层。本节不需要线性注意力的数学，只需要知道两种层里的线性层名字不一样。下一段会看到，这直接决定了 LoRA 的配置。

整条路径如下：一串 token id 进入模型，经过 embedding、32 个块、一次归一化，最后由 `lm_head` 投到词表（vocabulary）上，得到每个词的分数（logits）。logits 经 softmax 成为概率之后，再采样（sampling）出下一个 token。每生成一个 token 都要把这条路径走一遍。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../site_src/assets/fig-qwen-arch-dark.svg">
  <img class="fig" alt="Qwen3.5-9B 的主干：token ids 经 embedding、32 个 decoder 块、RMSNorm、lm_head 得到 logits，再 softmax 成下一个 token 的概率分布" src="../../site_src/assets/fig-qwen-arch-light.svg">
</picture>

再看一个块的内部。两种层的骨架完全一样，都是两段残差：先归一化、过 mixer、把结果加回输入；再归一化、过前馈网络、再加回一次。区别只在中间的 mixer。全注意力块里是 `self_attn`（分组查询注意力，带 RoPE 和 QK-Norm），线性注意力块里是 `linear_attn`（一个叫 GatedDeltaNet 的模块：一维卷积加一个随时间递推的状态）。前馈网络两种块共用同一种：SwiGLU。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../site_src/assets/fig-qwen-block-dark.svg">
  <img class="fig" alt="一个 decoder 块：输入先归一化再过 mixer，结果加回输入；再归一化过 SwiGLU 前馈网络，再加回一次。8 层用 self_attn，24 层用 linear_attn" src="../../site_src/assets/fig-qwen-block-light.svg">
</picture>

图中绿色的名字是 `nn.Linear`，灰色的是归一化、卷积和激活函数。后者没有权重矩阵可拆，LoRA 无法挂载。图在站点上可以点击放大。图中名词的含义如下，完整解释在[附录 D · Transformer 速查](../../appendix/transformer.md)：

- **残差连接**（residual connection）：把一个模块的输出加回它自己的输入（图中的 ⊕）。梯度能沿加法直接回到浅层，几十层的网络才能训练。
- **归一化**（RMSNorm）：把一个向量按它自身的均方根缩放，使后面的层收到的数值范围稳定。放在模块之前的叫 pre-norm，这个模型采用这种方式。
- **mixer**：块里负责让不同位置的 token 交换信息的模块。前馈网络只在每个位置上各算各的，不做位置之间的交换，交换全靠 mixer。
- **分组查询注意力**（GQA）：多个查询头共用一组 K、V。这里 16 个查询头共用 4 组，KV cache 因此只有原来的四分之一（day 03 计算）。
- **RoPE**：把位置信息以旋转的形式写进 Q 和 K，模型才能分辨 token 的先后。
- **QK-Norm**：计算注意力之前先把 Q、K 各归一化一次，训练更稳定。
- **SwiGLU**：前馈网络的一种。两条并行的线性变换，一条过 SiLU 作为“门”，逐元素乘到另一条上，再投回原来的维度。
- **GatedDeltaNet**：线性注意力的一种具体实现，用一个随位置递推更新的状态，代替“每个 token 对所有历史 token 逐一计算注意力”。数学不在此展开。

LoRA 挂载的位置可以逐个数出。`target_modules="all-linear"` 匹配除 `lm_head` 外的每一个 `nn.Linear`，共 248 个：

| 层类型 | 每层的线性层 | 层数 | 每层几个 |
|---|---|---|---|
| 线性注意力 | `in_proj_qkv` `in_proj_z` `in_proj_a` `in_proj_b` `out_proj` + `gate_proj` `up_proj` `down_proj` | 24 | 8 |
| 全注意力 | `q_proj` `k_proj` `v_proj` `o_proj` + `gate_proj` `up_proj` `down_proj` | 8 | 7 |

每个线性层贡献 $r(d_{\text{in}} + d_{\text{out}})$ 个可训练参数（§2.3）。$r = 16$，因此“每个”一列就是 $16\times(d_{\text{in}} + d_{\text{out}})$，逐项相加：

| 模块 | 个数 | $d_{\text{in}} \to d_{\text{out}}$ | 每个 | 小计 |
|---|---|---|---|---|
| `gate_proj` `up_proj` | 64 | 4096 → 12288 | 262 144 | 16 777 216 |
| `down_proj` | 32 | 12288 → 4096 | 262 144 | 8 388 608 |
| `in_proj_qkv` | 24 | 4096 → 8192 | 196 608 | 4 718 592 |
| `in_proj_z` `out_proj` | 48 | 4096 → 4096 | 131 072 | 6 291 456 |
| `in_proj_a` `in_proj_b` | 48 | 4096 → 32 | 66 048 | 3 170 304 |
| `q_proj` | 8 | 4096 → 8192 | 196 608 | 1 572 864 |
| `k_proj` `v_proj` | 16 | 4096 → 1024 | 81 920 | 1 310 720 |
| `o_proj` | 8 | 4096 → 4096 | 131 072 | 1 048 576 |
| **合计** | **248** | | | **43 278 336** |

训练脚本启动时打印的是 `trainable params: 43,278,336`，与上面这一列相加的结果相同。两者是同一个公式的两种算法。

`lm_head` 是唯一被跳过的线性层，因此是 248 而不是 249。这不是本仓库的配置，而是 PEFT 的行为：展开 `all-linear` 时它先收集所有 `nn.Linear`，再把 `model.get_output_embeddings()`（即 `lm_head`）从名单中移除。源码注释写的是 “ignore the last classification head for text generation models”[^peftsrc]。

跳过它的理由不是“它太大”。LoRA 的开销是 $r(d_{\text{in}} + d_{\text{out}})$，与原矩阵大小无关。挂上它也只多 $16 \times (4096 + 248320) \approx 4.0$ M 个参数，占 adapter 的 9%。跳过它的理由是它的功能与其他线性层不同：其余线性层都在 4096 维的隐空间里做变换，而 `lm_head` 把隐状态投到 248320 个词上，直接决定每个词的分数。要改的是说话风格，风格来自中间层的表示；改动输出头则会整体改变模型对所有 token 的打分。这是 PEFT 选择的默认，不是定理。需要改动它时，显式写进 `target_modules` 或 `modules_to_save` 即可。

### 2.10 模型眼里的一段对话

模型不认识“角色”“轮次”这些概念，它收到的只是一串整数。把一段对话变成这串整数的规则叫 **chat template**，每个模型自带一份，存在 tokenizer 里。Qwen 用的是 ChatML 风格。下面的 id 和渲染结果都能复现：

```bash
python code/peek_tokens.py --model Qwen/Qwen3.5-9B
```

- `<|im_start|>`（id 248045）：一轮开始，紧跟角色名（`system` / `user` / `assistant`）
- `<|im_end|>`（id 248046）：一轮结束。它同时是这个模型的 `eos_token`
- `<think>` / `</think>`（id 248068 / 248069）：思考段的边界
- `<|vision_start|>` / `<|image_pad|>`（id 248053 / 248056）：图像输入用。这个模型带一座 27 层的视觉塔，微调用不上
- `<|endoftext|>`（id 248044）：用作 pad

一条用户消息经过模板（`enable_thinking=False`）变成这样：

```text
'<|im_start|>user\n解释一下什么是 KV cache。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

thinking 是模板行为，不是模型的另一个开关。`enable_thinking=True` 时模板只写一个 `<think>\n` 就停止，把思考内容留给模型生成；`False` 时模板直接把 `<think>\n\n</think>\n\n` 这个空思考段补完，模型接着写的就是正式回答：

```text
'<|im_start|>user\n解释一下什么是 KV cache。<|im_end|>\n<|im_start|>assistant\n<think>\n'
```

本节全程用 `enable_thinking=False`：几百条数据教不会推理，而思考段会把生成时间拖长好几倍。

切成 token 之后，上面那段提示是 18 个：

```text
<|im_start|> | user | \n | 解释 | 一下 | 什么是 |  KV |  cache | 。 | <|im_end|> | \n | <|im_start|> | assistant | \n | <think> | \n\n | </think> | \n\n
```

三点说明：

1. `add_generation_prompt=True` 负责在末尾补 `<|im_start|>assistant`。不补，模型不知道轮到自己说话。
2. `\n\n` 是一个 token，不是两个。这一点在 §3.3 会变成一个具体的 bug。
3. 训练时要自己在答案末尾加上 `<|im_end|>`。模型只有在数据里见过结束符，推理时才会停止。不加它，模型会自己接着编下一轮 user 的发言。day 00 第一版就是因此出错的（§5）。

## 3. 动手

### 3.0 进容器（10 min）

训练用的容器叫 `t2t`，建法见 [SETUP](../../SETUP.md#项目容器)。已经建好时：

```bash
sudo docker exec -it -w "$PWD" t2t bash    # 容器内外同路径，进去还在当前目录
cd days/day00_lora-quickstart
bash code/setup_env.sh          # 安装依赖，几分钟。中途不要 Ctrl+C，装一半会留下缺包的环境
```

下面所有命令都在容器里、在这个目录下执行。宿主机上没有 torch/peft/trl，在外面运行会报 `ModuleNotFoundError`。

运行 GPU 任务前先过一遍安全检查（在宿主机上运行，不在容器里）：

```bash
bash common/jetson_preflight.sh   # 任何一项 FAIL 都不要启动
```

### 3.1 准备数据集（2 min）

仓库自带一份演示数据集，任何人 clone 下来都能直接用，不需要提供自己的数据：

```bash
python code/make_demo_dataset.py   # -> data/persona_demo.jsonl，169 条
```

这个文件本身就在仓库里，运行一遍脚本是为了看清它从哪来。脚本读取仓库中的文档，条数会随文档增删小幅变化，结束时会打印实际条数。正文中的 169 条是写作时的值。

它由两部分拼成，都在仓库里、都可复现：

- 本仓库文档中的中文段落（附录 A/B/C、课表、SETUP 等），即真实的技术散文
- `code/seeds.py` 中的中性问答种子，覆盖课表话题和日常闲聊，使模型在聊天时也带着这套语气

然后由 `code/stylize.py` 统一注入一组明确且可统计的风格标记：口癖开头（唔／诶／嗯…）、句尾 `～`、口癖结尾（……大概是这样吧～）。随机种子固定，因此风格的真值已知，训练效果可以直接量化。

先用注入的风格而不是真实语料，原因是真实语料的风格很浅（用词习惯、句子长度），微调后很难判断学没学到，容易自我欺骗。注入一组能数出来的标记，“有没有效果”就变成一个数字。这条链路跑通后，再换成自己的语料（见 §6）。

### 3.2 检查数据（5 min）

```bash
python code/peek.py data/persona_demo.jsonl -n 3
```

这一步不要省略。数据里有什么，模型就学什么；数据里没有的，训一万步也不会有。

### 3.3 训练脚本的四个部分

`code/train_lora.py` 不到 120 行，核心是四段。四个库各管一件事：

| 库 | 负责什么 |
|---|---|
| `transformers` | 加载 tokenizer 和 base 模型，提供 chat template |
| `peft` | 把 $A$、$B$ 插进选中的线性层，冻结其余参数 |
| `trl` | SFT 的训练循环（`SFTTrainer` / `SFTConfig`），是 `transformers.Trainer` 的封装 |
| `datasets` | 把一列 Python dict 变成 Trainer 能迭代的 `Dataset` |

**第一步：把一条对话变成 token 序列，外加一个 loss 掩码。**

先说明训练时在计算什么。模型只做一件事：给它一串 token，预测下一个。因此一条训练样本就是一串 token。模型在每个位置都给出一个“下一个 token 是什么”的预测，与真实的下一个 token 比较，得到这个位置的损失（loss）。整条样本的损失是各位置损失的平均。

但我们不希望它在所有位置上都学。一条样本的前半段是提问，那是 §3.1 用固定模板合成的句子，学它没有意义，还会让模型学会自己提问。要学的是后半段，即助手的回答。因此需要一个和 token 序列等长的 0/1 数组，标出哪些位置计算损失、哪些不计算。这个数组就是**掩码**（mask），脚本里叫 `completion_mask`：回答部分是 1，其余是 0。

构造这个数组只需要知道提问占了前多少个 token。做法是两段分别转换，再首尾相接：

```python
def encode(r):
    prompt_txt = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    ids_p = tok(prompt_txt, add_special_tokens=False)["input_ids"]
    ids_c = tok(r["messages"][-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids  = (ids_p + ids_c)[: a.max_seq]
    mask = ([0] * len(ids_p) + [1] * len(ids_c))[: a.max_seq]
    return {"input_ids": ids, "completion_mask": mask}
```

`messages[:-1]` 是提问部分，先经过 §2.10 的 chat template 变成带 `<|im_start|>` 标记的字符串，再转成 token 序列 `ids_p`。`messages[-1]` 是要学的回答，转成 `ids_c`。两段拼起来是整条样本，`len(ids_p)` 就是边界：掩码前 `len(ids_p)` 个位置写 0，后面写 1。回答末尾要自己加上 `tok.eos_token`（即 §2.10 的 `<|im_end|>`），模型只有在数据里见过结束符，推理时才知道在哪停止。

不使用 TRL 自动查找边界的原因如下。TRL 也能接收一整段对话、自己把提问和回答切开。它的办法是：把提问单独转成 token 得到序列 A，把“提问 + 回答”整段转成 token 得到序列 B，然后假设 A 正好是 B 的前 `len(A)` 个 token（这个关系叫“A 是 B 的前缀”）。假设成立时，边界就是 `len(A)`，和手工拼出来的一样。

这个假设在这个模型上不成立，原因在 §2.10 的 token 切分：分词器会把常见的字符组合并成一个 token，两个换行 `\n\n` 在这个词表里是一个 token，不是两个。而模板在两种场合写出的字符串结尾不一样：

| 模板渲染的是 | 结尾 |
|---|---|
| 只有提问，准备让模型接着写（TRL 这样渲染 A） | `…<think>\n` |
| 提问 + 回答的完整对话（B） | `…<think>\n\n</think>\n\n回答` |

于是 A 的第 15 个 token 是 `\n`，B 的第 15 个 token 是 `\n\n`，前缀假设失效。TRL 找不到边界，日志中出现 `Mismatch between tokenized prompt...`，掩码落到错误的位置上。day 00 第一版的生成结果就是因此出错的（§5）。

手工拼接没有这个问题：边界是数出来的，不依赖任何假设。另外，只要两边都固定 `enable_thinking=False`，前缀假设其实成立，169 条数据中 0 条不满足。可以自行验证：

```bash
python code/peek_tokens.py --model Qwen/Qwen3.5-9B
```

**第二步：告诉 PEFT 把 LoRA 插在哪。**

```python
peft_cfg = LoraConfig(
    r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules="all-linear",
)
```

| 参数 | 含义 |
|---|---|
| `r` | 低秩分解的秩，决定 $A$、$B$ 的形状（§2.2）。取 16 |
| `lora_alpha` | 缩放系数，前向时加的是 $\frac{\alpha}{r}BAx$（§2.5）。习惯取 $2r$，这里是 32 |
| `lora_dropout` | 只作用在 LoRA 旁路上的 dropout，小数据集上防过拟合 |
| `bias` | 是否一起训练 bias。`"none"` 表示不训练，adapter 里只有 $A$、$B$ |
| `task_type` | 告诉 PEFT 这是因果语言模型，保存时才知道该带上哪些层 |
| `target_modules` | 挂载到哪些模块。`"all-linear"` 表示除 `lm_head` 外所有 `nn.Linear`，即 §2.9 数出的 248 个 |

其他教程中这一项通常写成 `["q_proj","k_proj","v_proj","o_proj"]`。在这个模型上照抄会出问题，原因见 §5 第 1 条。

**第三步：训练参数和 Trainer。**

```python
cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs,
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=2,
                learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=3,
                bf16=True, gradient_checkpointing=True,
                max_length=a.max_seq, completion_only_loss=True)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=peft_cfg)
trainer.model.print_trainable_parameters()
trainer.train()
```

几个非默认值的理由：

- `gradient_accumulation_steps=2`：显存只够 batch 4，但想要 batch 8 的梯度，就累积两个 mini-batch 再更新一次。§4 中 43 个 mini-batch 变成 22 个优化步就是这么来的
- `gradient_checkpointing=True`：前向时不保留全部中间激活值，反向时重算。节省显存，代价是约 20% 到 30% 的时间
- `warmup_steps=3`：总共只有 66 步，按比例计算 warmup 已无意义。另外 transformers 5.x 去掉了 `warmup_ratio`
- `completion_only_loss=True`：让 Trainer 使用上面的 `completion_mask`

`peft_config` 传入 `SFTTrainer` 之后，它内部调用 `get_peft_model()` 把模型包起来。这一行打印出第一个数字：

```text
trainable params: 43,278,336 || all params: 8,997,081,600 || trainable%: 0.4810
```

与 §2.9 手算的 43 278 336 一致。

**第四步：保存。** `trainer.save_model()` 只写 adapter，不写 base 模型。目录里有两个关键文件：`adapter_config.json`（挂了哪些层、$r$ 多少）和 `adapter_model.safetensors`（166 MB 的 $A$、$B$）。

166 MB 这个数也能核对：$43\,278\,336 \times 4\ \text{字节} = 173\ \text{MB}$。PEFT 默认按 fp32 存 adapter，而训练时权重是 bf16。

### 3.4 微调（40 到 60 min）

另开一个终端启动遥测和看门狗：

```bash
nohup bash ../../common/jetson_telemetry.sh &        # 日志落在仓库 logs/
nohup bash ../../common/jetson_watchdog.sh 'train_lora' 85 &
```

然后训练：

```bash
python code/train_lora.py \
    --model Qwen/Qwen3.5-9B \
    --data data/persona_demo.jsonl \
    --out private/adapter \
    --epochs 3 --rank 16 --batch 4 --lr 1e-4
```

模型选型和 wheel 版本以 [Jetson AI Lab «Fine-tune LLMs on Jetson»](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/) 为准。那篇给出了 Thor 上验证过的 Full SFT (4B) / LoRA (9B) / QLoRA (27B) 三档配置。本仓库的脚本是通用的 TRL + PEFT 写法，具体版本 pin 见 §5。

### 3.5 对比（20 min）

```bash
python code/compare.py \
    --model Qwen/Qwen3.5-9B --adapter private/adapter \
    --prompts code/prompts.txt \
    --out private/before_after.md
```

### 3.6 度量风格命中率与知识保持（10 min）

```bash
python code/measure_style.py --model Qwen/Qwen3.5-9B --adapter private/adapter --prompts code/prompts.txt
```

它做两件事：

1. **风格命中率**：同一批问题分别用 base 和 adapter 生成，统计风格标记出现的次数。这是 §4 要填的数字。
2. **知识保持**：再问几个与训练语料完全无关的事实问题（`code/probes.txt`），看 adapter 是否还能回答。

第二项是必需的。风格学到了不等于成功。样本少、轮数多时，模型会开始背语料：问它“网易是什么公司”，它拿训练集里的句子来回答。只看风格命中率发现不了这件事。

判定用的是关键词匹配，只是近似。模型答“中华人民共和国的首都”而关键词写的是“中国”，就会被算成没答对。加 `--out results.json` 把每条原文存下来，标 ✗ 的条目先人工检查是真忘了还是换了说法。`probes.txt` 就是这样调出来的。

以上步骤合起来是 `bash code/run_all.sh`，运行完直接进 §3.8。

### 3.7 推理侧：挂载 adapter 与切回 base

`compare.py`、`measure_style.py`、`chat.py` 三个脚本共用 `code/_common.py` 里的三个函数。先是加载：

```python
base  = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                             device_map="cuda").eval()
model = PeftModel.from_pretrained(base, adapter).eval()
```

`PeftModel.from_pretrained` 把 adapter 挂到已经加载好的 base 上，不是再加载一个 9B。内存里始终只有一份 18 GB 的权重，外加 43 M 个 LoRA 参数。

然后是所有对照实验的基础：

```python
with model.disable_adapter():
    answer_base = generate(prompt)
answer_lora = generate(prompt)
```

`disable_adapter()` 是一个上下文管理器，进入时关闭 LoRA 旁路，退出时重新打开。§4 中 base vs adapter 的表就是这样得到的：同一个进程、同一份权重、同一个随机种子，唯一的变量是那 43 M 个参数是否参与前向。`chat.py` 里的 `/base` 和 `/lora` 也是这一行。

最后是把消息变成张量，以及让模型停止：

```python
def render(tok, messages, device):
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   enable_thinking=False, return_tensors="pt",
                                   return_dict=True).to(device)

def stop_ids(tok):
    return list({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
```

两个参数都在 §2.10 讲过：`add_generation_prompt=True` 补上 `<|im_start|>assistant`，`enable_thinking=False` 让模板把空思考段写完。`stop_ids` 要显式传给 `generate(eos_token_id=...)`。不传，模型说完一轮会继续编下一轮的 user 发言。

### 3.8 对话

```bash
python code/chat.py --model Qwen/Qwen3.5-9B --adapter private/adapter
```

流式输出、多轮记忆、已关闭 thinking。`/base` 切到原模型，`/lora` 切回 adapter，同一个问题两边各问一遍最能看出差别。`/reset` 清空对话，`/quit` 退出。

## 4. 结果

Jetson AGX Thor（120 W），Qwen3.5-9B bf16，LoRA r=16、α=32、`all-linear`，cosine + 3 步 warmup，batch 4 × 累积 2，3 epoch、lr 1e-4，即 §3.4 的命令。数据是仓库自带的 169 条。

| | 值 |
|---|---|
| 可训练参数 / 总参数 | 43.3 M / 8.997 B = 0.48% |
| 训练步数 | 66 步（169 条按 batch 4 分成 43 个 mini-batch，累积 2 步更新一次，每 epoch 22 步） |
| 训练耗时 | 4.0 min |
| 峰值内存 | 23.6 GB |
| tj 温度 | 起 38 °C，终 51 °C |
| train loss | 4.58 → 1.52 |
| token 准确率[^acc] | 0.31 → 0.64 |
| adapter 文件 | 166 MB |

对照 §2.4：权重 18 GB + 43 M × 16 B ≈ 0.7 GB，其余约 5 GB 是激活值和 CUDA 工作区。数字与推算一致。

![训练曲线](results/training_curves.png)

### 效果：风格标记命中率

同一批 8 个问题（`code/prompts.txt`），base 与 adapter 各生成一次：

| 标记 | base | + adapter |
|---|---|---|
| 句尾 `～` | 0 / 8 | 8 / 8 |
| 口癖开头（唔／诶／嗯…） | 0 / 8 | 8 / 8 |
| 口癖结尾（……大概是这样吧～） | 0 / 8 | 5 / 8 |

再问 12 个与训练语料完全无关的常识题（`code/probes.txt`）：base 答对 12 / 12，adapter 也是 12 / 12。语气换掉了，知识没有下降。

169 条样本、4 分钟、0.48% 的参数，足以让 9B 模型换一套说话方式。

这不是 prompt 工程，风格完全来自微调后的权重。`measure_style.py` 会先把真正送进模型的完整提示打印出来。base 和 adapter 收到的是逐字节相同的输入：

```
'<|im_start|>user\n解释一下什么是 KV cache。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

没有 system prompt，没有“请用可爱语气回答”这类指令，没有 few-shot 示例。两次生成唯一的差别是 `model.disable_adapter()` 开还是关，即那 43 M 个 LoRA 参数是否加入。`chat.py` 里的 `/base` 和 `/lora` 切换同理，可自行验证。

### 学习率对比

同一份数据、同样 3 epoch，只改学习率各训一次，再用同一批问题测一遍：

| lr | 句尾 `～` | 口癖开头 | 口癖结尾 | 常识 12 题 | 耗时 |
|---|---|---|---|---|---|
| 5e-5 | 7 / 8 | 8 / 8 | 4 / 8 | 11 / 12 | 3.7 min |
| 1e-4（§3.4 的命令） | 8 / 8 | 8 / 8 | 5 / 8 | 12 / 12 | 4.0 min |
| 2e-4 | 8 / 8 | 8 / 8 | 4 / 8 | 12 / 12 | 3.7 min |
| base（不加 adapter） | 0 / 8 | 0 / 8 | 0 / 8 | 12 / 12 | |

三档都把风格学了过去，彼此差一两次命中，8 个问题的样本量支撑不了更强的结论。因此不能由此得出“学习率越大越容易过拟合（overfitting）”。在这份 169 条、风格标记明确的数据上，它没有显现。

换个条件就会显现。语料更少、话题更集中时（比如自己写的一两百条，见 §6），同样 3 epoch、lr 2e-4，模型会开始逐字复述训练集，问一个无关的事实问题也拿语料里的句子来答。这是本仓库作者在自己语料上的一次观察，不是定理。是否过拟合由数据量、数据多样性、可训练参数量、轮数、学习率共同决定，而非学习率单独决定。day 31 取足够多的配置点，把这条曲线扫一遍。

## 5. 踩坑

环境和版本问题都已写进 `code/setup_env.sh` 和 `common/env.sh`，照跑即可。只有两件事值得知道原理：

1. **LoRA 挂在哪要看模型结构，不要照抄 `q_proj,k_proj,v_proj,o_proj`。** Qwen3.5-9B 是混合架构：32 层里 24 层是线性注意力（模块叫 `in_proj_qkv` / `out_proj`），只有 8 层有 `q/k/v/o_proj`。照抄只覆盖 3.9 M（0.04%）参数，`target_modules="all-linear"` 是 43.3 M（0.48%）。查法：`AutoModelForCausalLM.from_config(cfg)` 在 meta 设备上建空模型，列出 `nn.Linear` 的名字和形状，不用等待权重。
2. **只对 assistant 计算 loss 有两条路。** `assistant_only_loss=True` 要求 chat template 带 `{% generation %}` 标记，Qwen3.5 没有。改成 prompt/completion 格式，TRL 默认 `completion_only_loss=True`，不依赖模板。

## 6. 延伸

跑通之后，把演示数据集换成自己的语料，这条支线会一直走到 day 72：

```bash
# 同样在容器里跑
python code/collect_corpus.py --git ~/Code/your-repo --author-email "$(git config user.email)" \
    --markdown ~/notes --out private/corpus.jsonl
python code/build_sft.py --in private/corpus.jsonl --out private/sft.jsonl --min-chars 40
python code/add_batch.py private/paste_*.txt      # 手动粘贴的聊天记录，自动合并连续消息
```

个人语料一律放 `private/`（已 gitignore），不进仓库。

两个链接：

- [LoRA 论文](https://arxiv.org/abs/2106.09685)：§4.1 是 $\Delta W = BA$ 这个写法的出处，§7 是低秩假设的实验证据
- [Jetson AI Lab — Fine-tune LLMs on Jetson](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/)

day 31 要回答的问题：这个 adapter 改了模型的什么？`r=16` 是多少个参数，为什么够？

<!-- 参考文献用脚注 [^key] 写在这里，站点会自动汇总到文末的「参考文献」区 -->

[^acc]: 在计算 loss 的那些位置上，模型概率最高的 token 恰好等于真实下一个 token 的比例。它比 loss 直观，但只看它会漏掉“对得很勉强”的情况，两个一起看。
[^peftsrc]: PEFT 0.20.0 源码 `src/peft/tuners/tuners_utils.py` 的 `_maybe_include_all_linear_layers()`：[GitHub](https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py)。判断依据是 `model.get_output_embeddings()`，注释原文 “ignore the last classification head for text generation models”。
[^lora]: Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." [*ICLR* 2022](https://openreview.net/forum?id=nZeVKeeFYf9). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). §4.1 是 $\Delta W = BA$ 这个写法的出处，§7 是低秩假设的实验证据。
[^aghajanyan]: Aghajanyan, A., Zettlemoyer, L. & Gupta, S. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." [*ACL* 2021](https://aclanthology.org/2021.acl-long.568/). [arXiv:2012.13255](https://arxiv.org/abs/2012.13255).
[^adam]: Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." [*ICLR* 2019](https://openreview.net/forum?id=Bkg6RiCqY7). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)（AdamW；Adam 本身见 Kingma & Ba, [*ICLR* 2015](https://openreview.net/forum?id=8gmWwjFyLj), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)）。详细推导见[附录 B](../../appendix/optimizers.md)。
[^zero]: Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." [*SC* 2020](https://doi.org/10.1109/SC41405.2020.00024). [arXiv:1910.02054](https://arxiv.org/abs/1910.02054). §3 的混合精度 Adam 内存账：每参数 16 字节。
