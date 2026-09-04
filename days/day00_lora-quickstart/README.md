# Day 00 · LoRA 微调一个会用我语气说话的模型

> **Phase** 0 · Quickstart
> **日期** 2026-09-__ · **机器** Jetson AGX Thor · **耗时** ~2h

排在第一天不是因为它简单，是因为**先有一个自己训出来的模型，后面 72 天的理论才有落点**。这天不要求懂原理——照着跑就行，Phase 3（day 25–36）会回来把每一步拆开。

---

## 1. 为什么要学这个

整个仓库的动机是“别只有二手知识”。而“训练”是这里面最容易一直停留在二手的一环——读一百篇讲 LoRA 的文章，不如自己训出一个说话像自己的模型。

这天的目标不是学会什么，是**建立一个“我能训模型”的既成事实**，后面所有理论都挂在这个事实上。

## 2. 背景

### 2.1 `d_in` / `d_out` 是什么

Transformer 里绝大部分参数都在**线性层**里，而线性层就是一次矩阵乘法：

$$
\mathbf{y} = W\mathbf{x},
\qquad
\mathbf{x}\in\mathbb{R}^{d_{\text{in}}},\quad
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}},\quad
\mathbf{y}\in\mathbb{R}^{d_{\text{out}}}
$$

$\mathbf{x}$ 是一个 token 的向量表示，$W$ 把它从 $d_{\text{in}}$ 维映射到 $d_{\text{out}}$ 维。

以 hidden size $d=4096$ 的模型为例，注意力里的 `q_proj`：

$$
W_q \in \mathbb{R}^{4096\times 4096}
\quad\Longrightarrow\quad
4096^2 = 16\,777\,216 \ \text{个参数}
$$

**一个矩阵就 1678 万参数。** 一层里有 `q/k/v/o` 四个，再乘几十层——这就是 “9B” 的来源。

微调就是把这些 $W$ 改一改。记改动量为 $\Delta W$，最终权重是 $W + \Delta W$。

### 2.2 为什么拆成 $A$ 和 $B$：秩

全参微调直接学 $\Delta W$，那是每个矩阵 1678 万个自由参数。LoRA 的做法是把 $\Delta W$ 限制成 $BA$ 的形式。

> [!TIP]
> 下面会用到列空间、秩、基、零空间、秩–零化度定理。很久没碰线性代数的话，先看[附录 A · 线性代数速查](../../appendix/linear-algebra.md)，二十分钟，每个概念都有能手算的例子。要看懂这一步，需要三件事：**秩是什么**（定义）、**为什么秩 $\le r$ 的矩阵一定能写成 $BA$**（定理，可证）、以及**凭什么认为 $\Delta W$ 的秩很低**（经验发现，不是定理）。这三件事的性质不一样，下面分开说。

#### 2.2.1 秩的定义

矩阵 $M\in\mathbb{R}^{m\times n}$ 的**列空间**是它所有列向量张成的子空间：

$$
\operatorname{col}(M) = \{\,M\mathbf{x} : \mathbf{x}\in\mathbb{R}^{n}\,\} \subseteq \mathbb{R}^{m}
$$

**秩**定义为列空间的维数：$\operatorname{rank}(M) = \dim \operatorname{col}(M)$。线性代数的一个基本定理是行秩等于列秩，所以也可以用行空间定义，结果一样[^strang]。

直观上：$M$ 把整个 $\mathbb{R}^n$ 映到 $\mathbb{R}^m$，但**像**只填满了一个 $\operatorname{rank}(M)$ 维的子空间。$\mathbb{R}^{4096\times4096}$ 的矩阵秩最高 4096（像填满整个 $\mathbb{R}^{4096}$）；秩 16 意味着不管输入什么，输出永远落在同一个 16 维子空间里。

#### 2.2.2 定理：秩分解

> [!IMPORTANT]
> **定理（秩分解）**
>
> 对 $M\in\mathbb{R}^{m\times n}$，$\operatorname{rank}(M)\le r$ 当且仅当存在 $B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$ 使 $M = BA$。

**证明。**

（$\Rightarrow$）设 $\operatorname{rank}(M)=k\le r$。取列空间的一组基 $\mathbf{b}_1,\dots,\mathbf{b}_k\in\mathbb{R}^m$，排成 $B_0=[\mathbf{b}_1\ \cdots\ \mathbf{b}_k]\in\mathbb{R}^{m\times k}$。$M$ 的第 $j$ 列 $\mathbf{m}_j$ 属于列空间，所以能唯一地写成基的线性组合：

$$
\mathbf{m}_j=\sum_{i=1}^{k} a_{ij}\,\mathbf{b}_i
\qquad\Longleftrightarrow\qquad
M = B_0 A_0,\quad A_0=(a_{ij})\in\mathbb{R}^{k\times n}
$$

若 $k<r$，给 $B_0$ 补 $r-k$ 个零列、给 $A_0$ 补 $r-k$ 个零行，得到 $B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$，仍有 $M=BA$。

（$\Leftarrow$）若 $M=BA$，则对任意 $\mathbf{x}$，$M\mathbf{x}=B(A\mathbf{x})\in\operatorname{col}(B)$，所以 $\operatorname{col}(M)\subseteq\operatorname{col}(B)$，于是 $\operatorname{rank}(M)\le\dim\operatorname{col}(B)\le r$（$B$ 只有 $r$ 列，列空间维数不可能超过 $r$）。$\blacksquare$

所以「$\Delta W = BA$」**不是近似，是秩 $\le r$ 的等价刻画**：LoRA 把搜索范围从全体 $\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ 缩小到其中秩 $\le r$ 的那部分，而这部分恰好就是所有能写成 $BA$ 的矩阵。分解不唯一（对任意可逆 $G\in\mathbb{R}^{r\times r}$，$(BG)(G^{-1}A)$ 也是一组），但训练只需要存在性。

#### 2.2.3 SVD：分解怎么构造，「近似低秩」是什么意思

上面的证明是存在性的。**奇异值分解**（定义和构造见[附录 A.9–A.10](../../appendix/linear-algebra.md#a.9-长度正交标准正交)）给出一个具体构造：任意 $M$ 可写成 $M=U\Sigma V^{\top}$，其中 $\Sigma$ 的对角线是奇异值 $\sigma_1\ge\sigma_2\ge\cdots\ge 0$，非零奇异值的个数恰好等于 $\operatorname{rank}(M)$[^strang]。取前 $r$ 个：

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

这句话给了「近似低秩」一个可测的定义：**如果奇异值衰减得快，尾部 $\sum_{i>r}\sigma_i^2$ 就小，秩 $r$ 的近似就好。** 所以 LoRA 的假设可以被检验——把全参微调得到的 $\Delta W$ 做 SVD，看奇异值曲线。day 31 会真的做这个实验。

#### 2.2.4 「16 维子空间」到底是哪个空间的子空间

回到 $\Delta W\mathbf{x} = B(A\mathbf{x})$，两端各有一个子空间：

**输出端。** $B\in\mathbb{R}^{d_{\text{out}}\times 16}$ 只有 16 列，其列空间 $\operatorname{col}(B)$ 是 $\mathbb{R}^{d_{\text{out}}}=\mathbb{R}^{4096}$ 里一个维数 $\le 16$ 的子空间。由 2.2.2 的（$\Leftarrow$）方向，对**任何**输入 $\mathbf{x}$，修正量 $\Delta W\mathbf{x}$ 都落在这同一个子空间里。这就是「修正只能落在一个 16 维子空间」的确切含义：**不管输入是什么，模型只能往 16 个固定方向上改输出。**

**输入端。** $A\in\mathbb{R}^{16\times d_{\text{in}}}$ 的零空间 $\operatorname{null}(A)=\{\mathbf{x}:A\mathbf{x}=\mathbf{0}\}$ 维数 $\ge d_{\text{in}}-16 = 4080$（秩–零化度定理[^strang]）。把 $\mathbf{x}$ 分解成行空间分量加零空间分量 $\mathbf{x}=\mathbf{x}_{\parallel}+\mathbf{x}_{\perp}$，则 $A\mathbf{x}=A\mathbf{x}_{\parallel}$：**输入的 4080 个方向被这次微调完全忽略，只有落在 $A$ 的行空间（维数 $\le 16$）里的分量才起作用。** 这就是「只看得见 16 个线性特征」的确切含义。

所以 $r$ 是这次微调的信息瓶颈宽度：输入被压到 $r$ 个数，输出只能在 $r$ 个方向上动。

#### 2.2.5 凭什么认为 $\Delta W$ 的秩很低——这是经验，不是定理

上面说的都是数学事实。但**「微调的改动量 $\Delta W$ 可以用低秩矩阵很好地近似」不是定理**，没有任何证明说它对所有模型、所有任务成立。它是一个经验假设，依据来自两篇论文：

- **Aghajanyan 等 2020**[^aghajanyan]：把预训练模型的微调限制在一个随机的低维子空间里（用随机投影把可训练参数压到 $d$ 维），发现 RoBERTa 在很多任务上 $d$ 只要几百就能达到全参微调 90% 的效果。这说明「有效的更新方向」远少于参数总数。
- **Hu 等 2021（LoRA 论文）**[^lora]：直接把这个想法用到权重矩阵上。§7.2 的实验里，GPT-3 175B 上 $r=1$ 或 $2$ 就已经接近 $r=64$ 的效果；§7.3 分析不同 $r$ 学到的 $A$ 的奇异方向重叠度，发现前几个方向高度一致，说明有用的信息集中在很少的方向上。

> [!CAUTION]
> **LoRA 没有说 $\Delta W$ 真的是低秩的**
>
> 它说的是：把搜索空间限制到秩 $\le r$，在实验里效果没掉多少。这和「全参微调得到的 $\Delta W$ 本身就是低秩的」是两回事——后者在 Hu 等的 §7 里有部分证据，但也有后续工作（如 Lialin 等 2023[^relora]）指出预训练本身的更新不是低秩的，LoRA 的适用范围主要是微调。所以对新任务，$r$ 该取多少要自己测，这就是 day 31 的内容。
### 2.3 参数量：省在哪，什么时候不省

| | 参数量 | $4096\times4096,\ r=16$ |
|---|---|---|
| 全参 $\Delta W$ | $d_{\text{out}}d_{\text{in}}$ | $16\,777\,216$ |
| LoRA $A + B$ | $r(d_{\text{in}} + d_{\text{out}})$ | $16\times 8192 = \mathbf{131\,072}$ |
| 比值 | $\dfrac{r(d_{\text{in}}+d_{\text{out}})}{d_{\text{in}}d_{\text{out}}}$ | $\mathbf{0.78\%}$ |

**$r$ 为什么必须小**——把不等式解出来就知道了。要真的省参数，需要

$$
r\,(d_{\text{in}} + d_{\text{out}}) \;<\; d_{\text{in}}\,d_{\text{out}}
\quad\Longleftrightarrow\quad
r \;<\; \frac{d_{\text{in}}\,d_{\text{out}}}{d_{\text{in}} + d_{\text{out}}}
$$

方阵时 $d_{\text{in}}=d_{\text{out}}=d$，右边就是 $d/2$：

> [!WARNING]
> **临界点**
>
> $r$ 一旦到 $d/2$（这里是 **2048**），LoRA **一个参数都不省**。实践取 $r\in[8,64]$，相对 4096 是 $1/512 \sim 1/64$，才有意义。

反过来 $r$ 太小则瓶颈太窄，压不下足够信息。这是**表达力 vs 成本**的旋钮，不是越小越好——day 31 用实验找它的拐点。

### 2.4 真正省的是优化器状态

参数省 99% 只是表面，**显存大头在训练状态**。

设可训练参数量为 $N$。用 AdamW + bf16 权重 + fp32 优化器状态，每个**可训练**参数要存：

$$
\underbrace{2}_{\text{bf16 梯度}}
+\underbrace{4}_{m}
+\underbrace{4}_{v}
+\underbrace{4}_{\text{fp32 master}}
= 14\ \text{bytes}
$$

冻结的参数这四项**一项都不需要**。对 9B 模型：

| | 全参微调 $N=9\times10^9$ | LoRA $N\approx 2\times10^7$ |
|---|---|---|
| 权重（bf16，都要） | 18 GB | 18 GB |
| 梯度 + Adam $m,v$ + master | $9\times10^9\times14 \approx$ **126 GB** | $2\times10^7\times14\approx$ **0.28 GB** |
| **合计（不含激活）** | **~144 GB** | **~18.3 GB** |

一块 128 GB 统一内存的 Jetson AGX Thor 可用约 115 GB ——**全参微调 9B 直接放不下，LoRA 剩一大半空间留给激活值。** 24 GB 的独显更是只有 LoRA 或 QLoRA 一条路。这才是 LoRA 的真正意义：$W$ 冻结 $\Rightarrow$ 没有梯度、没有动量、没有 master copy。

> [!CAUTION]
> **一个常见误解**
>
> **LoRA 不减少反向传播的计算量。** 梯度仍然要穿过整个网络才能流到 $A$、$B$，中间激活值照样要存。所以 `gradient_checkpointing` 该开还得开。省的是**状态**，不是**算力**。

### 2.5 两个必须知道的实现细节

**初始化必须一零一随机**：$A\sim\mathcal{N}(0,\sigma^2)$，$B = 0$。

- 为什么 $B=0$：这样 $\Delta W = BA = \mathbf{0}$，**训练开始那一刻模型和原来完全一致**，预训练能力不会被随机噪声破坏。
- 为什么不能都取 0：梯度为

    $$
    \frac{\partial \mathcal{L}}{\partial B} = \boldsymbol{\delta}\,(A\mathbf{x})^{\!\top},
    \qquad
    \frac{\partial \mathcal{L}}{\partial A} = B^{\!\top}\boldsymbol{\delta}\,\mathbf{x}^{\!\top}
    $$

    两者会同时为零，**永远学不动**。取 $B=0$、$A$ 随机时，第一步 $\partial\mathcal{L}/\partial B \neq 0$，$B$ 先动；$B$ 一旦非零，$A$ 也开始收到梯度。

**`alpha` 是什么。** 实际用的是缩放版本：

$$
\Delta W = \frac{\alpha}{r}\,BA
$$

$r$ 变大时 $BA$ 的数值幅度大致随之变大，除以 $r$ 把尺度稳住，**这样调 $r$ 时不必重调学习率**。$\alpha$ 才是真正的强度旋钮，习惯取 $\alpha = 2r$（我们用 $r=16,\ \alpha=32$）。

### 2.6 推理时零开销

训完可以把 adapter 合并回去：

$$
W' = W + \frac{\alpha}{r}BA
$$

得到一个和原模型**形状完全一样**的权重矩阵。所以 $A$、$B$ 只在训练时存在，**部署时没有额外矩阵乘法、没有额外延迟**。adapter 文件只有几十 MB，因为它只是 $A$ 和 $B$。

---

### 2.7 SFT 数据长什么样

一堆 `(user 说什么, assistant 回什么)` 对。关键在于：

**风格是从 assistant 那一侧学的。** 所以 response 必须是你的原文，instruction 只是给它上下文。`build_sft.py` 就是反推一个提示词套在原文前面。

还有个细节今天默认开着、day 33 细讲：`assistant_only_loss=True`—— **只对 assistant 的 token 算 loss**，否则模型会去学那些我们瞎编的 instruction 模板。

### 2.8 为什么几百条就够

不是在教新知识（那要几十万条），只是在改**说话风格**：用词习惯、句子长度、语气词、起头收尾。风格是很浅的模式。

而且由 §2.3，可训练参数只有约 $2\times10^7$ 个。**参数少，需要的数据也少**——几百条不至于欠拟合，还正好能让你亲手把每一条看一遍。

> [!TIP]
> **[Thor]**
>
> 122 GB 统一内存在这里是纯优势：4B 模型的 LoRA 微调在 24–32 GB 独显上要精打细算，在 Thor 上可以放开 batch。慢是慢（120 W 功耗墙），但塞得下。

## 3. 动手

### 3.1 攒语料（30 min）

要“越像你越好”的文本：commit message、笔记、给同学解释技术问题的话、论文段落、issue 回复。**不要用你转发的、复制的、AI 生成的**——那些不是你的语气。

```bash
cd days/day00_lora-quickstart

# 从本地若干 git 仓库里抽自己的 commit message
python code/collect_corpus.py \
    --git ~/path/to/repo-a ~/path/to/repo-b \
    --author-email you@example.com \
    --markdown ~/notes \
    --out private/corpus.jsonl
```

> `private/` 已经在 `.gitignore` 里——**个人语料绝不进公开仓库**。

看一眼收了多少、长什么样：

```bash
wc -l private/corpus.jsonl
shuf -n 5 private/corpus.jsonl | python -m json.tool
```

### 3.2 转成 SFT 格式（20 min）

```bash
python code/build_sft.py --in private/corpus.jsonl --out private/sft.jsonl --min-chars 40
```

**这一步要手动看。** 打开 `private/sft.jsonl` 翻二三十条，删掉：纯粹的 `fix typo`、机器生成的、包含密钥或私人信息的、和你风格无关的。**数据质量在这个规模上比数量重要得多。**

### 3.3 微调（40–60 min）

```bash
source ../../common/env.sh
bash ../../common/jetson_preflight.sh        # 任何一项 FAIL 就别起跑
```

另开一个终端起遥测和看门狗：

```bash
nohup bash ../../common/jetson_telemetry.sh ~/telemetry/day00.log &
nohup bash ../../common/jetson_watchdog.sh 'train_lora' 85 &
```

然后训：

```bash
python code/train_lora.py \
    --model <4B 级 instruct 模型> \
    --data private/sft.jsonl \
    --out private/adapter-v1 \
    --epochs 2 --rank 16 --batch 4
```

> **模型选型和 wheel 版本以 [Jetson AI Lab «Fine-tune LLMs on Jetson»](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/)
> 为准**——那篇给了 Thor 上验证过的 Full SFT (4B) / LoRA (9B) / QLoRA (27B) 三档配置。
> 本仓库的脚本是通用 TRL + PEFT 写法，具体版本 pin 见 [踩坑](#5-踩坑)。

### 3.4 对比（20 min）

```bash
python code/compare.py \
    --model <同一个模型> --adapter private/adapter-v1 \
    --prompts code/prompts.txt \
    --out results/before_after.md
```

---

## 4. 结果

<!-- 跑完填。没有数字这天不算完成。 -->

| | base | + LoRA adapter |
|---|---|---|
| 训练数据条数 | — | _ |
| 可训练参数 / 总参数 | — | _ / _ |
| 训练耗时 | — | _ min |
| 峰值 tj 温度 | — | _ °C |
| 峰值内存 | — | _ GB |
| final train loss | — | _ |

同题对比：见 [`results/before_after.md`](results/before_after.md)

一句话结论：_待填_

## 5. 踩坑

<!-- 跑完填。这一节往往最有用。 -->

- 版本 pin：`torch` / `transformers` / `trl` / `peft` 的实际可用版本组合 → _待填_
- _待填_

## 6. 延伸

- [Jetson AI Lab — Fine-tune LLMs on Jetson](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/)
- LoRA 论文（day 31 会正经读）

**明天要回答的问题**：这个 adapter 到底改了模型的什么？`r=16` 是多少个参数，凭什么够？→ day 31。

<!-- 参考文献用脚注 [^key] 写在这里，站点会自动汇总到文末的「参考文献」区 -->

[^strang]: Strang, G. *Introduction to Linear Algebra*, 5th ed. Wellesley-Cambridge Press, 2016. 行秩等于列秩：§3.5；秩–零化度定理：§3.6；SVD 与秩：§7.1。
[^eckart]: Eckart, C. & Young, G. "The approximation of one matrix by another of lower rank." *Psychometrika* 1(3):211–218, 1936. Frobenius 范数下的最优低秩近似；Mirsky 1960 推广到所有酉不变范数。
[^aghajanyan]: Aghajanyan, A., Zettlemoyer, L. & Gupta, S. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL* 2021. arXiv:2012.13255.
[^lora]: Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR* 2022. arXiv:2106.09685. 低秩假设的实验证据在 §7。
[^relora]: Lialin, V. et al. "ReLoRA: High-Rank Training Through Low-Rank Updates." 2023. arXiv:2307.05695. 指出单次低秩更新不足以做预训练，需要多次叠加。
