# Day 00 · LoRA 微调一个会用我语气说话的模型

> **Phase** 0 · Quickstart
> **日期** 2026-09-__ · **机器** Jetson AGX Thor · **耗时** ~2h

排在第一天不是因为它简单，是因为**先有一个自己训出来的模型，后面 72 天的理论才有落点**。这天不要求懂原理——照着跑就行，Phase 3（day 25–36）会回来把每一步拆开。

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

### 2.2 LoRA 的参数化：$\Delta W = BA$

微调要学的是改动量 $\Delta W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$。全参微调把它的每个元素都当自由参数，每个矩阵 1678 万个。LoRA 论文 §4.1[^lora] 换了一种写法：

$$
W = W_0 + \Delta W = W_0 + BA,
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r},\quad
A\in\mathbb{R}^{r\times d_{\text{in}}},\quad
r\ll\min(d_{\text{in}},d_{\text{out}})
$$

$W_0$ 冻结不动，只训练 $A$ 和 $B$。前向变成 $\mathbf{y} = W_0\mathbf{x} + B(A\mathbf{x})$：原来那条路照常算，旁边多一条“先把 $\mathbf{x}$ 压到 $r$ 维、再展开回 $d_{\text{out}}$ 维”的支路，两条相加。

**这样写的代价是什么。** 任何能写成 $BA$ 的矩阵，秩都不超过 $r$（$B$ 只有 $r$ 列，见[附录 A.8](../../appendix/linear-algebra.md#a.8-矩阵乘积与列空间的包含)）；反过来任何秩 $\le r$ 的矩阵也都能写成 $BA$（[附录 A.12](../../appendix/linear-algebra.md#a.12-秩分解定理与-svd)有证明）。所以 LoRA 做的事一句话：**把 $\Delta W$ 的搜索范围限制在秩 $\le r$ 的矩阵里。** 不是近似，是限制——它搜不到秩更高的解。

**凭什么这个限制不伤效果。** 这不是定理，是经验：Aghajanyan 等[^aghajanyan] 发现预训练模型微调时“有效的更新方向”远少于参数总数；LoRA 论文 §7 在 GPT-3 175B 上实测 $r=1,2$ 已接近 $r=64$。对一个新任务，$r$ 该取多少要自己测（day 31）。

> [!NOTE]
> 想知道“为什么秩 $\le r$ 就一定能拆成 $BA$”、SVD 怎么给出显式分解、以及“近似低秩”怎么用奇异值曲线量化，看[附录 A.12](../../appendix/linear-algebra.md#a.12-秩分解定理与-svd)。跑通 LoRA 不需要那些；day 31 做 $\Delta W$ 的 SVD 实验时会用到。
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

参数量省 99% 只是表面，**训练时显存的大头不是参数，是每个可训练参数背后跟着的一堆状态。** 先说清这堆状态是什么。

**AdamW 每个参数要存两个动量。** AdamW[^adam] 的更新不是直接用梯度 $g_t$，而是用梯度的两个指数滑动平均（为什么要这么做、每一项什么意思，见[附录 B · 优化器速查](../../appendix/optimizers.md)，有手算例子）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,
\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

$m$ 是梯度的平均（方向），$v$ 是梯度平方的平均（尺度）。它们是**逐参数**的：每个参数都有自己的 $m$ 和 $v$，训练全程保留，所以模型有多少可训练参数，就要多存两倍这么多的数。

**混合精度还要一份 fp32 主副本。** 前向反向用 bf16 算得快，但 bf16 只有 7 位尾数、两三位有效数字（fp32/fp16/bf16 各是什么见[附录 C](../../appendix/numeric-formats.md)），一次更新 $\eta\,m/\sqrt v$ 常常小到加到 bf16 权重上直接被舍入成零。标准做法（ZeRO 论文 §3[^zero]）是另存一份 fp32 的“master weights”，更新在 fp32 上做，再转成 bf16 用于下一轮前向。

把这些加起来，**每个可训练参数**要占：

| 存什么 | 精度 | 字节 |
|---|---|---|
| 权重（前向用） | bf16 | 2 |
| 梯度 | bf16 | 2 |
| fp32 主副本 | fp32 | 4 |
| Adam $m$ | fp32 | 4 |
| Adam $v$ | fp32 | 4 |
| **合计** | | **16** |

而**冻结的参数**只需要第一行：2 字节。梯度不算、动量不存、主副本不要。

对 9B 模型，可训练参数 $N$：

| | 全参微调 $N = 9\times10^9$ | LoRA $N\approx2\times10^7$ |
|---|---|---|
| 冻结参数 × 2 B | 0 | $9\times10^9\times2$ = 18 GB |
| 可训练参数 × 16 B | $9\times10^9\times16$ = **144 GB** | $2\times10^7\times16$ = 0.3 GB |
| **合计（不含激活值）** | **~144 GB** | **~18.3 GB** |

一块 128 GB 统一内存的 Jetson AGX Thor 可用约 115 GB——**全参微调 9B 放不下，LoRA 剩一大半空间给激活值。** 24 GB 独显更是只有 LoRA/QLoRA 一条路。这才是 LoRA 的真正意义：$W_0$ 冻结 $\Rightarrow$ 它不需要梯度、动量、主副本。

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

**环境（Jetson AGX Thor，JetPack 7.0）。** 用现成的 `nvcr.io/nvidia/pytorch:25.08-py3` 容器（教程写的是 25.11，但 33 GB 的新镜像没必要为此多拉），里面 `pip install trl peft datasets accelerate` 得到 transformers 5.16.1 / trl 1.12.0 / peft 0.20.0 / torch 2.8.0（NVIDIA 构建）。pip 会抱怨 cudf 的 pyarrow 版本冲突，与训练无关，忽略。

**模型下载是这天最大的坑，占了一半时间。**

1. `hf-mirror.com` 对这个仓库基本不可用：并行拉分片 30 秒 0 MB，单文件测速 13 B/s。
2. 走代理后 API 通了（HTTP 200）但下载仍然 0 MB/s——原因是 `huggingface_hub` 1.30 默认用 **hf-xet** 传输，它不走 `HTTPS_PROXY`。`curl` 经同一个代理能跑到 7.7 MB/s，说明代理没问题。
3. 解决：`HF_HUB_DISABLE_XET=1`，回退到普通 HTTP 下载，立刻 9 MB/s。18 GB 约半小时。

一句话：**在需要代理的网络里，先 `export HF_HUB_DISABLE_XET=1`**，否则 `snapshot_download` 会静默卡死。

**`docker exec` 喂 heredoc 要加 `-i`**，否则脚本在容器里静默不执行、什么都不输出。

**`assistant_only_loss=True` 会报错。** 它需要 chat template 里有 `{% generation %}` 标记，Qwen3.5 的模板没有。改成 prompt/completion 格式（`prompt` = user 消息，`completion` = assistant 消息），TRL 对这种格式默认 `completion_only_loss=True`，不依赖模板，效果相同。

**不要写死 `target_modules=["q_proj","k_proj","v_proj","o_proj"]`。** 在 meta 设备上按 config 建空模型一查（不用等权重下完），Qwen3.5-9B 是**混合架构**：32 层里 24 层是线性注意力（Gated DeltaNet，模块叫 `in_proj_qkv` / `in_proj_z` / `out_proj`），只有 8 层是标准注意力。只挂 q/k/v/o 只覆盖这 8 层：

| target_modules（r=16） | 可训练参数 | 占比 |
|---|---|---|
| q/k/v/o_proj（仅 8 层全注意力） | 3.9 M | 0.04% |
| + 线性注意力 in_proj_qkv / out_proj | 11.8 M | 0.13% |
| **all-linear**（除 lm_head） | **43.3 M** | **0.48%** |

§2.3 那个“每层 q/k/v/o 各一个 4096×4096”的算例是教学简化；真实模型的 `q_proj` 是 4096→8192（GQA，多头 Q 少头 KV），MLP 是 4096→12288。**先 `from_config` 到 meta 设备看一眼模块名和形状，再决定挂哪里**——这一步 5 秒，能省掉训完发现没学到东西的一小时。

<!-- 训练阶段的坑跑完继续填 -->

## 6. 延伸

- [Jetson AI Lab — Fine-tune LLMs on Jetson](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/)
- LoRA 论文（day 31 会正经读）

**明天要回答的问题**：这个 adapter 到底改了模型的什么？`r=16` 是多少个参数，凭什么够？→ day 31。

<!-- 参考文献用脚注 [^key] 写在这里，站点会自动汇总到文末的「参考文献」区 -->

[^adam]: Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." [*ICLR* 2019](https://openreview.net/forum?id=Bkg6RiCqY7). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)（AdamW；Adam 本身见 Kingma & Ba, *ICLR* 2015, [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)）。
[^zero]: Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." [*SC* 2020](https://doi.org/10.1109/SC41405.2020.00024). [arXiv:1910.02054](https://arxiv.org/abs/1910.02054). §3 的混合精度 Adam 内存账：每参数 16 字节。
[^strang]: Strang, G. [*Introduction to Linear Algebra*](https://math.mit.edu/~gs/linearalgebra/), 5th ed. Wellesley-Cambridge Press, 2016. 行秩等于列秩：§3.5；秩–零化度定理：§3.6；SVD 与秩：§7.1。
[^aghajanyan]: Aghajanyan, A., Zettlemoyer, L. & Gupta, S. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." [*ACL* 2021](https://aclanthology.org/2021.acl-long.568/). [arXiv:2012.13255](https://arxiv.org/abs/2012.13255).
[^lora]: Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." [*ICLR* 2022](https://openreview.net/forum?id=nZeVKeeFYf9). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). 低秩假设的实验证据在 §7。
