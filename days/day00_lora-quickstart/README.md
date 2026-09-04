# Day 00 · 用 LoRA 微调一个 9B 模型

> **Phase** 0 · Quickstart
> **日期** 2026-09-__ · **机器** Jetson AGX Thor · **耗时** ~2h

这一天在一块 Jetson 上用 LoRA 微调 Qwen3.5-9B，让它换一套说话方式。全程约两小时，其中训练只占 3 分钟。

你会做这几件事：

- 准备数据集——仓库自带一份 137 条的演示数据，不需要你提供任何东西
- 跑一次 LoRA 微调
- 用一个能统计的指标验证微调确实生效，而不是靠感觉
- 和微调后的模型对话，随时切回原模型对比

跑完手上会有：一个 166 MB 的 adapter、一张训练曲线、一组 base 与微调后的对比数字。

**不需要先懂原理。** §2 解释了会用到的概念，但你也可以直接跳到 §3 开跑，回头再看。Phase 3（day 25–36）会把训练这条线从头拆一遍。

## 1. 为什么要学这个

微调是把通用模型改成自己需要的样子最直接的手段。读文章能知道 LoRA 是什么，但只有真训一次才知道：显存会涨到多少、loss 降到多少算正常、学习率高一点会发生什么、多少条数据才够。这些数字不自己撞一遍是记不住的。

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

### 2.2 LoRA 怎么表示 $\Delta W$：拆成 $BA$

微调要学的是改动量 $\Delta W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$。全参微调把它的每个元素都当自由参数，一个 $4096\times4096$ 的矩阵就是 1678 万个可训练参数。LoRA[^lora] 不直接学 $\Delta W$，而是把它约束成两个小矩阵的乘积（论文 §4.1）：

$$
W = W_0 + \Delta W = W_0 + BA,
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r},\quad
A\in\mathbb{R}^{r\times d_{\text{in}}},\quad
r\ll\min(d_{\text{in}},d_{\text{out}})
$$

![](../../site_src/assets/fig-lora-arch-light.svg){.fig .light-content fig-alt="LoRA 在一个线性层里的结构"}
![](../../site_src/assets/fig-lora-arch-dark.svg){.fig .dark-content fig-alt="LoRA 在一个线性层里的结构"}

$W_0$ 冻结不动，只训练 $A$ 和 $B$。前向变成 $\mathbf{y} = W_0\mathbf{x} + \frac{\alpha}{r}B(A\mathbf{x})$：原来那条路照常算，旁边多一条“先把 $\mathbf{x}$ 压到 $r$ 维、再展开回 $d_{\text{out}}$ 维”的支路，两条相加。两路输出都是 $d_{\text{out}}$ 维，所以能逐元素相加。$\alpha/r$ 是缩放约定，§2.5 会讲。

| | 参数量 | 训练时每参数占 |
|---|---|---|
| $W_0$ 冻结 | $4096^2$ = 16.8 M | 2 字节（只存权重） |
| $A$ + $B$ 可训练 | $r(d_{\text{in}}+d_{\text{out}})$ = 131 072 | 16 字节（还要梯度、正本、$m$、$v$） |

图中省略了 bias；实际训练时挑哪些线性层挂 LoRA 是可配的，不是每层都挂。

**这样写，代价是什么。** 全参微调时 $\Delta W$ 的每个元素都能独立取值，训练结束它可以是任意一个 $4096\times4096$ 的矩阵。LoRA 只更新 $A$ 和 $B$，所以不管这两个小矩阵学成什么样，乘出来的 $\Delta W$ 的秩永远不超过 $r$（$B$ 只有 $r$ 列，见[附录 A.8](../../appendix/linear-algebra.md#a.8-矩阵乘积与列空间的包含)）；反过来，秩 $\le r$ 的矩阵也都能写成 $BA$（[附录 A.12](../../appendix/linear-algebra.md#a.12-秩分解定理与-svd)有证明）。

也就是说：**秩高于 $r$ 的那些 $\Delta W$，LoRA 根本表示不出来，训练再久也到不了。** 这和“近似”不是一回事——不是先算出真正的 $\Delta W$ 再拿低秩去逼近它，而是从头到尾只更新 $A$ 和 $B$，那些高秩的解从来没被碰过。

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

反过来 $r$ 太小，$A$、$B$ 装不下足够的信息，效果会掉。所以 $r$ 是在“能学到多少”和“花多少显存”之间取舍，不是越小越好——day 31 用实验找这个平衡点。

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

**混合精度还要一份 fp32 正本。** 前向反向用 bf16 算得快，但 bf16 只有 7 位尾数、两三位有效数字（fp32/fp16/bf16 各是什么见[附录 C](../../appendix/numeric-formats.md)），一次更新 $\eta\,m/\sqrt v$ 常常小到加到 bf16 权重上直接被舍入成零。

所以标准做法（ZeRO 论文[^zero] §3）是**同一个参数存两份**：

- **fp32 正本**（论文里叫 master weights）——训练期间它才是这个参数的权威值，所有更新都累加在它身上，全程不降精度；
- **bf16 工作副本**——从正本复制出来的低精度版本，只用来算前向和反向，因为 bf16 矩阵乘快。每更新完一步，就从正本重新复制一份覆盖它。

换句话说，bf16 那份是「用完就可以扔、随时能从正本再生成」的，fp32 正本才是真正被训练的东西。

![](../../site_src/assets/fig-train-step-light.svg){.fig .light-content fig-alt="一步训练里 bf16 工作副本与 fp32 正本各自何时被用到"}
![](../../site_src/assets/fig-train-step-dark.svg){.fig .dark-content fig-alt="一步训练里 bf16 工作副本与 fp32 正本各自何时被用到"}

先看一步训练里这些东西各自什么时候用到。设某个可训练参数是 $\theta$：

1. **前向**：用 $\theta$ 的 bf16 副本参与矩阵乘，算出 loss。bf16 走 Tensor Core，比 fp32 快得多。
2. **反向**：算出梯度 $g$，也是 bf16。
3. **更新**：把 $g$ 转成 fp32，更新 $m$ 和 $v$，算出更新量 $\eta\,\hat m/\sqrt{\hat v}$，**加到 $\theta$ 的 fp32 正本上**。
4. **同步**：把更新后的 fp32 正本转成 bf16，覆盖第 1 步用的那份工作副本，供下一轮前向。

第 3 步之所以必须在 fp32 正本上做，是因为更新量常常只有 $10^{-4}$ 量级，直接加到 bf16 上会被舍入成零（[附录 C](../../appendix/numeric-formats.md)）。

精度换算：bf16 是 16 位 = **2 字节**，fp32 是 32 位 = **4 字节**。逐项加起来：

| 存什么 | 什么时候用 | 精度 | 每个参数占 |
|---|---|---|---|
| $\theta$ 的 bf16 工作副本 | 步骤 1、2 算前向反向 | bf16 | 2 |
| 梯度 $g$ | 步骤 2 产出，步骤 3 消费 | bf16 | 2 |
| $\theta$ 的 fp32 正本 | 步骤 3 累加更新 | fp32 | 4 |
| Adam $m$ | 步骤 3，全程保留 | fp32 | 4 |
| Adam $v$ | 步骤 3，全程保留 | fp32 | 4 |
| **合计** | | | **16 字节** |

**冻结的参数只需要第一行的 2 字节。** 它不参与步骤 2–4：没有梯度、没有动量、不需要 fp32 正本——因为它根本不更新，bf16 那一份就是全部。

对 9B 模型，可训练参数 $N$：

| | 全参微调 $N = 9\times10^9$ | LoRA $N\approx2\times10^7$ |
|---|---|---|
| 冻结参数 × 2 B | 0 | $9\times10^9\times2$ = 18 GB |
| 可训练参数 × 16 B | $9\times10^9\times16$ = **144 GB** | $2\times10^7\times16$ = 0.3 GB |
| **合计（不含激活值）** | **~144 GB** | **~18.3 GB** |

一块 128 GB 统一内存的 Jetson AGX Thor 可用约 115 GB——**全参微调 9B 放不下，LoRA 剩一大半空间给激活值。** 24 GB 独显更是只有 LoRA/QLoRA 一条路。LoRA 省的就是这一块。90 亿个参数全部冻结，每个只占权重那 2 字节；额外那 14 字节只落在 4300 万个可训练参数头上。

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

**`alpha` 是干什么的。** §2.2 写的 $\Delta W = BA$ 是简化版。实际实现里还会乘一个系数：

$$
\Delta W = \frac{\alpha}{r}\,BA
$$

$r$ 变大时 $BA$ 的数值幅度大致随之变大，除以 $r$ 把尺度稳住，**这样改 $r$ 之后不必把学习率整个重调一遍**（LoRA 论文的原话是 reduces the need to retune，不是完全免除）。$\alpha$ 才是真正的强度旋钮，习惯取 $\alpha = 2r$（我们用 $r=16,\ \alpha=32$）。

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

### 3.0 准备环境（10 min）

**先做这一步。** 后面所有 `python code/…` 都在容器里跑，宿主机上没有 torch/peft/trl，直接跑会 `ModuleNotFoundError`。

在 Jetson 上用 NGC 的 PyTorch 容器：

```bash
cd days/day00_lora-quickstart
source ../../common/env.sh                    # HF_HUB_DISABLE_XET=1 等
bash ../../common/jetson_preflight.sh         # 任何一项 FAIL 就别起跑

sudo docker run -d --name t2t-day00 --runtime nvidia --ipc=host --network host \
  -e HF_HUB_DISABLE_XET=1 -e HF_HOME=/root/.cache/huggingface \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.08-py3 sleep infinity

sudo docker exec -it t2t-day00 bash code/setup_env.sh   # 装 pin 好的 trl/peft/transformers
```

容器起好后，后面每条命令前面加 `sudo docker exec -it t2t-day00`；嫌烦就 `sudo docker exec -it t2t-day00 bash` 进去一次，之后照抄命令即可。

**用独显的机器**可以跳过容器，在自己的 conda 环境里跑一次 `bash code/setup_env.sh`，后面所有命令去掉 `sudo docker exec -it t2t-day00` 前缀。


### 3.1 准备数据集（2 min）

仓库自带一份**演示数据集**，任何人 clone 下来都能直接用，不需要交出自己的任何数据：

```bash
sudo docker exec -it t2t-day00 python code/make_demo_dataset.py   # -> data/persona_demo.jsonl，137 条
```

它由两部分拼成，都在仓库里、都可复现：

- **本仓库文档里的中文段落**（附录 A/B/C、课表、SETUP 等）——真实技术散文；
- **`code/seeds.py` 里的中性问答种子**——覆盖课表话题和日常闲聊，让模型在聊天时也带着这套语气。

然后由 `code/stylize.py` 统一注入一组**明确且可统计**的风格标记：口癖开头（唔／诶／嗯…）、句尾 `～`、口癖结尾（……大概是这样吧～）。随机种子固定，所以**风格的真值已知**，训练效果可以直接量化。

> 为什么先用注入的风格而不是真实语料：真实语料的风格很浅（用词习惯、句子长度），微调完肉眼很难判断学没学到，容易自我欺骗。注入一组能数出来的标记，"有没有效果"就变成一个数字。等这条链路跑通了，再换成自己的语料（见 §6）。

### 3.2 看一眼数据（5 min）

```bash
sudo docker exec -it t2t-day00 python code/peek.py data/persona_demo.jsonl -n 3
```

（`data/persona_demo.jsonl` 是 JSONL——一行一个 JSON 对象。`python -m json.tool` 只能解析单个对象，喂多行会报 `Extra data`，所以用 `peek.py`。）

**这一步别跳过。** 数据里有什么，模型就学什么；数据里没有的，训一万步也不会有。

### 3.3 微调（40–60 min）

另开一个终端起遥测和看门狗：

```bash
nohup bash ../../common/jetson_telemetry.sh ~/telemetry/day00.log &
nohup bash ../../common/jetson_watchdog.sh 'train_lora' 85 &
```

然后训：

```bash
sudo docker exec -it t2t-day00 python code/train_lora.py \
    --model Qwen/Qwen3.5-9B \
    --data data/persona_demo.jsonl \
    --out private/adapter \
    --epochs 3 --rank 16 --batch 4 --lr 1e-4
```

> **模型选型和 wheel 版本以 [Jetson AI Lab «Fine-tune LLMs on Jetson»](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/)
> 为准**——那篇给了 Thor 上验证过的 Full SFT (4B) / LoRA (9B) / QLoRA (27B) 三档配置。
> 本仓库的脚本是通用 TRL + PEFT 写法，具体版本 pin 见 [踩坑](#5-踩坑)。

### 3.4 对比（20 min）

```bash
sudo docker exec -it t2t-day00 python code/compare.py \
    --model Qwen/Qwen3.5-9B --adapter private/adapter \
    --prompts code/prompts.txt \
    --out private/before_after.md
```

### 3.4b 量一下到底学到没有

```bash
sudo docker exec -it t2t-day00 python code/measure_style.py --model Qwen/Qwen3.5-9B --adapter private/adapter --prompts code/prompts.txt
```

同一批问题分别用 base 和 adapter 生成，统计风格标记的命中率。**这就是这天要留下的数字。**

> 以上串起来就是 `sudo docker exec -it t2t-day00 bash code/run_all.sh`，跑完直接进 3.5。

### 3.5 和它聊天

```bash
sudo docker exec -it t2t-day00 python code/chat.py --model Qwen/Qwen3.5-9B --adapter private/adapter
```

流式输出、多轮记忆、已关 thinking。`/base` 切到原模型、`/lora` 切回 adapter，同一个问题两边各问一遍最能看出差别；`/reset` 清空对话，`/quit` 退出。

## 4. 结果

Jetson AGX Thor（120 W），Qwen3.5-9B bf16，LoRA r=16、α=32、`all-linear`，cosine + 3 步 warmup，batch 4 × 累积 2，2 epoch。

| | 值 |
|---|---|
| 可训练参数 / 总参数 | 43.3 M / 8.997 B = **0.48%** |
| 训练耗时 | **3.2–3.9 min** |
| 峰值内存 | **23.7 GB** |
| 峰值 tj 温度 | **56 °C** |
| train loss | **4.42 → 2.82** |
| adapter 文件 | 166 MB |

对照 §2.4：权重 18 GB + 43 M × 16 B ≈ 0.7 GB，其余约 5 GB 是激活值和 CUDA 工作区。数字和推算对得上。

![训练曲线](results/training_curves.png)

### 效果：风格标记命中率

用仓库自带的 `data/persona_demo.jsonl`（137 条）训练，同一批 8 个问题，base 与 adapter 各生成一次：

| 标记 | base | + adapter |
|---|---|---|
| 句尾 `～` | 0 / 8 | **8 / 8** |
| 口癖开头（唔／诶／嗯…） | 0 / 8 | **5 / 8** |
| 口癖结尾（……大概是这样吧～） | 0 / 8 | **7 / 8** |

**137 条样本、3 分钟、0.48% 的参数，足以让 9B 模型换一套说话方式。**

> [!IMPORTANT]
> **这不是 prompt 工程，风格完全来自微调后的权重**
>
> `measure_style.py` 会先把真正送进模型的完整提示打印出来。base 和 adapter 收到的是**逐字节相同**的输入：
>
> ```
> '<|im_start|>user\n解释一下什么是 KV cache。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
> ```
>
> 没有 system prompt，没有"请用可爱语气回答"这类指令，没有 few-shot 示例。两次生成唯一的差别是 `model.disable_adapter()` 开还是关——也就是那 43 M 个 LoRA 参数加不加进去。`chat.py` 里的 `/base` 和 `/lora` 切换同理，你可以自己验。

### 学习率决定"学到风格"还是"背下语料"

同一份数据、同样轮数，只改学习率：

| lr | 表现 |
|---|---|
| 2e-4 | **过拟合**：开始逐字背训练语料，问 A 答 B |
| 1e-4 | 风格稳定命中，知识基本保留（上表就是这一档） |
| 5e-5 | **欠拟合**：回答和 base 几乎没区别 |

一百多条样本配 43 M 可训练参数，容量远大于数据量，学习率就是那个平衡钮。day 31 会正经扫这条曲线。

## 5. 踩坑

环境和版本问题都已经写进 `code/setup_env.sh` 和 `common/env.sh`，照跑即可。只有两件事值得知道原理：

1. **LoRA 挂在哪要看模型结构，不要照抄 `q_proj,k_proj,v_proj,o_proj`。** Qwen3.5-9B 是混合架构：32 层里 24 层是线性注意力（模块叫 `in_proj_qkv` / `out_proj`），只有 8 层有 `q/k/v/o_proj`。照抄只覆盖 **3.9 M（0.04%）** 参数；`target_modules="all-linear"` 是 **43.3 M（0.48%）**。查法：`AutoModelForCausalLM.from_config(cfg)` 在 meta 设备上建空模型，列 `nn.Linear` 的名字和形状，不用等权重。
2. **只对 assistant 算 loss 有两条路。** `assistant_only_loss=True` 要求 chat template 带 `{% generation %}` 标记，Qwen3.5 没有；改成 prompt/completion 格式，TRL 默认 `completion_only_loss=True`，不依赖模板。

## 6. 延伸

跑通之后，把演示数据集换成**你自己的语料**，这条支线会一直走到 day 72：

```bash
# 同样在容器里跑
sudo docker exec -it t2t-day00 python code/collect_corpus.py --git ~/Code/your-repo --author-email "$(git config user.email)" \
    --markdown ~/notes --out private/corpus.jsonl
sudo docker exec -it t2t-day00 python code/build_sft.py --in private/corpus.jsonl --out private/sft.jsonl --min-chars 40
sudo docker exec -it t2t-day00 python code/add_batch.py private/paste_*.txt      # 手动粘贴的聊天记录，自动合并连续消息
```

个人语料一律放 `private/`（已 gitignore），**不要进仓库**。

两个链接：

- [LoRA 论文](https://arxiv.org/abs/2106.09685)：§4.1 是 $\Delta W = BA$ 这个写法的出处，§7 是低秩假设的实验证据
- [Jetson AI Lab — Fine-tune LLMs on Jetson](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson/)

**明天要回答的问题**：这个 adapter 到底改了模型的什么？`r=16` 是多少个参数，凭什么够？→ day 31。

<!-- 参考文献用脚注 [^key] 写在这里，站点会自动汇总到文末的「参考文献」区 -->

[^lora]: Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." [*ICLR* 2022](https://openreview.net/forum?id=nZeVKeeFYf9). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). §4.1 是 $\Delta W = BA$ 这个写法的出处，§7 是低秩假设的实验证据。
[^aghajanyan]: Aghajanyan, A., Zettlemoyer, L. & Gupta, S. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." [*ACL* 2021](https://aclanthology.org/2021.acl-long.568/). [arXiv:2012.13255](https://arxiv.org/abs/2012.13255).
[^adam]: Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." [*ICLR* 2019](https://openreview.net/forum?id=Bkg6RiCqY7). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)（AdamW；Adam 本身见 Kingma & Ba, [*ICLR* 2015](https://openreview.net/forum?id=8gmWwjFyLj), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)）。详细推导见[附录 B](../../appendix/optimizers.md)。
[^zero]: Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." [*SC* 2020](https://doi.org/10.1109/SC41405.2020.00024). [arXiv:1910.02054](https://arxiv.org/abs/1910.02054). §3 的混合精度 Adam 内存账：每参数 16 字节。
