---
title: "附录 D · Transformer 速查：从一串 token 到下一个 token"
---

这一页把课表里反复出现的那块模型知识从头讲一遍：一个语言模型收到什么、吐出什么，中间那 32 层里到底在算什么，以及为什么推理时要有 KV cache。**不读这一页也能跑通 Day 00**——那天只需要知道“模型里有一堆矩阵，LoRA 给每个矩阵挂一对小矩阵”。但从 day 02 起，serving、CUDA、训练三条线都会不断回到这里。

概念按依赖顺序排：前一节是后一节的前提，从头读下来不会遇到没定义过的词。所有具体数字都用课表里那个模型（Qwen3.5-9B），这样和 [Day 00](../days/day00_lora-quickstart/README.md) 的实测对得上。

## D.0 读之前需要什么

真正卡人的前置只有四条，每条都能在这个仓库里补：

| 需要什么 | 用在哪 | 去哪补 |
|---|---|---|
| 矩阵乘向量、矩阵形状怎么对齐 | 全篇 | [附录 A](linear-algebra.md) §A.1–A.3 |
| 梯度下降、Adam 在存什么 | D.1 的训练目标；显存账 | [附录 B](optimizers.md) |
| fp32 / bf16 各占几个字节 | KV cache 和显存的所有估算 | [附录 C](numeric-formats.md) |
| softmax 与交叉熵 | D.1、D.3 | 下面 D.1 直接讲 |

不需要先学的：卷积网络、RNN/LSTM、强化学习。它们和这门课的主线没有依赖关系，等 Phase 5 讲 VLA 时再按需补。

会读一点 PyTorch 有帮助，但只需要认得 `nn.Linear(d_in, d_out)` 是“一次矩阵乘法（外加一个可选的偏置）”，以及 `forward()` 是数据流过模块的顺序。

## D.1 语言模型在做的唯一一件事

**定义（token 与词表）。** 一段文本先被切成一串整数，每个整数叫一个 **token**，取值范围是 $\{0,1,\dots,V-1\}$，$V$ 叫**词表大小**（vocabulary size）。Qwen3.5-9B 的 $V = 248320$。切分规则由分词器（tokenizer）决定，它是模型的一部分，不能换。

**定义（语言模型）。** 给定前 $t$ 个 token $x_1,\dots,x_t$，模型输出一个长度为 $V$ 的向量 $\mathbf{z}\in\mathbb{R}^V$，叫 **logits**；每个分量是“下一个 token 是这个词”的分数。把分数变成概率用 **softmax**：

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}},\qquad i = 1,\dots,V
$$

softmax 做两件事：指数保证结果为正，除以总和保证加起来等于 1。于是 $\mathbf{p}$ 是词表上的一个概率分布。

**训练目标。** 训练数据给出了真实的下一个 token $y$，损失取它的负对数概率，叫**交叉熵（cross-entropy）**：

$$
\mathcal{L} = -\log p_y
$$

猜得越准（$p_y$ 越接近 1），损失越接近 0；$p_y$ 接近 0 时损失趋于无穷。整条样本的损失是各位置损失的平均——[Day 00 §3.3](../days/day00_lora-quickstart/README.md) 里那个只在回答位置上算损失的掩码，掩的就是这个平均。

**推理。** 拿到 $\mathbf{p}$ 之后按它采样（sampling）出下一个 token，接到输入末尾，再算一次。**每生成一个 token，整个模型就从头到尾跑一遍**——这句话是后面所有性能讨论的起点。

::: {.callout-note title="温度和 top-p 是什么"}
采样前通常先把 logits 除以一个数 $T$（**温度**）再做 softmax。$T < 1$ 让分布更尖（更保守），$T > 1$ 更平（更随机），$T \to 0$ 等价于永远取最大的那个。**top-p**（核采样）则是先把概率从大到小排序，只保留累计概率刚超过 $p$ 的那一批候选，其余置零后重新归一化。Day 00 的 `chat.py` 用的是 $T = 0.7$、$p = 0.9$。
:::

## D.2 每个位置一个向量

模型内部不直接处理整数。第一步是查表：一个 $V \times d$ 的矩阵（`embed_tokens`）把每个 token id 映射成一个 $d$ 维向量，$d$ 叫**隐藏维度**（hidden size，Qwen3.5-9B 是 4096）。

于是长度为 $T$ 的输入变成一个 $T \times d$ 的矩阵：**每一行是一个位置的当前表示**。接下来 32 个块做的事，都是在改这 $T$ 行向量；形状自始至终是 $T \times d$，不变。

每个块的写法都是“算点东西，加回去”：

$$
\mathbf{h} \leftarrow \mathbf{h} + \text{某个模块}(\mathbf{h})
$$

这个加法叫**残差连接（residual connection）**。它带来两个好处：模块只需要学“该在原表示上补什么”，不必重新表达整个向量；反向传播时梯度可以顺着加法直接回到浅层，几十层的网络才训得动[^resnet]。

## D.3 注意力：让不同位置交换信息

到这里为止，每个位置各算各的，谁也不知道别人是什么。注意力就是负责交换的那一步。

### 先看它要解决的问题

> 小明把书放在桌上，然后他打开了**它**。

读到「它」的时候，模型要把「书」的信息取过来。难点在于：**该取谁，取决于内容，不取决于位置**——「它」不总是指前面第 5 个词。所以需要一种“按内容去找”的机制。

### 三个角色

关键的一步是：**同一个位置的向量，乘三个不同的矩阵，扮演三个不同角色**。

| 角色 | 怎么来的 | 意思 |
|---|---|---|
| **K**（键，key） | $\mathbf{k}_j = \mathbf{x}_j W_K$ | 我是什么，挂出去给别人看 |
| **V**（值，value） | $\mathbf{v}_j = \mathbf{x}_j W_V$ | 如果你选中我，我给你什么内容 |
| **Q**（查询，query） | $\mathbf{q}_i = \mathbf{x}_i W_Q$ | 我在找什么 |

在上面那句话里：「它」这个位置发出的查询大意是“我在找一个能被打开的、前面提过的东西”；「书」的键大意是“我是个可被指代的物件”。两者对得上，于是「它」把「书」的**值**取过来。

**为什么非要三个、不能只用一个向量。** 如果直接拿两个位置的表示做内积，衡量的是“谁和我像”。但这里需要的是“谁能回答我的问题”——问句和答案本来就不像。$W_Q$ 和 $W_K$ 的存在，就是让模型自己学出“什么样的提问该配什么样的应答”。$W_V$ 再单独一个，是因为“凭什么被选中”和“被选中后交出什么”也不是一回事。

### 一个能手算的例子

取 3 个 token、每个头 2 维（真实模型是 256 维，道理一样）。设第 3 个位置（「它」）的查询和三个位置的键、值分别是：

$$
\mathbf{q}_3 = (1.4,\ 0.2),\qquad
\begin{aligned}
\mathbf{k}_1 &= (1,\ 0) \\ \mathbf{k}_2 &= (0,\ 1) \\ \mathbf{k}_3 &= (0.2,\ 0.2)
\end{aligned}
\qquad
\begin{aligned}
\mathbf{v}_1 &= (2,\ 0) \\ \mathbf{v}_2 &= (0,\ 2) \\ \mathbf{v}_3 &= (0.1,\ 0.1)
\end{aligned}
$$

**第一步，打分。** 查询和每个键做内积，再除以 $\sqrt{d_k} = \sqrt{2} \approx 1.414$：

$$
\frac{\mathbf{q}_3\cdot\mathbf{k}_1}{\sqrt 2} = \frac{1.4}{1.414} = 0.99,\qquad
\frac{\mathbf{q}_3\cdot\mathbf{k}_2}{\sqrt 2} = \frac{0.2}{1.414} = 0.14,\qquad
\frac{\mathbf{q}_3\cdot\mathbf{k}_3}{\sqrt 2} = \frac{0.32}{1.414} = 0.23
$$

**第二步，softmax 变权重。** 按 D.1 的公式：$e^{0.99}=2.69$、$e^{0.14}=1.15$、$e^{0.23}=1.25$，总和 5.10，于是

$$
w = (0.53,\ 0.23,\ 0.25),\qquad w_1 + w_2 + w_3 = 1
$$

**第三步，按权重取值。**

$$
0.53\,\mathbf{v}_1 + 0.23\,\mathbf{v}_2 + 0.25\,\mathbf{v}_3 = (1.08,\ 0.48)
$$

结果被 $\mathbf{v}_1$ 主导——「它」这个位置的新表示里，装的主要是「书」的内容。**这就是注意力的全部。** 后面所有的公式和优化，都是这三步的高效批量版本。

```{=html}
<svg class="fig light-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 500" role="img" aria-label="注意力三步：查询与每个键内积得到分数，softmax 变成和为一的权重，再按权重把各位置的值加权求和"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#6B7280"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >注意力的三步</text><text x="30" y="58" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >「它」这个位置去看前面每一个位置，决定各取多少</text><rect x="30" y="82" width="200" height="62" rx="8" fill="#FFFFFF" stroke="#F5A623" stroke-width="1.2" /><text x="130.0" y="110" class="t2t-m" font-size="16.0" fill="#1A1A1A" text-anchor="middle" font-weight="600" >q₃ = (1.4, 0.2)</text><text x="130.0" y="132" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="middle" font-weight="400" >「它」在找什么</text><text x="30" y="176" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >位置</text><text x="150" y="176" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >键 k（我是什么）</text><text x="330" y="176" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >值 v（我给什么）</text><text x="500" y="176" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >分数 q·k/√2</text><text x="620" y="176" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >权重</text><rect x="30" y="194" width="690" height="52" rx="8" fill="#FFFFFF" stroke="#D7DBE0" stroke-width="1.2" /><text x="46" y="226" class="t2t-s" font-size="16.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >书</text><text x="150" y="226" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >k₁ = (1, 0)</text><text x="330" y="226" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >v₁ = (2, 0)</text><text x="500" y="226" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="400" >0.99</text><rect x="620" y="212" width="42.4" height="18" rx="4" fill="#76B900" fill-opacity=".85"/><text x="706" y="226" class="t2t-m" font-size="12.8" fill="#1A1A1A" text-anchor="end" font-weight="400" >0.53</text><rect x="30" y="256" width="690" height="52" rx="8" fill="#F5F6F8" stroke="#D7DBE0" stroke-width="1.2" /><text x="46" y="288" class="t2t-s" font-size="16.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >桌</text><text x="150" y="288" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >k₂ = (0, 1)</text><text x="330" y="288" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >v₂ = (0, 2)</text><text x="500" y="288" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="400" >0.14</text><rect x="620" y="274" width="18.4" height="18" rx="4" fill="#76B900" fill-opacity=".85"/><text x="706" y="288" class="t2t-m" font-size="12.8" fill="#1A1A1A" text-anchor="end" font-weight="400" >0.23</text><rect x="30" y="318" width="690" height="52" rx="8" fill="#F5F6F8" stroke="#D7DBE0" stroke-width="1.2" /><text x="46" y="350" class="t2t-s" font-size="16.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >它</text><text x="150" y="350" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >k₃ = (.2, .2)</text><text x="330" y="350" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >v₃ = (.1, .1)</text><text x="500" y="350" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="400" >0.23</text><rect x="620" y="336" width="20.0" height="18" rx="4" fill="#76B900" fill-opacity=".85"/><text x="706" y="350" class="t2t-m" font-size="12.8" fill="#1A1A1A" text-anchor="end" font-weight="400" >0.25</text><text x="30" y="404" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >① 打分：查询和每个键做内积，除以 √d 稳定数值</text><text x="30" y="428" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >② softmax：把三个分数变成加起来等于 1 的权重</text><text x="30" y="452" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >③ 加权求和：0.53·v₁ + 0.23·v₂ + 0.25·v₃</text><rect x="430" y="400" width="290" height="66" rx="8" fill="#FFFFFF" stroke="#76B900" stroke-width="1.2" /><text x="575.0" y="428" class="t2t-m" font-size="16.0" fill="#1A1A1A" text-anchor="middle" font-weight="600" >= (1.08, 0.48)</text><text x="575.0" y="450" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="middle" font-weight="400" >「它」的新表示，主要是「书」</text></svg>
```
```{=html}
<svg class="fig dark-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 500" role="img" aria-label="注意力三步：查询与每个键内积得到分数，softmax 变成和为一的权重，再按权重把各位置的值加权求和"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#9AA1A9"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >注意力的三步</text><text x="30" y="58" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >「它」这个位置去看前面每一个位置，决定各取多少</text><rect x="30" y="82" width="200" height="62" rx="8" fill="#242830" stroke="#F5A623" stroke-width="1.2" /><text x="130.0" y="110" class="t2t-m" font-size="16.0" fill="#E8E8E8" text-anchor="middle" font-weight="600" >q₃ = (1.4, 0.2)</text><text x="130.0" y="132" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="middle" font-weight="400" >「它」在找什么</text><text x="30" y="176" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >位置</text><text x="150" y="176" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >键 k（我是什么）</text><text x="330" y="176" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >值 v（我给什么）</text><text x="500" y="176" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >分数 q·k/√2</text><text x="620" y="176" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >权重</text><rect x="30" y="194" width="690" height="52" rx="8" fill="#242830" stroke="#3A404A" stroke-width="1.2" /><text x="46" y="226" class="t2t-s" font-size="16.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >书</text><text x="150" y="226" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >k₁ = (1, 0)</text><text x="330" y="226" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >v₁ = (2, 0)</text><text x="500" y="226" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="400" >0.99</text><rect x="620" y="212" width="42.4" height="18" rx="4" fill="#76B900" fill-opacity=".85"/><text x="706" y="226" class="t2t-m" font-size="12.8" fill="#E8E8E8" text-anchor="end" font-weight="400" >0.53</text><rect x="30" y="256" width="690" height="52" rx="8" fill="#1E222A" stroke="#3A404A" stroke-width="1.2" /><text x="46" y="288" class="t2t-s" font-size="16.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >桌</text><text x="150" y="288" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >k₂ = (0, 1)</text><text x="330" y="288" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >v₂ = (0, 2)</text><text x="500" y="288" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="400" >0.14</text><rect x="620" y="274" width="18.4" height="18" rx="4" fill="#76B900" fill-opacity=".85"/><text x="706" y="288" class="t2t-m" font-size="12.8" fill="#E8E8E8" text-anchor="end" font-weight="400" >0.23</text><rect x="30" y="318" width="690" height="52" rx="8" fill="#1E222A" stroke="#3A404A" stroke-width="1.2" /><text x="46" y="350" class="t2t-s" font-size="16.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >它</text><text x="150" y="350" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >k₃ = (.2, .2)</text><text x="330" y="350" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >v₃ = (.1, .1)</text><text x="500" y="350" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="400" >0.23</text><rect x="620" y="336" width="20.0" height="18" rx="4" fill="#76B900" fill-opacity=".85"/><text x="706" y="350" class="t2t-m" font-size="12.8" fill="#E8E8E8" text-anchor="end" font-weight="400" >0.25</text><text x="30" y="404" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >① 打分：查询和每个键做内积，除以 √d 稳定数值</text><text x="30" y="428" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >② softmax：把三个分数变成加起来等于 1 的权重</text><text x="30" y="452" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >③ 加权求和：0.53·v₁ + 0.23·v₂ + 0.25·v₃</text><rect x="430" y="400" width="290" height="66" rx="8" fill="#242830" stroke="#76B900" stroke-width="1.2" /><text x="575.0" y="428" class="t2t-m" font-size="16.0" fill="#E8E8E8" text-anchor="middle" font-weight="600" >= (1.08, 0.48)</text><text x="575.0" y="450" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="middle" font-weight="400" >「它」的新表示，主要是「书」</text></svg>
```

### 写成矩阵

把所有位置的查询、键、值各自堆成矩阵 $Q, K, V$（每行一个位置），上面三步就是一个式子[^attn]：

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V
$$

逐项对上面的例子：$QK^{\top}$ 是一个 $T\times T$ 的分数矩阵，第 $i$ 行第 $j$ 列就是“位置 $i$ 的查询 · 位置 $j$ 的键”；softmax 逐行做，把每一行变成和为 1 的权重；乘 $V$ 就是逐行的加权求和。

**为什么除以 $\sqrt{d_k}$。** 两个 $d_k$ 维随机向量的内积，方差随 $d_k$ 线性增长。$d_k = 256$ 时分数动辄几十，softmax 会尖锐到几乎只剩一个非零权重，梯度趋近于零。除以 $\sqrt{d_k}$ 把方差拉回常数量级（原论文[^attn] §3.2.1）。

**$M$ 是因果掩码（causal mask）。** 生成任务里位置 $i$ 只能看 $j \le i$：令 $M_{ij} = -\infty$（当 $j > i$），softmax 之后这些位置的权重正好是 0。所以真实的权重矩阵是下三角的。

### 多头（multi-head）

上一节从头到尾只算了一次注意力：一个查询、一组键、一组值，出一个输出。真实模型不是这么干的——但也**不是把整件事重复 16 遍**，而是**把向量切成 16 段，每段各算一遍**。

拿这个模型的数字说：$d = 4096$，头数 $h = 16$，每段 $d_k = 4096 / 16 = 256$ 维。

1. **切。** $\mathbf{q}$、$\mathbf{k}$、$\mathbf{v}$ 本来都是 4096 维的向量，按顺序切成 16 段，每段 256 维。**段与段之间互不来往**：第 3 段的查询只和第 3 段的键做内积，不碰别的段。
2. **各算各的。** 每一段独立走一遍前面那三步（打分、softmax、加权求和），于是得到 **16 张互相独立的 $T\times T$ 权重表**——下面那个可以点的图里，“第几个头”切的就是这 16 张表。
3. **拼回去。** 每段的输出是 256 个数，把 16 段按顺序排在一起——第 1 段的 256 个数在最前面，第 2 段接在后面，第 16 段在最后——就是 $16 \times 256 = 4096$ 个数，又一个 4096 维向量。最后乘一个 $4096\times4096$ 的矩阵 $W_O$（代码里的 `o_proj`），把 16 段的结果混合一遍，再加回主干。

::: {.callout-note title="“拼回去”就是把数排在一起，没有任何运算"}
不是相加，也不是平均。用 6 维、3 个头（每段 2 维）看得清楚些：

| | 内容 |
|---|---|
| 切之前的向量 | $(a_1, a_2, b_1, b_2, c_1, c_2)$ |
| 切成 3 段 | $(a_1,a_2)$　$(b_1,b_2)$　$(c_1,c_2)$ |
| 各段各自算完 | $(x_1,x_2)$　$(y_1,y_2)$　$(z_1,z_2)$ |
| 拼回去 | $(x_1, x_2, y_1, y_2, z_1, z_2)$ |

在代码里“切”和“拼”都只是 `reshape`：同一块数据，$(T, 4096)$ 换个看法成 $(T, 16, 256)$，算完再换回来，**一次乘法都没有**。真正在算的是每段里的注意力，以及最后那个 `o_proj`。
:::

```{=html}
<svg class="fig light-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 552" role="img" aria-label="多头注意力：4096 维切成 16 段各 256 维，每段各算一张 T×T 权重表，输出拼回 4096 维后过 o_proj"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#6B7280"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >多头：把向量切开，每段各算一遍</text><text x="30" y="74" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >查询 q（4096 维）——键 k、值 v 同样切</text><rect x="30.0" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="73.1" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="116.2" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="159.4" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="202.5" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="245.6" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="288.8" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="331.9" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="375.0" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="418.1" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="461.2" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="504.4" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="547.5" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="590.6" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="633.8" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="676.9" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><text x="720" y="142" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="end" font-weight="400" >16 段，每段 256 维</text><rect x="70.0" y="196.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="70.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="86.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="70.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="86.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="102.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="70.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="86.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="102.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="118.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="70.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="86.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="102.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="118.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="134.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="70.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="86.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="102.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="118.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="134.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="150.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="70" y="196" width="96" height="96" fill="none" stroke="#D7DBE0" stroke-width="1"/><text x="118.0" y="316" class="t2t-s" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >头 1</text><path d="M51.6 122V174H118.0V196" fill="none" stroke="#6B7280" stroke-width="1.5" marker-end="url(#qa)"/><rect x="300.0" y="196.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="300.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="316.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="300.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="316.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="332.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="300.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="316.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="332.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="348.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="300.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="316.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="332.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="348.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="364.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="300.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="316.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="332.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="348.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="364.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="380.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="300" y="196" width="96" height="96" fill="none" stroke="#D7DBE0" stroke-width="1"/><text x="348.0" y="316" class="t2t-s" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >头 2</text><path d="M94.7 122V174H348.0V196" fill="none" stroke="#6B7280" stroke-width="1.5" marker-end="url(#qa)"/><rect x="560.0" y="196.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="560.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="576.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="560.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="576.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="592.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="560.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="576.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="592.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="608.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="560.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="576.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="592.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="608.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="624.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="560.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="576.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="592.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="608.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="624.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="640.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="560" y="196" width="96" height="96" fill="none" stroke="#D7DBE0" stroke-width="1"/><text x="608.0" y="316" class="t2t-s" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >头 16</text><path d="M698.4 122V174H608.0V196" fill="none" stroke="#6B7280" stroke-width="1.5" marker-end="url(#qa)"/><text x="455" y="250.0" class="t2t-s" font-size="19.2" fill="#6B7280" text-anchor="middle" font-weight="400" >…</text><text x="30" y="344" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >每段自己走一遍“打分 → softmax → 加权求和”，得到自己的一张 T×T 权重表</text><text x="30" y="374" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >16 段的输出首尾相接，又是 4096 维</text><rect x="30.0" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="73.1" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="116.2" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="159.4" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="202.5" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="245.6" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="288.8" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="331.9" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="375.0" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="418.1" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="461.2" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="504.4" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="547.5" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="590.6" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="633.8" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="676.9" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><text x="720" y="442" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="end" font-weight="400" ></text><rect x="240" y="462" width="270" height="56" rx="8" fill="#FFFFFF" stroke="#76B900" stroke-width="1.2" /><text x="375.0" y="490" class="t2t-m" font-size="16.0" fill="#1A1A1A" text-anchor="middle" font-weight="600" >o_proj</text><text x="375.0" y="512" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="middle" font-weight="400" >4096 × 4096</text><path d="M375.0 422V462" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><text x="526" y="496.0" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >混合 16 段的结果，再加回主干</text></svg>
```
```{=html}
<svg class="fig dark-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 552" role="img" aria-label="多头注意力：4096 维切成 16 段各 256 维，每段各算一张 T×T 权重表，输出拼回 4096 维后过 o_proj"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#9AA1A9"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >多头：把向量切开，每段各算一遍</text><text x="30" y="74" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >查询 q（4096 维）——键 k、值 v 同样切</text><rect x="30.0" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="73.1" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="116.2" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="159.4" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="202.5" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="245.6" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="288.8" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="331.9" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="375.0" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="418.1" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="461.2" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="504.4" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="547.5" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="590.6" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="633.8" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="676.9" y="84" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><text x="720" y="142" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="end" font-weight="400" >16 段，每段 256 维</text><rect x="70.0" y="196.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="70.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="86.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="70.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="86.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="102.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="70.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="86.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="102.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="118.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="70.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="86.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="102.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="118.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="134.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="70.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="86.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="102.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="118.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="134.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="150.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="70" y="196" width="96" height="96" fill="none" stroke="#3A404A" stroke-width="1"/><text x="118.0" y="316" class="t2t-s" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >头 1</text><path d="M51.6 122V174H118.0V196" fill="none" stroke="#9AA1A9" stroke-width="1.5" marker-end="url(#qa)"/><rect x="300.0" y="196.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="300.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="316.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="300.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="316.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="332.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="300.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="316.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="332.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="348.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="300.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="316.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="332.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="348.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="364.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="300.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="316.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="332.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="348.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="364.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="380.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="300" y="196" width="96" height="96" fill="none" stroke="#3A404A" stroke-width="1"/><text x="348.0" y="316" class="t2t-s" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >头 2</text><path d="M94.7 122V174H348.0V196" fill="none" stroke="#9AA1A9" stroke-width="1.5" marker-end="url(#qa)"/><rect x="560.0" y="196.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="560.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="576.0" y="212.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="560.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="576.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="592.0" y="228.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="560.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="576.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="592.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="608.0" y="244.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="560.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="576.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="592.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="608.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="624.0" y="260.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="560.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="576.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="592.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="608.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.75"/><rect x="624.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.25"/><rect x="640.0" y="276.0" width="15.0" height="15.0" fill="#76B900" fill-opacity="0.50"/><rect x="560" y="196" width="96" height="96" fill="none" stroke="#3A404A" stroke-width="1"/><text x="608.0" y="316" class="t2t-s" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >头 16</text><path d="M698.4 122V174H608.0V196" fill="none" stroke="#9AA1A9" stroke-width="1.5" marker-end="url(#qa)"/><text x="455" y="250.0" class="t2t-s" font-size="19.2" fill="#9AA1A9" text-anchor="middle" font-weight="400" >…</text><text x="30" y="344" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >每段自己走一遍“打分 → softmax → 加权求和”，得到自己的一张 T×T 权重表</text><text x="30" y="374" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >16 段的输出首尾相接，又是 4096 维</text><rect x="30.0" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="73.1" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="116.2" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="159.4" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="202.5" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="245.6" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="288.8" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="331.9" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="375.0" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="418.1" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="461.2" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="504.4" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="547.5" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="590.6" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><rect x="633.8" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.3" stroke="#0074C8" stroke-width="1"/><rect x="676.9" y="384" width="41.1" height="38" rx="3" fill="#0074C8" fill-opacity="0.5" stroke="#0074C8" stroke-width="1"/><text x="720" y="442" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="end" font-weight="400" ></text><rect x="240" y="462" width="270" height="56" rx="8" fill="#242830" stroke="#76B900" stroke-width="1.2" /><text x="375.0" y="490" class="t2t-m" font-size="16.0" fill="#E8E8E8" text-anchor="middle" font-weight="600" >o_proj</text><text x="375.0" y="512" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="middle" font-weight="400" >4096 × 4096</text><path d="M375.0 422V462" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><text x="526" y="496.0" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >混合 16 段的结果，再加回主干</text></svg>
```

**整层的形状怎么变。** 把这三步放回一层里，张量的形状是这样走的。写法和 PyTorch 一致：$T$ 是这次输入的 token 数，`@` 是 Python 的矩阵乘法运算符（`a @ b` 就是矩阵 $a$ 乘矩阵 $b$，NumPy 和 PyTorch 都用它），$k^{\top}$ 是 $k$ 的转置——行列互换（[附录 A.1](linear-algebra.md)）。

```{=html}
<svg class="fig light-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 896" role="img" aria-label="一层注意力里张量形状的变化：(T,4096) 经拆头成 (16,T,256)，算出 (16,T,T) 的权重，输出 (16,T,256)，合头回 (T,4096)"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#6B7280"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >一层注意力里，形状怎么变</text><text x="30" y="58" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >T = 这次输入的 token 数；16 个头 × 每头 256 维 = 4096</text><rect x="30" y="84" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#0074C8" stroke-width="1.2" /><text x="46" y="118.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >输入 h</text><rect x="316" y="92" width="170" height="40" rx="6" fill="#0074C8" fill-opacity=".12" stroke="#0074C8" stroke-width="1.2"/><text x="401" y="118.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(T, 4096)</text><text x="502" y="118.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >每个位置一个 4096 维向量</text><path d="M164.0 140V170" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="170" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#0074C8" stroke-width="1.2" /><text x="46" y="204.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >q  k  v</text><rect x="316" y="178" width="170" height="40" rx="6" fill="#0074C8" fill-opacity=".12" stroke="#0074C8" stroke-width="1.2"/><text x="401" y="204.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(T, 4096) 各一个</text><text x="502" y="204.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >各乘 W_Q / W_K / W_V</text><path d="M164.0 226V256" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="256" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#F5A623" stroke-width="1.2" /><text x="46" y="290.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >拆头（reshape）</text><rect x="316" y="264" width="170" height="40" rx="6" fill="#F5A623" fill-opacity=".12" stroke="#F5A623" stroke-width="1.2"/><text x="401" y="290.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(16, T, 256)</text><text x="502" y="290.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >4096 拆成 16 × 256</text><path d="M164.0 312V342" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="342" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#76B900" stroke-width="1.2" /><text x="46" y="376.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >分数 = q @ kᵀ / √256</text><rect x="316" y="350" width="170" height="40" rx="6" fill="#76B900" fill-opacity=".12" stroke="#76B900" stroke-width="1.2"/><text x="401" y="376.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(16, T, T)</text><text x="502" y="376.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >每个头一张 T×T 表</text><path d="M164.0 398V428" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="428" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#76B900" stroke-width="1.2" /><text x="46" y="462.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >加因果掩码，逐行 softmax</text><rect x="316" y="436" width="170" height="40" rx="6" fill="#76B900" fill-opacity=".12" stroke="#76B900" stroke-width="1.2"/><text x="401" y="462.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(16, T, T)</text><text x="502" y="462.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >形状不变，每行和为 1</text><path d="M164.0 484V514" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="514" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#76B900" stroke-width="1.2" /><text x="46" y="548.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >权重 @ v</text><rect x="316" y="522" width="170" height="40" rx="6" fill="#76B900" fill-opacity=".12" stroke="#76B900" stroke-width="1.2"/><text x="401" y="548.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(16, T, 256)</text><text x="502" y="548.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >每个头输出 256 维</text><path d="M164.0 570V600" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="600" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#F5A623" stroke-width="1.2" /><text x="46" y="634.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >合头（reshape）</text><rect x="316" y="608" width="170" height="40" rx="6" fill="#F5A623" fill-opacity=".12" stroke="#F5A623" stroke-width="1.2"/><text x="401" y="634.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(T, 4096)</text><text x="502" y="634.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >16 × 256 拼回 4096</text><path d="M164.0 656V686" stroke="#6B7280" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="686" width="268" height="56" rx="8" fill="#FFFFFF" stroke="#0074C8" stroke-width="1.2" /><text x="46" y="720.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="600" >× W_O（o_proj）</text><rect x="316" y="694" width="170" height="40" rx="6" fill="#0074C8" fill-opacity=".12" stroke="#0074C8" stroke-width="1.2"/><text x="401" y="720.0" class="t2t-m" font-size="14.1" fill="#1A1A1A" text-anchor="middle" font-weight="700" >(T, 4096)</text><text x="502" y="720.0" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >4096 × 4096 的矩阵</text><text x="30" y="796" class="t2t-s" font-size="14.1" fill="#1A1A1A" text-anchor="start" font-weight="700" >进来 (T, 4096)，出去还是 (T, 4096)——注意力不改变形状，只改变内容。</text><text x="30" y="822" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >橙色那两步只是换个看法（reshape），一次乘法都没有；真正的计算在绿色三步和两端的矩阵乘法里。</text><text x="30" y="846" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >这个模型用 GQA：k、v 只有 4 组，形状是 (4, T, 256)，算的时候一组给 4 个查询头共用。</text><text x="30" y="872" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >@ 是 Python 的矩阵乘法运算符（a @ b 即矩阵 a 乘矩阵 b）；kᵀ 是 k 的转置，行列互换。</text></svg>
```
```{=html}
<svg class="fig dark-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 896" role="img" aria-label="一层注意力里张量形状的变化：(T,4096) 经拆头成 (16,T,256)，算出 (16,T,T) 的权重，输出 (16,T,256)，合头回 (T,4096)"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#9AA1A9"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >一层注意力里，形状怎么变</text><text x="30" y="58" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >T = 这次输入的 token 数；16 个头 × 每头 256 维 = 4096</text><rect x="30" y="84" width="268" height="56" rx="8" fill="#242830" stroke="#0074C8" stroke-width="1.2" /><text x="46" y="118.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >输入 h</text><rect x="316" y="92" width="170" height="40" rx="6" fill="#0074C8" fill-opacity=".12" stroke="#0074C8" stroke-width="1.2"/><text x="401" y="118.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(T, 4096)</text><text x="502" y="118.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >每个位置一个 4096 维向量</text><path d="M164.0 140V170" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="170" width="268" height="56" rx="8" fill="#242830" stroke="#0074C8" stroke-width="1.2" /><text x="46" y="204.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >q  k  v</text><rect x="316" y="178" width="170" height="40" rx="6" fill="#0074C8" fill-opacity=".12" stroke="#0074C8" stroke-width="1.2"/><text x="401" y="204.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(T, 4096) 各一个</text><text x="502" y="204.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >各乘 W_Q / W_K / W_V</text><path d="M164.0 226V256" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="256" width="268" height="56" rx="8" fill="#242830" stroke="#F5A623" stroke-width="1.2" /><text x="46" y="290.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >拆头（reshape）</text><rect x="316" y="264" width="170" height="40" rx="6" fill="#F5A623" fill-opacity=".12" stroke="#F5A623" stroke-width="1.2"/><text x="401" y="290.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(16, T, 256)</text><text x="502" y="290.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >4096 拆成 16 × 256</text><path d="M164.0 312V342" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="342" width="268" height="56" rx="8" fill="#242830" stroke="#76B900" stroke-width="1.2" /><text x="46" y="376.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >分数 = q @ kᵀ / √256</text><rect x="316" y="350" width="170" height="40" rx="6" fill="#76B900" fill-opacity=".12" stroke="#76B900" stroke-width="1.2"/><text x="401" y="376.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(16, T, T)</text><text x="502" y="376.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >每个头一张 T×T 表</text><path d="M164.0 398V428" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="428" width="268" height="56" rx="8" fill="#242830" stroke="#76B900" stroke-width="1.2" /><text x="46" y="462.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >加因果掩码，逐行 softmax</text><rect x="316" y="436" width="170" height="40" rx="6" fill="#76B900" fill-opacity=".12" stroke="#76B900" stroke-width="1.2"/><text x="401" y="462.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(16, T, T)</text><text x="502" y="462.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >形状不变，每行和为 1</text><path d="M164.0 484V514" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="514" width="268" height="56" rx="8" fill="#242830" stroke="#76B900" stroke-width="1.2" /><text x="46" y="548.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >权重 @ v</text><rect x="316" y="522" width="170" height="40" rx="6" fill="#76B900" fill-opacity=".12" stroke="#76B900" stroke-width="1.2"/><text x="401" y="548.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(16, T, 256)</text><text x="502" y="548.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >每个头输出 256 维</text><path d="M164.0 570V600" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="600" width="268" height="56" rx="8" fill="#242830" stroke="#F5A623" stroke-width="1.2" /><text x="46" y="634.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >合头（reshape）</text><rect x="316" y="608" width="170" height="40" rx="6" fill="#F5A623" fill-opacity=".12" stroke="#F5A623" stroke-width="1.2"/><text x="401" y="634.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(T, 4096)</text><text x="502" y="634.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >16 × 256 拼回 4096</text><path d="M164.0 656V686" stroke="#9AA1A9" stroke-width="1.6" marker-end="url(#qa)"/><rect x="30" y="686" width="268" height="56" rx="8" fill="#242830" stroke="#0074C8" stroke-width="1.2" /><text x="46" y="720.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="600" >× W_O（o_proj）</text><rect x="316" y="694" width="170" height="40" rx="6" fill="#0074C8" fill-opacity=".12" stroke="#0074C8" stroke-width="1.2"/><text x="401" y="720.0" class="t2t-m" font-size="14.1" fill="#E8E8E8" text-anchor="middle" font-weight="700" >(T, 4096)</text><text x="502" y="720.0" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >4096 × 4096 的矩阵</text><text x="30" y="796" class="t2t-s" font-size="14.1" fill="#E8E8E8" text-anchor="start" font-weight="700" >进来 (T, 4096)，出去还是 (T, 4096)——注意力不改变形状，只改变内容。</text><text x="30" y="822" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >橙色那两步只是换个看法（reshape），一次乘法都没有；真正的计算在绿色三步和两端的矩阵乘法里。</text><text x="30" y="846" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >这个模型用 GQA：k、v 只有 4 组，形状是 (4, T, 256)，算的时候一组给 4 个查询头共用。</text><text x="30" y="872" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >@ 是 Python 的矩阵乘法运算符（a @ b 即矩阵 a 乘矩阵 b）；kᵀ 是 k 的转置，行列互换。</text></svg>
```

一句话记住：**进来 $(T, 4096)$，出去还是 $(T, 4096)$**。注意力只改内容、不改形状，所以 32 层才能一层接一层地摞起来。中间冒出来的 $(16, T, T)$ 就是那 16 张权重表，也是显存和计算量真正的大头。

**为什么要切，不直接算一次 4096 维的注意力？** 因为一次注意力只能产生**一套**权重：4096 个维度共用同一张“谁看谁”的表。切成 16 段之后就有 16 套独立的权重，模型可以拿不同的段去追不同的关系——有的段盯指代，有的段盯句法距离。

**代价是零。** 一段的打分是 256 维的内积，16 段加起来 $16 \times 256 = 4096$，和一次 4096 维内积的计算量一模一样。多头换来的多样性不额外花算力（真实实现里 16 段合成一个批量矩阵乘法，一次算完）。

::: {.callout-note title="这个模型的头不是标准多头"}
标准多头是 16 段查询各配 16 段键和值。Qwen3.5-9B 用的是 GQA：查询仍然 16 段，键和值只有 4 组，每 4 段查询共用一组。省下来的是 KV cache，见 D.8。
:::

### 真实的权重长什么样

上面那组数字是我编的，为了能手算。下面这张是**真的**：把同一句话喂给 Qwen3.5-9B 跑一次前向，把权重导出来（`scripts/dump_attention.py`，一次前向几秒钟）。

**怎么读这张图。** 行 = 谁在看（发出查询的位置），列 = 被看的位置（提供键的位置），颜色越深权重越大。上三角是空的，因为因果掩码不让往后看。每一行加起来等于 1。

图上有两组按钮，对应刚讲过的两件事：

- **层**：模型是 32 个块摞起来的（D.2），每个块里都有一次注意力。“第 11 层”就是**从下往上数第 11 个块里的那次注意力**。这里只有 8 个层号可选（3、7、11…31），因为这个模型 32 层里只有这 8 层用标准注意力，其余 24 层用的是另一种不产生 $T\times T$ 矩阵的算法（D.9 会提）。
- **头**：同一层里注意力不是只算一次，而是切成 16 份各算一遍——就是上一小节的“多头（multi-head）”。“平均”是把 16 个头的权重平均了看，点数字则是单看某一个头。逐头数据只导了第 31 层。

```{=html}
<script type="application/json" id="attn-data">{"text":"小明把书放在桌上，然后他打开了它。","tokens":["小明","把","书","放在","桌上","，","然后","他","打开了","它","。"],"layers":["3","7","11","15","19","23","27","31"],"mean":{"3":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.598,0.402,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.423,0.317,0.259,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.269,0.205,0.266,0.26,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.297,0.132,0.153,0.21,0.208,0.0,0.0,0.0,0.0,0.0,0.0],[0.294,0.115,0.084,0.116,0.132,0.259,0.0,0.0,0.0,0.0,0.0],[0.133,0.127,0.074,0.12,0.107,0.211,0.228,0.0,0.0,0.0,0.0],[0.27,0.121,0.083,0.071,0.075,0.134,0.137,0.109,0.0,0.0,0.0],[0.135,0.068,0.162,0.123,0.133,0.084,0.067,0.05,0.177,0.0,0.0],[0.149,0.05,0.138,0.087,0.085,0.095,0.07,0.071,0.122,0.132,0.0],[0.18,0.051,0.048,0.046,0.058,0.089,0.063,0.053,0.111,0.066,0.233]],"7":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.3,0.7,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.212,0.251,0.537,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.189,0.158,0.183,0.469,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.095,0.102,0.15,0.153,0.5,0.0,0.0,0.0,0.0,0.0,0.0],[0.143,0.134,0.053,0.079,0.202,0.388,0.0,0.0,0.0,0.0,0.0],[0.14,0.116,0.027,0.099,0.127,0.22,0.27,0.0,0.0,0.0,0.0],[0.1,0.262,0.04,0.051,0.06,0.173,0.104,0.21,0.0,0.0,0.0],[0.075,0.04,0.118,0.103,0.064,0.131,0.069,0.025,0.375,0.0,0.0],[0.077,0.031,0.214,0.082,0.08,0.135,0.044,0.024,0.123,0.189,0.0],[0.062,0.039,0.023,0.044,0.035,0.138,0.101,0.032,0.133,0.092,0.3]],"11":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.426,0.574,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.234,0.161,0.604,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.184,0.17,0.257,0.39,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.26,0.091,0.126,0.081,0.442,0.0,0.0,0.0,0.0,0.0,0.0],[0.254,0.14,0.114,0.113,0.108,0.27,0.0,0.0,0.0,0.0,0.0],[0.156,0.141,0.038,0.082,0.115,0.18,0.289,0.0,0.0,0.0,0.0],[0.072,0.294,0.042,0.061,0.06,0.09,0.072,0.31,0.0,0.0,0.0],[0.099,0.032,0.192,0.113,0.069,0.101,0.044,0.025,0.326,0.0,0.0],[0.045,0.032,0.439,0.081,0.079,0.068,0.017,0.024,0.095,0.121,0.0],[0.122,0.045,0.042,0.055,0.051,0.095,0.069,0.047,0.147,0.053,0.274]],"15":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.533,0.467,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.489,0.11,0.401,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.448,0.12,0.157,0.276,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.317,0.084,0.065,0.121,0.414,0.0,0.0,0.0,0.0,0.0,0.0],[0.472,0.107,0.057,0.087,0.098,0.178,0.0,0.0,0.0,0.0,0.0],[0.216,0.107,0.067,0.139,0.175,0.106,0.191,0.0,0.0,0.0,0.0],[0.125,0.29,0.102,0.112,0.08,0.08,0.048,0.163,0.0,0.0,0.0],[0.141,0.022,0.137,0.104,0.108,0.09,0.057,0.023,0.319,0.0,0.0],[0.09,0.016,0.248,0.083,0.104,0.048,0.029,0.017,0.162,0.202,0.0],[0.234,0.051,0.035,0.05,0.061,0.068,0.058,0.048,0.102,0.083,0.209]],"19":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.702,0.298,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.473,0.167,0.36,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.39,0.081,0.217,0.313,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.236,0.09,0.106,0.152,0.415,0.0,0.0,0.0,0.0,0.0,0.0],[0.199,0.183,0.094,0.122,0.195,0.206,0.0,0.0,0.0,0.0,0.0],[0.188,0.12,0.065,0.108,0.164,0.108,0.246,0.0,0.0,0.0,0.0],[0.173,0.102,0.066,0.082,0.154,0.089,0.15,0.185,0.0,0.0,0.0],[0.166,0.025,0.265,0.041,0.072,0.038,0.033,0.038,0.32,0.0,0.0],[0.131,0.041,0.178,0.057,0.04,0.053,0.059,0.036,0.139,0.267,0.0],[0.293,0.091,0.025,0.027,0.03,0.077,0.092,0.085,0.033,0.033,0.213]],"23":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.577,0.423,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.241,0.247,0.512,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.284,0.088,0.273,0.355,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.201,0.139,0.248,0.162,0.25,0.0,0.0,0.0,0.0,0.0,0.0],[0.221,0.116,0.184,0.072,0.079,0.328,0.0,0.0,0.0,0.0,0.0],[0.156,0.111,0.096,0.085,0.095,0.347,0.109,0.0,0.0,0.0,0.0],[0.074,0.101,0.126,0.062,0.08,0.373,0.086,0.097,0.0,0.0,0.0],[0.048,0.02,0.233,0.038,0.107,0.324,0.021,0.022,0.188,0.0,0.0],[0.133,0.041,0.13,0.044,0.027,0.293,0.031,0.028,0.173,0.098,0.0],[0.161,0.076,0.06,0.051,0.033,0.169,0.043,0.046,0.074,0.022,0.264]],"27":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.579,0.421,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.221,0.337,0.442,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.212,0.088,0.252,0.448,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.174,0.174,0.214,0.18,0.258,0.0,0.0,0.0,0.0,0.0,0.0],[0.15,0.081,0.15,0.035,0.062,0.522,0.0,0.0,0.0,0.0,0.0],[0.085,0.151,0.073,0.076,0.055,0.528,0.032,0.0,0.0,0.0,0.0],[0.037,0.169,0.06,0.062,0.05,0.495,0.029,0.097,0.0,0.0,0.0],[0.044,0.014,0.266,0.016,0.08,0.467,0.007,0.017,0.09,0.0,0.0],[0.167,0.036,0.14,0.03,0.021,0.372,0.044,0.027,0.118,0.045,0.0],[0.204,0.039,0.066,0.018,0.019,0.2,0.052,0.021,0.056,0.008,0.317]],"31":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.37,0.63,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.133,0.294,0.572,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.114,0.095,0.252,0.538,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.134,0.129,0.141,0.16,0.435,0.0,0.0,0.0,0.0,0.0,0.0],[0.106,0.074,0.103,0.048,0.108,0.561,0.0,0.0,0.0,0.0,0.0],[0.103,0.109,0.059,0.053,0.058,0.286,0.331,0.0,0.0,0.0,0.0],[0.089,0.105,0.039,0.06,0.039,0.24,0.115,0.311,0.0,0.0,0.0],[0.066,0.047,0.133,0.036,0.079,0.208,0.032,0.038,0.361,0.0,0.0],[0.133,0.049,0.066,0.029,0.04,0.232,0.034,0.037,0.104,0.275,0.0],[0.153,0.028,0.048,0.017,0.025,0.123,0.029,0.033,0.033,0.037,0.476]]},"head_layer":31,"heads":[[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.547,0.453,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.151,0.41,0.438,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.062,0.066,0.316,0.555,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.161,0.122,0.167,0.293,0.258,0.0,0.0,0.0,0.0,0.0,0.0],[0.104,0.063,0.177,0.071,0.146,0.438,0.0,0.0,0.0,0.0,0.0],[0.182,0.11,0.097,0.076,0.076,0.408,0.052,0.0,0.0,0.0,0.0],[0.068,0.153,0.056,0.105,0.06,0.473,0.035,0.05,0.0,0.0,0.0],[0.071,0.014,0.281,0.016,0.097,0.463,0.005,0.007,0.046,0.0,0.0],[0.285,0.062,0.119,0.051,0.038,0.27,0.012,0.017,0.127,0.02,0.0],[0.375,0.043,0.107,0.024,0.056,0.122,0.021,0.015,0.038,0.01,0.188]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.547,0.453,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.165,0.328,0.508,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.172,0.063,0.283,0.482,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.146,0.129,0.22,0.206,0.299,0.0,0.0,0.0,0.0,0.0,0.0],[0.099,0.077,0.196,0.047,0.112,0.471,0.0,0.0,0.0,0.0,0.0],[0.142,0.16,0.125,0.071,0.046,0.408,0.049,0.0,0.0,0.0,0.0],[0.066,0.192,0.066,0.11,0.033,0.408,0.038,0.085,0.0,0.0,0.0],[0.072,0.021,0.312,0.017,0.147,0.332,0.009,0.018,0.07,0.0,0.0],[0.25,0.063,0.152,0.034,0.028,0.268,0.014,0.035,0.119,0.037,0.0],[0.264,0.057,0.17,0.018,0.022,0.146,0.014,0.025,0.033,0.013,0.239]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.453,0.547,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.119,0.547,0.332,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.121,0.094,0.465,0.32,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.141,0.149,0.181,0.231,0.297,0.0,0.0,0.0,0.0,0.0,0.0],[0.218,0.097,0.181,0.071,0.117,0.316,0.0,0.0,0.0,0.0,0.0],[0.149,0.141,0.085,0.109,0.062,0.406,0.048,0.0,0.0,0.0,0.0],[0.066,0.158,0.055,0.102,0.045,0.486,0.035,0.051,0.0,0.0,0.0],[0.047,0.023,0.338,0.022,0.125,0.359,0.005,0.012,0.066,0.0,0.0],[0.254,0.05,0.145,0.047,0.024,0.346,0.018,0.022,0.083,0.012,0.0],[0.49,0.045,0.058,0.013,0.012,0.132,0.013,0.02,0.03,0.005,0.18]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.126,0.293,0.582,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.142,0.11,0.361,0.385,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.105,0.09,0.216,0.295,0.295,0.0,0.0,0.0,0.0,0.0,0.0],[0.097,0.042,0.213,0.071,0.142,0.436,0.0,0.0,0.0,0.0,0.0],[0.134,0.111,0.111,0.081,0.063,0.438,0.063,0.0,0.0,0.0,0.0],[0.059,0.15,0.062,0.133,0.059,0.436,0.049,0.052,0.0,0.0,0.0],[0.05,0.014,0.318,0.031,0.125,0.318,0.014,0.008,0.125,0.0,0.0],[0.117,0.027,0.15,0.063,0.046,0.248,0.024,0.01,0.281,0.034,0.0],[0.178,0.025,0.189,0.049,0.052,0.147,0.046,0.018,0.104,0.018,0.172]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.547,0.453,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.115,0.471,0.414,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.116,0.102,0.486,0.295,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.125,0.232,0.192,0.17,0.281,0.0,0.0,0.0,0.0,0.0,0.0],[0.114,0.081,0.228,0.079,0.167,0.332,0.0,0.0,0.0,0.0,0.0],[0.08,0.169,0.149,0.091,0.103,0.204,0.204,0.0,0.0,0.0,0.0],[0.076,0.165,0.1,0.083,0.071,0.176,0.129,0.199,0.0,0.0,0.0],[0.069,0.026,0.273,0.031,0.213,0.188,0.041,0.052,0.107,0.0,0.0],[0.11,0.069,0.073,0.042,0.052,0.289,0.054,0.047,0.117,0.146,0.0],[0.127,0.022,0.035,0.022,0.024,0.153,0.051,0.04,0.051,0.058,0.416]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.484,0.516,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.11,0.311,0.578,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.112,0.038,0.346,0.504,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.114,0.241,0.165,0.107,0.373,0.0,0.0,0.0,0.0,0.0,0.0],[0.122,0.095,0.115,0.032,0.122,0.516,0.0,0.0,0.0,0.0,0.0],[0.079,0.229,0.048,0.051,0.07,0.295,0.229,0.0,0.0,0.0,0.0],[0.065,0.166,0.035,0.051,0.037,0.227,0.089,0.33,0.0,0.0,0.0],[0.045,0.027,0.139,0.023,0.189,0.202,0.051,0.157,0.167,0.0,0.0],[0.058,0.075,0.028,0.02,0.04,0.459,0.035,0.058,0.109,0.116,0.0],[0.093,0.033,0.02,0.008,0.018,0.127,0.041,0.087,0.05,0.082,0.441]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.484,0.516,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.243,0.455,0.303,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.173,0.067,0.275,0.484,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.287,0.145,0.21,0.174,0.186,0.0,0.0,0.0,0.0,0.0,0.0],[0.124,0.04,0.09,0.058,0.062,0.625,0.0,0.0,0.0,0.0,0.0],[0.166,0.069,0.071,0.086,0.046,0.512,0.051,0.0,0.0,0.0,0.0],[0.181,0.08,0.053,0.097,0.034,0.49,0.034,0.03,0.0,0.0,0.0],[0.121,0.037,0.11,0.043,0.062,0.523,0.023,0.017,0.061,0.0,0.0],[0.137,0.039,0.039,0.016,0.023,0.613,0.02,0.012,0.029,0.073,0.0],[0.141,0.018,0.029,0.032,0.015,0.262,0.022,0.006,0.021,0.021,0.434]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.629,0.371,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.163,0.305,0.535,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.228,0.115,0.4,0.258,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.129,0.273,0.194,0.22,0.183,0.0,0.0,0.0,0.0,0.0,0.0],[0.08,0.141,0.066,0.04,0.045,0.629,0.0,0.0,0.0,0.0,0.0],[0.05,0.171,0.054,0.104,0.044,0.449,0.129,0.0,0.0,0.0,0.0],[0.045,0.174,0.029,0.096,0.026,0.404,0.08,0.145,0.0,0.0,0.0],[0.036,0.031,0.108,0.022,0.084,0.426,0.063,0.086,0.143,0.0,0.0],[0.049,0.056,0.03,0.036,0.029,0.469,0.067,0.054,0.115,0.095,0.0],[0.082,0.04,0.013,0.012,0.01,0.12,0.055,0.148,0.043,0.073,0.404]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.095,0.906,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.131,0.116,0.754,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.042,0.225,0.042,0.691,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.111,0.098,0.067,0.041,0.684,0.0,0.0,0.0,0.0,0.0,0.0],[0.111,0.184,0.034,0.028,0.043,0.602,0.0,0.0,0.0,0.0,0.0],[0.099,0.063,0.019,0.011,0.032,0.173,0.602,0.0,0.0,0.0,0.0],[0.188,0.027,0.019,0.007,0.011,0.051,0.083,0.613,0.0,0.0,0.0],[0.051,0.188,0.016,0.083,0.025,0.061,0.054,0.013,0.508,0.0,0.0],[0.177,0.061,0.057,0.014,0.061,0.048,0.035,0.051,0.017,0.48,0.0],[0.028,0.01,0.003,0.001,0.005,0.052,0.01,0.004,0.004,0.012,0.871]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.005,0.996,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.001,0.002,0.996,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.005,0.001,0.992,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.001,0.0,0.0,0.001,0.996,0.0,0.0,0.0,0.0,0.0,0.0],[0.003,0.003,0.001,0.002,0.007,0.984,0.0,0.0,0.0,0.0,0.0],[0.004,0.001,0.0,0.0,0.001,0.006,0.988,0.0,0.0,0.0,0.0],[0.004,0.0,0.0,0.0,0.0,0.001,0.024,0.969,0.0,0.0,0.0],[0.0,0.0,0.0,0.003,0.0,0.0,0.0,0.0,0.996,0.0,0.0],[0.001,0.0,0.001,0.0,0.004,0.0,0.0,0.002,0.003,0.988,0.0],[0.002,0.0,0.0,0.0,0.0,0.004,0.001,0.0,0.001,0.001,0.992]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.063,0.938,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.062,0.089,0.848,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.012,0.064,0.032,0.891,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.037,0.022,0.045,0.051,0.844,0.0,0.0,0.0,0.0,0.0,0.0],[0.041,0.028,0.025,0.028,0.063,0.816,0.0,0.0,0.0,0.0,0.0],[0.028,0.013,0.012,0.012,0.049,0.218,0.668,0.0,0.0,0.0,0.0],[0.063,0.015,0.017,0.008,0.022,0.104,0.207,0.562,0.0,0.0,0.0],[0.013,0.043,0.012,0.091,0.016,0.028,0.02,0.01,0.766,0.0,0.0],[0.045,0.021,0.035,0.021,0.101,0.065,0.027,0.035,0.101,0.547,0.0],[0.02,0.006,0.008,0.005,0.012,0.12,0.013,0.006,0.015,0.017,0.777]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.377,0.621,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.124,0.52,0.357,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.188,0.165,0.137,0.512,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.293,0.26,0.046,0.108,0.293,0.0,0.0,0.0,0.0,0.0,0.0],[0.152,0.118,0.052,0.042,0.195,0.439,0.0,0.0,0.0,0.0,0.0],[0.189,0.332,0.024,0.054,0.084,0.157,0.157,0.0,0.0,0.0,0.0],[0.147,0.275,0.019,0.102,0.074,0.13,0.095,0.157,0.0,0.0,0.0],[0.172,0.151,0.081,0.081,0.071,0.104,0.043,0.076,0.22,0.0,0.0],[0.225,0.121,0.036,0.03,0.055,0.198,0.043,0.073,0.106,0.113,0.0],[0.27,0.058,0.023,0.025,0.066,0.119,0.041,0.041,0.025,0.047,0.285]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.072,0.93,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.015,0.056,0.93,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.017,0.042,0.053,0.887,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.019,0.01,0.028,0.071,0.871,0.0,0.0,0.0,0.0,0.0,0.0],[0.01,0.009,0.005,0.007,0.021,0.945,0.0,0.0,0.0,0.0,0.0],[0.007,0.008,0.002,0.001,0.006,0.024,0.949,0.0,0.0,0.0,0.0],[0.004,0.003,0.003,0.001,0.001,0.003,0.111,0.871,0.0,0.0,0.0],[0.003,0.003,0.001,0.004,0.002,0.002,0.004,0.003,0.977,0.0,0.0],[0.003,0.001,0.003,0.001,0.004,0.003,0.003,0.004,0.024,0.953,0.0],[0.008,0.001,0.001,0.001,0.002,0.043,0.004,0.002,0.003,0.007,0.93]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.363,0.637,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.142,0.264,0.594,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.113,0.145,0.348,0.395,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.147,0.157,0.201,0.201,0.293,0.0,0.0,0.0,0.0,0.0,0.0],[0.141,0.124,0.141,0.117,0.141,0.338,0.0,0.0,0.0,0.0,0.0],[0.113,0.106,0.094,0.068,0.106,0.164,0.348,0.0,0.0,0.0,0.0],[0.124,0.075,0.075,0.046,0.071,0.116,0.231,0.262,0.0,0.0,0.0],[0.064,0.047,0.044,0.028,0.034,0.068,0.068,0.073,0.574,0.0,0.0],[0.104,0.071,0.063,0.037,0.061,0.081,0.091,0.091,0.182,0.22,0.0],[0.134,0.041,0.041,0.029,0.035,0.111,0.071,0.063,0.071,0.063,0.342]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.32,0.68,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.134,0.303,0.562,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.158,0.116,0.295,0.43,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.106,0.061,0.187,0.226,0.42,0.0,0.0,0.0,0.0,0.0,0.0],[0.124,0.033,0.046,0.023,0.181,0.594,0.0,0.0,0.0,0.0,0.0],[0.097,0.022,0.012,0.007,0.046,0.383,0.434,0.0,0.0,0.0,0.0],[0.095,0.02,0.008,0.002,0.014,0.156,0.33,0.375,0.0,0.0,0.0],[0.112,0.056,0.027,0.022,0.02,0.093,0.06,0.039,0.57,0.0,0.0],[0.179,0.042,0.07,0.026,0.024,0.079,0.037,0.029,0.157,0.355,0.0],[0.102,0.014,0.021,0.007,0.034,0.179,0.018,0.01,0.02,0.108,0.486]],[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.438,0.562,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.332,0.242,0.426,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.173,0.099,0.196,0.531,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.226,0.078,0.137,0.165,0.395,0.0,0.0,0.0,0.0,0.0,0.0],[0.152,0.05,0.077,0.06,0.162,0.5,0.0,0.0,0.0,0.0,0.0],[0.138,0.039,0.039,0.025,0.101,0.33,0.33,0.0,0.0,0.0,0.0],[0.176,0.032,0.032,0.015,0.061,0.187,0.271,0.226,0.0,0.0,0.0],[0.123,0.065,0.062,0.054,0.054,0.167,0.054,0.042,0.377,0.0,0.0],[0.139,0.032,0.054,0.032,0.042,0.277,0.062,0.045,0.102,0.215,0.0],[0.132,0.034,0.049,0.018,0.032,0.124,0.038,0.038,0.019,0.055,0.461]]],"model":"Qwen/Qwen3.5-9B"}</script>

<div id="attn-demo" class="attn">
  <div class="attn-head">
    <strong>真实的注意力权重</strong>
    <span class="attn-sub">Qwen3.5-9B 跑一次前向导出来的真实权重</span>
  </div>
  <div class="attn-ctl"><span class="attn-lbl">第几层</span><span id="attn-layers"></span></div>
  <div class="attn-ctl"><span class="attn-lbl">第几个头</span><span id="attn-heads"></span></div>
<div id="attn-note" class="attn-note"></div>
  <div class="attn-grid-wrap"><table id="attn-grid" class="attn-grid"></table></div>
  <div id="attn-say" class="attn-say">把鼠标放到格子上，或点左边的 token。</div>
</div>

<style>
/* 深浅两套颜色都写出来：靠 var(--bs-*) 兜底的话，深色主题下没定义的变量会
   落回浅色值，格子边框在深底上直接发白。 */
.attn { --a-line: #d7dbe0; --a-cell: #e6e9ec; --a-sub: #6b7280; --a-green: #76B900;
  border: 1px solid var(--a-line); border-radius: 10px; padding: 16px 18px; margin: 1.2rem 0; }
body.quarto-dark .attn { --a-line: #3a404a; --a-cell: #333a44; --a-sub: #9aa1a9; --a-green: #8ed000; }
.attn-head strong { font-size: 1.02rem; }
.attn-sub, .attn-lbl, .attn-note { color: var(--a-sub); }
.attn-sub { font-size: .86rem; margin-left: .5rem; }
.attn-ctl { margin-top: 10px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.attn-lbl { font-size: .86rem; margin-right: 4px; }
.attn-note { margin-top: 8px; font-size: .82rem; }
.attn button { font: inherit; font-size: .82rem; padding: 2px 9px; border-radius: 6px;
  border: 1px solid var(--a-line); background: transparent; color: inherit; cursor: pointer; }
.attn button:hover { border-color: var(--a-green); }
.attn button[aria-pressed="true"] { background: var(--a-green); border-color: var(--a-green); color: #14180f; }
.attn-grid-wrap { overflow-x: auto; margin-top: 14px; }
.attn-grid { border-collapse: collapse; font-size: .8rem; }
.attn-grid th { font-weight: 500; color: var(--a-sub); padding: 2px 6px; white-space: nowrap; }
.attn-grid th.rowlab { text-align: right; cursor: pointer; }
.attn-grid th.rowlab:hover, .attn-grid tr.on th.rowlab { color: var(--a-green); font-weight: 700; }
.attn-grid td { width: 30px; height: 30px; border: 1px solid var(--a-cell); }
.attn-grid td.cell { cursor: crosshair; }
.attn-grid tr.on td.cell { outline: 1px solid var(--a-green); }
.attn-say { margin-top: 12px; font-size: .9rem; min-height: 1.5em; }
.attn-say b { color: var(--a-green); }
</style>

<script>
(function () {
  var box = document.getElementById("attn-demo");
  if (!box) return;
  var node = document.getElementById("attn-data");
  if (!node) { document.getElementById("attn-say").textContent = "注意力数据没有嵌进页面。"; return; }
  (function (d) {
    var layer = d.layers[2], head = "mean";   // 默认第 11 层：指代看得最清楚的一层
    var Lbtns = document.getElementById("attn-layers");
    var Hbtns = document.getElementById("attn-heads");
    var grid = document.getElementById("attn-grid");
    var say = document.getElementById("attn-say");

    function matrix() {
      if (head === "mean") return d.mean[layer];
      return d.heads[+head];
    }
    function color(v) { return "rgba(118, 185, 0, " + Math.pow(v, 0.55).toFixed(3) + ")"; }

    function draw() {
      var m = matrix(), T = d.tokens, n = T.length, html = "<tr><th></th>";
      for (var j = 0; j < n; j++) html += "<th>" + T[j] + "</th>";
      html += "</tr>";
      for (var i = 0; i < n; i++) {
        html += '<tr data-i="' + i + '"><th class="rowlab">' + T[i] + "</th>";
        for (var j2 = 0; j2 < n; j2++) {
          if (j2 > i) { html += "<td></td>"; continue; }
          var v = m[i][j2];
          html += '<td class="cell" data-i="' + i + '" data-j="' + j2 + '" data-v="' + v +
                  '" style="background:' + color(v) + '"></td>';
        }
        html += "</tr>";
      }
      grid.innerHTML = html;
      Array.prototype.forEach.call(Lbtns.children, function (b) {
        b.setAttribute("aria-pressed", b.dataset.v === layer);
      });
      Array.prototype.forEach.call(Hbtns.children, function (b) {
        b.setAttribute("aria-pressed", b.dataset.v === head);
      });
      document.getElementById("attn-note").textContent =
        head === "mean" ? "第 " + layer + " 层，16 个头的平均"
                        : "第 " + d.head_layer + " 层的第 " + head + " 个头（逐头数据只导了这一层）";
    }

    function tell(i, j) {
      var m = matrix(), T = d.tokens;
      var row = m[i].slice(0, i + 1);
      var best = row.indexOf(Math.max.apply(null, row));
      var s = "「<b>" + T[i] + "</b>」这个位置：";
      if (j !== undefined) s += "看「<b>" + T[j] + "</b>」的权重 <b>" + m[i][j].toFixed(3) + "</b>；";
      s += "看得最重的是「<b>" + T[best] + "</b>」（" + row[best].toFixed(3) + "），一行加起来等于 1。";
      say.innerHTML = s;
    }

    d.layers.forEach(function (L) {
      var b = document.createElement("button");
      b.textContent = L; b.dataset.v = L;
      b.onclick = function () { layer = L; if (L !== String(d.head_layer)) head = "mean"; draw(); };
      Lbtns.appendChild(b);
    });
    ["mean"].concat(d.heads.map(function (_, i) { return String(i); })).forEach(function (H) {
      var b = document.createElement("button");
      b.textContent = H === "mean" ? "平均" : H; b.dataset.v = H;
      b.onclick = function () { head = H; if (H !== "mean") layer = String(d.head_layer); draw(); };
      Hbtns.appendChild(b);
    });

    grid.addEventListener("mouseover", function (e) {
      var td = e.target.closest("td.cell"); if (!td) return;
      Array.prototype.forEach.call(grid.rows, function (r) { r.classList.remove("on"); });
      grid.rows[+td.dataset.i + 1].classList.add("on");
      tell(+td.dataset.i, +td.dataset.j);
    });
    grid.addEventListener("click", function (e) {
      var th = e.target.closest("th.rowlab"); if (!th) return;
      var i = +th.parentNode.dataset.i;
      Array.prototype.forEach.call(grid.rows, function (r) { r.classList.remove("on"); });
      grid.rows[i + 1].classList.add("on");
      tell(i);
    });

    draw();
  })(JSON.parse(node.textContent));
})();
</script>

```

三件值得注意的事：

1. **指代确实出现了，但只在中间几层。** 「它」那一行在**第 11 层**有 0.44 的权重压在「书」上，第 7 层和第 15 层也在 0.2 上下；到第 23 层之后就转到「，」和「小明」去了。
2. **越靠后的层越爱盯标点和句首。** 这不是这个模型的怪癖，是被反复观察到的现象，叫**注意力沉没**（attention sink）：模型需要一个“没什么信息但总是在那儿”的位置，把用不掉的注意力权重倒进去[^sink]。
3. **同一层里不同的头看的东西不一样。** 切到第 31 层再逐头看：头 3 最看重「打开了」，头 0 最看重「小明」，头 1 给「书」的权重最高。这就是 D.3 里说的“不同的头可以盯不同的关系”，但请注意这是**一次观察**，不是这些头的定义。

::: {.callout-warning title="注意力权重不等于“重要性”"}
看得多不等于因果上重要——拿注意力权重当模型的解释，学界有明确的争论和反例[^attnexp]。这里的用法是安全的：我们只是在确认那个 $T\times T$ 矩阵长什么样、因果掩码确实让上三角为零。
:::

### 代价

那个 $T\times T$ 的分数矩阵是关键：**序列长度翻倍，注意力的计算量变四倍**。长上下文贵就贵在这里，后面一整条优化线（D.8）都在跟它较劲。

**分组查询注意力（GQA）。** 让多个查询头共用一组 $K$、$V$[^gqa]。Qwen3.5-9B 是 16 个查询头配 4 组 KV，每头 256 维——查询还是 16 份，但要存下来的键值只有四分之一。省的是什么见 D.8。

## D.4 前馈网络：每个位置各自变换

交换完信息，每个位置再各自过一个两层的小网络，叫**前馈网络**（feed-forward network，缩写 FFN）。经典写法是 $\max(0, xW_1)W_2$；现在常用的是 **SwiGLU**[^glu]：

$$
\text{FFN}(x) = \big(\operatorname{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\big)W_{\text{down}},
\qquad \operatorname{SiLU}(z) = z\,\sigma(z)
$$

$\odot$ 是逐元素相乘（element-wise product），$\sigma$ 是 sigmoid。两条并行的线性变换，一条经过 SiLU 之后当作“门”去缩放另一条，再投回 $d$ 维。中间那个维度（Qwen3.5-9B 是 12288，即 $3d$）叫 FFN 中间维。

**这里是参数最多的地方。** 一层 FFN 有 $3 \times 4096 \times 12288 = 150\,994\,944$ 个参数，而一层全注意力的四个矩阵加起来只有 $58\,720\,256$ 个。整个模型的参数量可以这样加出来：

| 部分 | 每个多少参数 | 个数 | 小计 |
|---|---|---|---|
| `embed_tokens` | $248320 \times 4096$ | 1 | 1 017 118 720 |
| 全注意力块（注意力 + FFN） | 58 720 256 + 150 994 944 | 8 | 1 677 721 600 |
| 线性注意力块（mixer + FFN） | 67 371 008 + 150 994 944 | 24 | 5 240 782 848 |
| `lm_head` | $4096 \times 248320$ | 1 | 1 017 118 720 |
| **合计** | | | **8 952 741 888** |

模型自己报的是 8 997 081 600，差 44 M，其中 43.3 M 是 Day 00 挂上去的 LoRA 参数，剩下约 1 M 是各处的归一化权重和线性注意力（linear attention）里的一维卷积——这些模块每个只有几千个参数，但它们不是矩阵乘法，LoRA 挂不上去。

## D.5 归一化：让每层拿到的数值范围稳定

每个模块之前会先做一次归一化。**LayerNorm** 减均值除标准差再缩放；**RMSNorm** 省掉减均值这一步[^rmsnorm]：

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \mathbf{g}
$$

$\mathbf{g}\in\mathbb{R}^d$ 是可学习的缩放向量，$\epsilon$ 防止除零。少一步减均值，省一点计算，效果基本不变——现在的大模型基本都用它。

**放在模块之前还是之后**（pre-norm / post-norm）不是小事：原论文是 post-norm，需要小心的学习率（learning rate）预热才能收敛；pre-norm 把归一化挪到残差分支里面，训练稳定得多，现在几乎都用 pre-norm[^prenorm]。这就是 [Day 00 §2.9](../days/day00_lora-quickstart/README.md) 那张块结构图里，归一化画在 mixer 和 FFN **前面**的原因。

## D.6 位置信息：RoPE

注意力的公式里没有任何地方用到“第几个位置”——把输入的行打乱，输出也只是跟着打乱。所以位置必须显式喂进去。

现在的主流做法是 **RoPE**（旋转位置编码）[^rope]：把 $Q$、$K$ 的每两维看成一个复数（平面上的一个向量），按位置 $m$ 旋转一个角度 $m\theta$。这样两个位置的内积只依赖它们的**相对距离**，而不是绝对位置。它不增加参数，也不改变形状，只是在算注意力之前对 $Q$、$K$ 各做一次旋转。

推导见原论文 §3.4；这里只需要知道两件事：位置信息加在 $Q$、$K$ 上（不是加在输入 embedding 上），以及模型能处理多长的上下文由训练时见过的位置范围决定（Qwen3.5-9B 是 262144）。

## D.7 从最后一层到下一个 token

前面几节都在讲一个块里发生什么。现在把整条路串起来——这一步很多教程会跳过，但不讲清楚，前面的东西就落不了地。

**形状自始至终没变。** 输入 $T$ 个 token，embedding 之后是一个 $T\times d$ 的矩阵；32 个块每个都在往这些行上加东西（D.2），出来还是 $T\times d$。**每一行仍然对应输入里的一个位置**，只不过它现在装的不只是这个 token 自己的意思，还混进了它从前面各位置取来的信息。

**只有最后一行用来预测下一个 token。** 因果掩码保证第 $i$ 行只看得到前 $i$ 个位置，所以第 $i$ 行代表的是“读完前 $i$ 个 token 之后的状态”。要预测第 $T+1$ 个 token，取的就是第 $T$ 行：

$$
\mathbf{h}_T \in \mathbb{R}^{d} \;\xrightarrow{\ \text{RMSNorm}\ }\;
\tilde{\mathbf{h}}_T \;\xrightarrow{\ \times W_{\text{lm\_head}}\ }\;
\mathbf{z} \in \mathbb{R}^{V}
$$

$W_{\text{lm\_head}}$ 是一个 $d \times V$ 的矩阵（Qwen3.5-9B 是 $4096 \times 248320$）。这一步做的事很朴素：**拿这个 4096 维向量和词表里每个词的一列做内积**，得到 248320 个分数。分数高的词，就是模型认为接下来该出现的词。再按 D.1 的 softmax 变成概率、采样，就得到下一个 token。

串起来是这样（对应 [Day 00 §2.9](../days/day00_lora-quickstart/README.md) 那张主干图）：

$$
\text{token ids} \to \text{embedding} \to \underbrace{\text{块} \to \cdots \to \text{块}}_{32\ \text{层}} \to \text{RMSNorm} \to \text{lm\_head} \to \text{logits} \to \text{softmax} \to \text{采样}
$$

**那前面 $T-1$ 行呢？** 推理时用不上，但训练时全都要用：第 $i$ 行的 logits 用来预测第 $i+1$ 个 token，一次前向就产生了 $T$ 个训练信号。这就是为什么 [Day 00 §3.3](../days/day00_lora-quickstart/README.md) 的掩码是逐位置的 0/1 数组——它决定这 $T$ 个信号里哪些进 loss。

**一个常见的误解。** 模型不是“一次想好整句话再吐出来”。它每次只产生一个 token 的概率分布，采样出来的 token 接到输入末尾，然后**整条路重新走一遍**。你在聊天界面看到的逐字蹦出来，就是这个循环在跑，而不是打字机效果。

## D.8 推理是两个阶段：prefill 与 decode

回到 D.1 的那句话：每生成一个 token，模型要从头跑一遍。如果每次都把整个前缀重算一遍，第 $n$ 个 token 就要算 $n$ 次注意力，总量随长度平方增长。

但注意到：位置 $j$ 的 $K_j$、$V_j$ 只依赖 $x_j$，**和后面生成什么无关**。所以算过一次就可以存起来，这就是 **KV cache**。于是推理分成两个阶段：

- **prefill**：把用户输入的 $T$ 个 token 一次性喂进去，算出所有位置的 K、V 存好，并产出第一个输出 token。这一步是计算密集的（大矩阵乘法）。
- **decode**：之后每一步只算**一个**新位置的 Q、K、V，读取缓存里已有的 K、V 做注意力，再把新的 K、V 追加进去。这一步是访存密集的（矩阵乘向量，算得少读得多）。

```{=html}
<svg class="fig light-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 494" role="img" aria-label="prefill 一次算完整个输入并把 K、V 存进缓存；decode 每步只算一个新位置，读缓存再追加"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#6B7280"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >推理的两个阶段与 KV cache</text><text x="30" y="76" class="t2t-s" font-size="16.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >prefill：把输入的 5 个 token 一次算完</text><text x="30" y="100" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >大矩阵乘法，吃算力</text><rect x="30" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="56.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K1</text><rect x="90" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="116.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K2</text><rect x="150" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="176.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K3</text><rect x="210" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="236.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K4</text><rect x="270" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="296.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K5</text><text x="340" y="142" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >K、V 存进缓存</text><text x="30" y="214" class="t2t-s" font-size="16.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >decode：每步只算 1 个新位置</text><text x="30" y="238" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >矩阵乘向量，算得少、读得多，吃带宽</text><rect x="30" y="254" width="52" height="40" rx="6" fill="#FFFFFF" stroke="#D7DBE0" stroke-width="1.2" /><text x="56.0" y="280" class="t2t-m" font-size="12.8" fill="#6B7280" text-anchor="middle" font-weight="400" >K1</text><rect x="90" y="254" width="52" height="40" rx="6" fill="#FFFFFF" stroke="#D7DBE0" stroke-width="1.2" /><text x="116.0" y="280" class="t2t-m" font-size="12.8" fill="#6B7280" text-anchor="middle" font-weight="400" >K2</text><rect x="150" y="254" width="52" height="40" rx="6" fill="#FFFFFF" stroke="#D7DBE0" stroke-width="1.2" /><text x="176.0" y="280" class="t2t-m" font-size="12.8" fill="#6B7280" text-anchor="middle" font-weight="400" >K3</text><rect x="210" y="254" width="52" height="40" rx="6" fill="#FFFFFF" stroke="#D7DBE0" stroke-width="1.2" /><text x="236.0" y="280" class="t2t-m" font-size="12.8" fill="#6B7280" text-anchor="middle" font-weight="400" >K4</text><rect x="270" y="254" width="52" height="40" rx="6" fill="#FFFFFF" stroke="#D7DBE0" stroke-width="1.2" /><text x="296.0" y="280" class="t2t-m" font-size="12.8" fill="#6B7280" text-anchor="middle" font-weight="400" >K5</text><rect x="330" y="254" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="356.0" y="280" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K6</text><text x="400" y="280" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >读已有的 5 个，追加第 6 个</text><text x="30" y="320" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >绿色虚线 = 这一步新算出来的；灰色实线 = 缓存里已经有的</text><text x="30" y="348" class="t2t-s" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >重算一遍要 O(T²)，缓存下来每步只要 O(T)——代价是显存里多一块随长度增长的缓存。</text><rect x="30" y="368" width="690" height="96" rx="10" fill="#F5F6F8" stroke="#D7DBE0" stroke-width="1.2" /><text x="48" y="398" class="t2t-s" font-size="16.0" fill="#1A1A1A" text-anchor="start" font-weight="700" >每 token 每层要存多少</text><text x="48" y="424" class="t2t-m" font-size="14.1" fill="#6B7280" text-anchor="start" font-weight="400" >2（K 和 V）× KV 头数 × 每头维度 × 每个数的字节数</text><text x="48" y="448" class="t2t-s" font-size="12.8" fill="#6B7280" text-anchor="start" font-weight="400" >Qwen3.5-9B 全注意力层：2 × 4 × 256 × 2 = 4 KB；32 层里只有 8 层是全注意力，所以每 token 32 KB</text></svg>
```
```{=html}
<svg class="fig dark-content" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 494" role="img" aria-label="prefill 一次算完整个输入并把 K、V 存进缓存；decode 每步只算一个新位置，读缓存再追加"><style>.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}</style><defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0.8 8 4.5 0 8.2z" fill="#9AA1A9"/></marker></defs><text x="30" y="34" class="t2t-s" font-size="17.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >推理的两个阶段与 KV cache</text><text x="30" y="76" class="t2t-s" font-size="16.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >prefill：把输入的 5 个 token 一次算完</text><text x="30" y="100" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >大矩阵乘法，吃算力</text><rect x="30" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="56.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K1</text><rect x="90" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="116.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K2</text><rect x="150" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="176.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K3</text><rect x="210" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="236.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K4</text><rect x="270" y="116" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="296.0" y="142" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K5</text><text x="340" y="142" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >K、V 存进缓存</text><text x="30" y="214" class="t2t-s" font-size="16.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >decode：每步只算 1 个新位置</text><text x="30" y="238" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >矩阵乘向量，算得少、读得多，吃带宽</text><rect x="30" y="254" width="52" height="40" rx="6" fill="#242830" stroke="#3A404A" stroke-width="1.2" /><text x="56.0" y="280" class="t2t-m" font-size="12.8" fill="#9AA1A9" text-anchor="middle" font-weight="400" >K1</text><rect x="90" y="254" width="52" height="40" rx="6" fill="#242830" stroke="#3A404A" stroke-width="1.2" /><text x="116.0" y="280" class="t2t-m" font-size="12.8" fill="#9AA1A9" text-anchor="middle" font-weight="400" >K2</text><rect x="150" y="254" width="52" height="40" rx="6" fill="#242830" stroke="#3A404A" stroke-width="1.2" /><text x="176.0" y="280" class="t2t-m" font-size="12.8" fill="#9AA1A9" text-anchor="middle" font-weight="400" >K3</text><rect x="210" y="254" width="52" height="40" rx="6" fill="#242830" stroke="#3A404A" stroke-width="1.2" /><text x="236.0" y="280" class="t2t-m" font-size="12.8" fill="#9AA1A9" text-anchor="middle" font-weight="400" >K4</text><rect x="270" y="254" width="52" height="40" rx="6" fill="#242830" stroke="#3A404A" stroke-width="1.2" /><text x="296.0" y="280" class="t2t-m" font-size="12.8" fill="#9AA1A9" text-anchor="middle" font-weight="400" >K5</text><rect x="330" y="254" width="52" height="40" rx="6" fill="none" stroke="#76B900" stroke-width="2" stroke-dasharray="5 4"/><text x="356.0" y="280" class="t2t-m" font-size="12.8" fill="#76B900" text-anchor="middle" font-weight="400" >K6</text><text x="400" y="280" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >读已有的 5 个，追加第 6 个</text><text x="30" y="320" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >绿色虚线 = 这一步新算出来的；灰色实线 = 缓存里已经有的</text><text x="30" y="348" class="t2t-s" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >重算一遍要 O(T²)，缓存下来每步只要 O(T)——代价是显存里多一块随长度增长的缓存。</text><rect x="30" y="368" width="690" height="96" rx="10" fill="#1E222A" stroke="#3A404A" stroke-width="1.2" /><text x="48" y="398" class="t2t-s" font-size="16.0" fill="#E8E8E8" text-anchor="start" font-weight="700" >每 token 每层要存多少</text><text x="48" y="424" class="t2t-m" font-size="14.1" fill="#9AA1A9" text-anchor="start" font-weight="400" >2（K 和 V）× KV 头数 × 每头维度 × 每个数的字节数</text><text x="48" y="448" class="t2t-s" font-size="12.8" fill="#9AA1A9" text-anchor="start" font-weight="400" >Qwen3.5-9B 全注意力层：2 × 4 × 256 × 2 = 4 KB；32 层里只有 8 层是全注意力，所以每 token 32 KB</text></svg>
```

**缓存要多大。** 每个 token、每层要存 K 和 V 各一份：

$$
\text{每 token 每层字节数} = 2 \times n_{\text{kv}} \times d_{\text{head}} \times \text{dtype 字节数}
$$

代入 Qwen3.5-9B 的全注意力层（$n_{\text{kv}} = 4$，$d_{\text{head}} = 256$，bf16 占 2 字节）：$2\times4\times256\times2 = 4096$ 字节 = 4 KB。这个模型 32 层里只有 8 层是全注意力，所以每个 token 一共 32 KB；1000 个 token 约 32 MB，跑满 262144 的上下文约 8.4 GB。

作为对照：如果 32 层全是标准多头注意力（16 组 KV），每 token 每层是 $2\times16\times256\times2 = 16$ KB，32 层就是 512 KB，**是前者的 16 倍**。GQA 和混合注意力省下的就是这个。day 03 会把这个公式和实测显存对上账。

（线性注意力层不存逐 token 的 K、V，它维持的是一个固定大小的状态，不随长度增长——这正是混合架构的动机。）

## D.9 后来的扩展与优化

上面是一个块的基本形态。工业上用的模型在此之上有一堆改动，每一条都在解决一个具体的瓶颈。下表按“解决什么问题”排，最后一列是课表里会动手做的地方。

| 改动 | 解决什么 | 代价 | 课表位置 |
|---|---|---|---|
| MQA / GQA[^mqa][^gqa] | KV cache 太大 | 表达能力略降 | day 03 |
| 滑窗 / 线性注意力 / 混合架构 | 长序列的 $T^2$ 和缓存增长 | 远距离信息可能丢 | day 03、本页 D.8 |
| FlashAttention[^flash] | 注意力的显存带宽瓶颈 | 实现复杂、依赖硬件 | day 20 |
| 连续批处理 + PagedAttention[^paged] | 吞吐低、缓存碎片 | 调度复杂度 | day 04 |
| 量化到 INT4 / NVFP4 | 权重和带宽 | 精度损失需要评测 | day 08、附录 C |
| 混合专家（MoE）[^moe] | 想要更多参数但不想更多计算 | 显存占用不降、路由不均衡 | day 07 |
| LoRA[^lora] | 全参微调显存放不下 | 只能表示低秩的更新 | day 00、day 31 |
| 投机解码[^spec] | decode 阶段一次只出一个 token | 需要一个草稿模型 | day 09 |

这张表不需要现在读懂，等做到对应那天回来看一眼就行。

## D.10 想再深入

- **代码路线**：Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 是最短的一条“看完能自己写一个”的路，约 300 行训练代码。day 26 会手敲一遍。
- **论文路线**：先读 Transformer 原论文[^attn] §3，再读 RoPE[^rope] §3、GQA[^gqa] §2，就够看懂现在大部分开源模型的结构了。

## 参考文献

[^sink]: Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024，§3 对“初始 token 吸走大量注意力”的观察与解释。[arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
[^attnexp]: Jain & Wallace, *Attention is not Explanation*, NAACL 2019，§4 给出注意力权重与模型输出不一致的构造。[arXiv:1902.10186](https://arxiv.org/abs/1902.10186)
[^attn]: Vaswani et al., *Attention Is All You Need*, NeurIPS 2017，§3.2 注意力定义、§3.2.1 缩放、§3.2.2 多头。[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
[^resnet]: He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016，§3.1 残差学习的动机。[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
[^gqa]: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023，§2 方法与质量/速度权衡。[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
[^mqa]: Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019，§2 多查询注意力。[arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
[^glu]: Shazeer, *GLU Variants Improve Transformer*, 2020，§2 SwiGLU 定义与实验。[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
[^rmsnorm]: Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019，§3 定义。[arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
[^prenorm]: Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020，§3 pre-norm 与 post-norm 的梯度分析。[arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
[^rope]: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021，§3.4 旋转位置编码。[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
[^flash]: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022，§3 分块与重算。[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
[^paged]: Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023，§4 PagedAttention。[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
[^moe]: Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022，§2 路由。[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
[^lora]: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022，§4 方法。[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
[^spec]: Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023，§2 算法与接受率。[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
