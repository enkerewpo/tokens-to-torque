# ROADMAP — 72 天

顺序：serving → CUDA → training → VLM → VLA → WAM。每周 6 天（5 天推进 + 1 天整合日），第 7 天休息。每天格式：**目标 / 动手 / 产出**。产出栏里的东西没拿到，这天就不算过。

Day 编号和 `days/dayNN_topic-name/` 目录一一对应。总览和进度见 [README](README.md)。

## 你需要什么硬件

**课表不绑定某一块板子。** 下面的任务描述里凡是写“你的板子”的地方，换成你手上任何一块 CUDA 设备都成立；只在少数几处（统一内存、`nvpmodel` 功耗墙、NVFP4 部署）才真的需要 Jetson，那些会标 `[Jetson]`。教程里的实测数字来自 Jetson AGX Thor，标 `[Thor]`——你自己的数会不一样，这正是要你亲手测的原因。

| 你有的 | 建议怎么走 |
|---|---|
| **Jetson AGX Thor / Orin**（32–128 GB 统一内存） | 全程照做 |
| **Jetson Orin Nano / NX**（8–16 GB） | Phase 1–4 换 1–4B 模型；Phase 5 用 SmolVLA 这类小 VLA；Phase 6 只跑最小的 world model |
| **独显 ≥ 24 GB** | Phase 1–4 完全没问题，Phase 5–6 大部分能跑。跳过所有 `[Jetson]` 小节 |
| **独显 12–16 GB** | Phase 1–3 照做（模型降到 4B 级 + QLoRA）；Phase 4 起挑着跑 |
| **只有 CPU 或 Mac** | 概念、源码、公式全都能跟。实验部分改成读别人的数字并复核其合理性——这本身是门手艺 |

显存不够时的通用替换：**模型降一档**（9B → 4B → 1.5B）、**上量化**（bf16 → NVFP4/INT4）、**减 batch 加梯度累积**、**开 gradient checkpointing**。这四招在 Phase 3 会正经讲。

---

# Day 00 · 先训一个（不等，今晚或这周末，2 小时）

排在最前面是因为：**先有一个自己训出来的模型，后面 12 周的理论才有落点。**

| | 目标 | 动手 | 产出 |
|---|---|---|---|
| **00** | 在你的板子上 LoRA 微调一个小模型，让它用你的语气说话 | 1. 攒 200–500 条你自己写的文本（commit message、笔记、给同学的解释、论文段落都行，越“你”越好）<br>2. 转成 SFT 格式（instruction → response）<br>3. 跟 Jetson AI Lab 的 finetune 教程，LoRA 一个 4B 级模型，1–2 epoch<br>4. 和 base 模型问同一个问题，对比 | 一个 adapter 文件 + 一组“before/after”对话对比 |

技术上不用懂原理，照着跑就行——Phase 3 会回来把每一步拆开讲。`[Jetson]` 预检 + 遥测照常。这个 job 大概 20–40 分钟，温度不会有事。

---

# 🌱 Side quest：训一个自己用的模型（贯穿 12 周，每周末 1–2 小时）

目标：**12 周后，有一个跑在自己机器上、离线、用自己的数据微调过、每天真的会用的模型。**

> 这条支线用的是**个人数据**。所有语料、数据集、checkpoint 一律留在本地 `private/`（已 gitignore），
> 仓库里只放方法和跑通后的结论。

| 时间 | 任务 |
|---|---|
| 第 1–2 周 | 攒数据。写脚本自动收集自己写的文本（commit、笔记、摘要）。**只收集，不整理。** |
| 第 3–4 周 | 清洗 + 去重 + 转 SFT 格式。顺便学会看数据分布（长度、主题、质量）。 |
| 第 5 周 | 用 Phase 3 学到的从零训练知识，训一个 tiny 模型在你的数据上过拟合一次——**看它开始模仿你说话的那一刻**。 |
| 第 6 周 | 正式 LoRA。和 day 00 那个对比，看数据量/质量带来多少提升。 |
| 第 7–8 周 | 加视觉：让它能看你的实验图、jtop 截图、3D 场景预览图，说人话。 |
| 第 9–10 周 | 接上工具：让它能查你的 notes/、跑 bench 脚本、读机器温度。变成一个真 agent。 |
| 第 11–12 周 | 常驻你的机器，开机自启，每天用。写一篇“训了个自己用的模型”的复盘。 |

这条支线的规矩：**允许烂，不允许停。** 每周末哪怕只跑通一行也算。

---

# Week 1 · 服务化基础：把模型跑起来，并且测准  `day 01–06`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **01** | 有一个能用的 vLLM | `[Jetson]` 装 JetPack 对应的 NGC vLLM wheel 或容器；独显直接 `pip install vllm`；起一个 4B 级模型的 OpenAI 兼容 server；`curl` 通一次 | 启动命令写进 `code/serve.sh`；第一条 latency 数字 |
| **02** | 一个请求在 vLLM 里经历了什么 | 读 engine 主循环 + scheduler，跟一个请求从 add_request 到 output | 一张手画的请求路径图 + 3 个关键类名 |
| **03** | KV cache 到底占多少显存 | 手算 KV cache 公式（layers × heads × head_dim × 2 × dtype × seq × batch）；改 `--max-model-len` / `--gpu-memory-utilization` 三组，验证 | 预测 vs 实测对照表，误差 < 15% |
| **04** | 连续批处理为什么快 | 并发 1/4/16/64 各跑一轮，拆出 TTFT 和 TPOT | throughput-latency 帕累托曲线一张 |
| **05** | 会做可复现的 benchmark | 用 `vllm bench serve`，固定 seed / 输入长度分布 / warmup；同一配置跑三遍看方差 | `code/bench.sh` + `results.csv`，三遍方差 < 5% |
| **06** | 整合 | 回答一个问题：**你这块卡上，这个模型的“可用工作点”在哪（并发多少、延迟多少、温度多少）** | 一页结论，含温度曲线 |

---

# Week 2 · 服务化进阶：量化、投机解码、多模型共存  `day 07–12`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **07** | 搞清量化格式全景 | FP8 / NVFP4 / AWQ / GPTQ / INT4 各自压什么、Blackwell 上哪些走 Tensor Core | 一张“格式 × 是否真加速 × 精度代价”表 |
| **08** | 亲手量化一个模型 | `[Jetson]` TensorRT Edge-LLM（量化 → ONNX → engine）；独显走 AWQ / GPTQ / FP8 | 量化前后：显存、吞吐、一个小评测集的精度差 |
| **09** | 投机解码 | 跟 Jetson AI Lab 的 speculative decoding 教程（MTP / DFlash / DSpark）；测 accept rate | accept rate 与实际加速比的关系曲线 |
| **10** | **边缘多模型共存** | 同一块卡上同时跑 LLM + VLM + 一个感知模型，测互相干扰 | 干扰矩阵：单跑 vs 混跑的延迟劣化 % |
| **11** | prefix caching / RadixAttention | agent 场景（长 system prompt 反复出现）下 SGLang vs vLLM | 命中率 + 省下多少 TTFT |
| **12** | 整合 | 写「边缘推理层需求清单」 | 一页总结 |

---

# Week 3 · CUDA 基础：从 kernel 到 profile  `day 13–18`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **13** | 编程模型 | grid/block/warp；写 saxpy，用你这块卡的 arch 编译（`nvidia-smi --query-gpu=compute_cap`；Thor 是 `sm_110`）；测带宽 | 实测带宽 / 理论带宽 |
| **14** | 内存层次 | naive matmul → tiled matmul（shared memory）；理解 coalescing | 两版本 GFLOPS 对比，加速比 |
| **15** | Nsight 上手 | 用 Nsight Systems 看 day 01 的 vLLM 跑批时间线；Nsight Compute 挖前 3 热点 kernel | 热点清单 + 每个的瓶颈类型（memory / compute / latency） |
| **16** | reduction / scan / warp shuffle | 写 4 个版本的 reduction，逐步优化 | 优化路径表，每步为什么快 |
| **17** | Tensor Core | mma/wmma 概念；跑一个 CUTLASS example 并读懂它在做什么 | 能说清 tile / warp-tile / instruction-tile 三层 |
| **18** | 整合 | 把 day 14 的 matmul 推到 cuBLAS 的百分之多少 | 一个数字 + 差距原因分析 |

---

# Week 4 · CUDA 应用：Triton 与 attention  `day 19–24`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **19** | Triton 入门 | vector add → fused softmax；对比手写 CUDA 的开发成本 | 两个 kernel 跑通 |
| **20** | Triton matmul + autotune | 写 matmul，开 autotune，看它选了什么配置 | 和 day 14 的 tiled CUDA 版对比 |
| **21** | Flash Attention 原理 | tiling + online softmax，为什么不需要存 N×N | 手推 online softmax 递推式 |
| **22** | 改一个 attention kernel | 拿 Triton 版 flash-attn，改一处（比如 causal mask 或 head_dim），和 PyTorch SDPA 对数值 + 对速度 | 数值误差 < 1e-2，速度比 |
| **23** | **你这块卡的 roofline** | 测实际算力/带宽比；判断你关心的算子落在哪边 | roofline 图 + 你的 VLA/VLM 主要算子的定位 |
| **24** | 整合 | 「这块卡上什么 memory-bound、什么 compute-bound」 | 一页，指导后面所有优化决策 |

---

# Week 5 · 训练：从零训一个属于你的小模型 🎉  `day 25–30`

这周的核心体验是**看着一个随机初始化的东西慢慢学会说话**。用小模型学，因为小模型一小时就能跑完一轮，你能真的做实验而不是等。

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **25** | 数据与 tokenizer | 拿自己攒的语料 + 一份公开小语料；训一个 BPE tokenizer；看 vocab 里出现了什么词 | tokenizer + 一句话被切成什么的截图（这一步经常很好笑） |
| **26** | 手写一个 ~20M 的 GPT | 参考 nanoGPT，自己敲 attention / MLP / block / 训练循环（**不要 copy，敲一遍**） | 跑通，loss 从 ~10 开始下降 |
| **27** | 优化器与调度 | AdamW、warmup、cosine、grad clip；做一次 lr sweep（3–5 个值） | loss 曲线族图，找到“炸掉”和“太慢”的边界 |
| **28** | 混合精度与显存 | bf16、gradient checkpointing、梯度累积；每项单独开关测 | 显存-吞吐权衡表（`[Jetson]` 统一内存在这里是优势） |
| **29** | 评估与过拟合 | val loss、perplexity；故意在自己那份小数据上过拟合一次 | **让它模仿你说话，把生成结果贴出来** |
| **30** | 整合 | 一个能生成通顺片段的小模型 + 全部超参和曲线归档 | day30 README + 归档 checkpoint |

---

# Week 6 · 微调：把它变成真能用的  `day 31–36`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **31** | LoRA 原理 | 低秩分解、rank/alpha/target_modules 各自影响什么；为什么省显存 | 能手算一个 LoRA 加了多少参数 |
| **32** | 三个档位都跑一遍 | Jetson AI Lab finetune 教程：Full SFT (4B) / LoRA (9B) / QLoRA (27B)，挑两档实跑（显存不够就只跑 QLoRA 那档） | 三档的显存/时间/效果对比表 |
| **33** | 数据集构造（最被低估的一步） | chat template、loss mask（只对 assistant token 算 loss）、坏样本过滤 | 一个干净的 SFT 数据集 + 你能说清每条为什么留 |
| **34** | **微调到一个真实下游任务** | 挑一个自己真的需要、且通用 LLM 做不好的任务做成 SFT（自选，要有可量化指标） | 微调模型 vs 通用 LLM 的指标 |
| **35** | 评测要有 GT 闭环 | 先用 GT 对 GT 验证评测链（应该 ~100），再评模型 | 可信的评测脚本 |
| **36** | 分布式补课 | FSDP / DeepSpeed ZeRO 三个 stage 各切了什么；在一台多卡机器上跑一次双卡 | 单卡 vs 双卡的 scaling 效率 |

---

# Week 7 · VLM：把眼睛接上去  `day 37–42`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **37** | 架构谱系 | ViT + connector + LLM；三派对比（cross-attn / projector / early fusion） | 一张架构对比图 |
| **38** | 拆一个真模型 | 挑一个开源 VLM，在 forward 里打印每步张量形状 | shape 流水账，能说清图像怎么变成 token |
| **39** | 视觉 token 的代价 | 分辨率 / AnyRes tiling 策略如何影响 token 数；接 day 04 测延迟 | token 数 → 延迟的曲线 |
| **40** | 训一个 connector | 冻结 ViT 和 LLM，只训 projector，小数据集 | 训练前后的描述质量对比 |
| **41** | VLM 评测（挑你关心的） | 空间关系、计数、OCR | 一个小 benchmark 的分数 |
| **42** | 整合 | Cosmos Reason2 8B 在你这块卡上的可用性报告（延迟/显存/精度） | 一页，可直接判断能不能进生产 |

---

# Week 8 · VLM 落到具身系统  `day 43–48`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **43** | VLM 当感知前端 | 和传统检测/分割管线在同一场景上对比 | 对比表（准确率、延迟、显存） |
| **44** | 结构化输出 | JSON schema、constrained decoding（xgrammar / outlines）；测约束带来的延迟代价 | 100% 合法 JSON 的方案 + 开销数字 |
| **45** | 长时序输入 | 视频/多帧、采样策略、KV cache 复用 | 帧数 → 效果/延迟曲线 |
| **46** | VLM-as-judge | 用 VLM 自动评你的 3D 场景重建结果，先用已知答案验证评判器 | 可信的自动评测器 |
| **47** | 延迟预算 | 一个具身 agent 控制循环里，VLM 能占多少 ms？倒推模型规模上限 | 预算表 |
| **48** | 整合 + 给自己的模型接视觉 | 让你的私人模型能看图说话 | demo 一个 |

---

# Week 9 · VLA：动作生成  `day 49–54`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **49** | VLA 谱系 | RT-2 → OpenVLA → π0/π0.5 → GR00T N；action tokenization vs flow matching vs diffusion 三条技术路线 | 一张谱系图 + 三种动作头的优缺点 |
| **50** | 部署 π0.5 | `[Jetson]` 跟 Jetson AI Lab 的 NVFP4 教程；独显用 bf16 或 FP8 | 实测控制频率（Hz）+ 温度曲线 |
| **51** | 部署 GR00T 1.7 | 同上 | 同上，和 π0.5 对比 |
| **52** | 统一评测 | 把 π0.5 / GR00T / 你已有的 OFT-7B、SmolVLA 放同一 LIBERO 评测下 | 四模型同表：成功率 × 频率 × 显存 × 功耗 |
| **53** | action chunking 与异步 | chunk 长度、执行/推理重叠、时序一致性——这是调度层的核心 | chunk 长度 → 成功率/延迟的权衡曲线 |
| **54** | 整合 | 「边缘 VLA 的可行域」一页纸 | 一张图 |

---

# Week 10 · VLA 微调：训一个你自己的策略  `day 55–60`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **55** | LeRobot 数据格式 | 数据集结构、episode/frame、如何转换 | 一个小数据集转换成功 |
| **56** | 微调 GR00T N1.5/1.7 | 跟 Seeed 的 SO-101 教程走一遍（没有真机就用仿真数据） | 微调 loss 曲线 + 一段 rollout 视频 |
| **57** | 微调 SmolVLA 或 π0.5 | 同一数据，换模型 | 两者对比 |
| **58** | 评测微调效果 | 和 day 52 的 base 比，同一 LIBERO 套件 | 提升多少，哪些任务提升哪些退化 |
| **59** | 量化后掉多少 | 微调模型走 NVFP4，测精度损失 | 精度-速度权衡表 |
| **60** | 整合 | 全流程可复现脚本 | `code/vla_finetune.sh` |

---

# Week 11 · WAM：世界动作模型  `day 61–66`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **61** | 世界模型谱系 | Dreamer → Genie → Cosmos → V-JEPA2；WAM 的定义（**联合预测未来状态 + 动作**） | 谱系图 + WAM 与 VLA 的本质差别 |
| **62** | 读 DreamZero / Cosmos Policy | 弄清“video diffusion backbone 直接当 policy”怎么做；动作/未来状态/value 如何编成 latent frame | 两篇的方法对比笔记 |
| **63** | 视频扩散基础 | latent diffusion、flow matching、时序注意力 | 能手推一次 flow matching 的训练目标 |
| **64** | 跑一个小 world model | 可跑规模的（不是 14B）；给定当前帧+动作，预测未来帧 | 预测帧 vs 真实帧的对比图 |
| **65** | **WAM 的推理开销** | 这是边缘部署的核心矛盾：一次“想象”要多少 ms | 开销表 + 能不能进控制回路的结论 |
| **66** | 整合 | 「WAM 在边缘可行吗」一页纸，用数字回答 | 结论 + 依据 |

---

# Week 12 · 整合  `day 67–72`

| Day | 目标 | 动手 | 产出 |
|---|---|---|---|
| **67** | WAM 的用法综述 | 读三篇用 WAM 做规划/评估/数据生成的论文，各一段 | 一页笔记 |
| **68** | 用 world model 评策略 | 复现一篇公开工作；先验证评估器可信度 | 可行性结论 |
| **69** | WAM 的数据管线 | 视频 + 动作怎么切成训练样本；跑通一个公开数据集的转换 | 一个能跑的转换脚本 |
| **70–71** | 最小闭环 demo | 感知 + VLA（动作）+ WAM（想象）跑通一个玩具任务 | 一段视频 |
| **72** | 收官 | 12 周产出清单；自己的模型上线；下一步方向候选 3 个 | 总结 + 模型常驻 |

---

## 弹性规则

- **落后了不补课**，直接跳到今天该做的。补课会让人放弃。
- **某天特别有意思就多花时间**，第二天顺延。课表是脚手架不是 KPI。
- **本职工作有 deadline 的那几天**，只做 25 分钟的“读”，动手部分挪到周末。
- 每周整合日（D6）如果没时间，至少花 15 分钟把这周的数字汇到 `LOG.md`。
