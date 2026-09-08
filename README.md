<div align="center">

# tokens → torque

**从 token 到扭矩：72 天，把一个大模型一路跑到机器人关节上。**

`serving` · `CUDA` · `training` · `VLM` · `VLA` · `WAM`

[![Stars](https://img.shields.io/github/stars/enkerewpo/tokens-to-torque?style=flat&color=76B900)](https://github.com/enkerewpo/tokens-to-torque/stargazers) [![Discussions](https://img.shields.io/github/discussions/enkerewpo/tokens-to-torque?style=flat&color=76B900)](https://github.com/enkerewpo/tokens-to-torque/discussions) [![Site](https://img.shields.io/badge/site-enkerewpo.github.io-informational.svg)](https://enkerewpo.github.io/tokens-to-torque/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Progress](https://img.shields.io/badge/progress-3%2F73_days-76B900.svg) ![Hardware](https://img.shields.io/badge/hardware-Jetson%20%2F%20any%20CUDA%20GPU-76B900.svg)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="site_src/assets/hero-dark.svg">
  <img class="hero" alt="从 token 到扭矩：serving、CUDA、training、VLM、VLA、WAM 六个阶段，72 天" src="site_src/assets/hero-light.svg">
</picture>

</div>

这是一份动手课表。每天两小时：读一节，跑一个实验，记下一个数字。从推理服务开始，经过 CUDA、训练和视觉语言模型，最后到能输出动作的 VLA 和世界模型。全部实验在一块 Jetson AGX Thor 上跑过，大部分内容换任何一块 CUDA GPU 都成立。

**在线阅读：<https://enkerewpo.github.io/tokens-to-torque/>**（公式、搜索、深色模式、评论）

## 跑完会有什么

- 一个常驻的 LLM 服务，挂着自己微调的 adapter，配一个自己写的浏览器客户端。启动命令里每个参数的含义都清楚（Phase 1）
- 手写的 CUDA kernel 和 Triton kernel，以及一个数字：离 cuBLAS 还差百分之几（Phase 2）
- 一个从零训练的 20M GPT，和一个用 LoRA、QLoRA 微调过的 9B 模型（Phase 3）
- 一个接上视觉的模型，projector 是自己训的（Phase 4）
- 一个在边缘设备上运行的 VLA 策略，附控制频率和温度曲线（Phase 5）
- 一个感知 + VLA + 世界模型的最小闭环（Phase 6）

每一天都以一个数字收尾。已经跑出来的两组：

| Day | 问题 | 实测 |
|---|---|---|
| [00](days/day00_lora-quickstart/) | LoRA 给一个 9B 模型加了多少参数 | 43 M 个参数，adapter 166 MB，占全模型 0.48% |
| [01](days/day01_vllm-first-serve/) | vLLM 比 `transformers.generate` 快多少 | 首 token 快 3.3×，吞吐快 1.6×，单请求 |
| [02](days/day02_vllm-request-path/) | 一条请求在引擎里的时间花在哪 | 排队 0 ms，prefill 23 ms，decode 677 ms |

## 适合谁

已经在用这些模型，但对底层只有二手知识的人：知道 vLLM 快，说不清快在哪；知道 LoRA 省显存，说不出省了多少、省在哪一项。每一天从一个具体问题出发，先跑出数字，再解释这个数字为什么是这样。

前置要求：会 Python，看得懂矩阵乘法。Transformer、线性代数、优化器和数值格式在[附录](appendix/)里各有一份速查，正文用到时会指过去。

> **“从零”的含义**：从零基础和一块空开发板开始，不是每个组件都重写一遍。该用 vLLM 就用 vLLM，该部署 GR00T 就部署 GR00T，然后读它的源码，把它拆开。手写的部分（GPT、CUDA kernel、训练循环、projector）会明确标出。

## 硬件

| 手上有的 | 能跟到哪 |
|---|---|
| Jetson AGX Thor / Orin（32 到 128 GB 统一内存） | 全程 |
| Jetson Orin Nano / NX（8 到 16 GB） | Phase 1 到 4 基本完整，Phase 5 到 6 换小模型或只读 |
| 任意 CUDA 独显（12 GB 以上） | Phase 1 到 3 完整，Phase 4 到 6 看显存 |
| 只有 CPU / Mac | 概念和源码阅读部分，实验换成读别人的数字 |

选 Jetson 是因为课表最后要落到机器人上，边缘部署的约束（功耗墙、统一内存、无独显）正是要研究的对象。Jetson 特有的内容标 `[Jetson]`，只在 Thor 上验证过的数字标 `[Thor]`。

> [!WARNING]
> Jetson 到达热保护点时是硬件复位或关机，不是降频。所有 GPU 任务都走 [`common/`](common/) 的预检、遥测、看门狗和优雅停止四个脚本。看门狗阈值从内核读取本机的 trip point 再留余量，不写死数字。详见 [SETUP.md](SETUP.md#jetson-安全)。

## 课表

| Phase | 主题 | Days | 进度 |
|---|---|---|---|
| 0 | Quickstart：微调入门 | 00 | 1 / 1 |
| 1 | Serving：把模型跑起来并测准 | 01 到 12 | 2 / 12 |
| 2 | CUDA：从 kernel 到 profile | 13 到 24 | 0 / 12 |
| 3 | Training：从零训到微调 | 25 到 36 | 0 / 12 |
| 4 | VLM：接上视觉 | 37 到 48 | 0 / 12 |
| 5 | VLA：生成动作 | 49 到 60 | 0 / 12 |
| 6 | WAM：世界动作模型 | 61 到 72 | 0 / 12 |
| | | 合计 | 3 / 73 |

完整的每日目标、动手内容和产出在 [ROADMAP.md](ROADMAP.md)。

<details open>
<summary><b>Phase 0 · Quickstart</b>（day 00）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| [00](days/day00_lora-quickstart/) | 用 LoRA 微调一个 9B 模型 | adapter + 风格命中率 |

</details>

<details open>
<summary><b>Phase 1 · Serving</b>（day 01 到 12）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| [01](days/day01_vllm-first-serve/) | 把模型变成一个服务 | `serve.sh` + 第一组延迟数字 + 浏览器客户端 |
| [02](days/day02_vllm-request-path/) | 一个请求在 vLLM 里经历了什么 | 请求路径图 + 分段耗时 |
| 03 | KV cache 占多少显存 | 手算 vs 实测，误差 < 15% |
| 04 | 连续批处理为什么快 | throughput–latency 帕累托曲线 |
| 05 | 可复现的 benchmark | `bench.sh`，三遍方差 < 5% |
| 06 | Thor 上的可用工作点 | 并发、延迟、温度三维结论 |
| 07 | 量化格式全景 | 格式 × 是否真加速 × 精度代价 |
| 08 | 在 Thor 上量化一个模型 | 量化前后三项对比 |
| 09 | 投机解码 | accept rate 到实际加速比的曲线 |
| 10 | 边缘多模型共存 | 干扰矩阵 |
| 11 | prefix caching：SGLang vs vLLM | 命中率 + TTFT 节省 |
| 12 | 边缘推理层需求清单 | 一页总结 |

</details>

<details>
<summary><b>Phase 2 · CUDA</b>（day 13 到 24）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| 13 | CUDA 编程模型，跑通第一个 kernel | 实测带宽 / 理论带宽 |
| 14 | 内存层次：naive 到 tiled matmul | 两版 GFLOPS 与加速比 |
| 15 | Nsight 上手，profile day 01 的 vLLM | 前三热点 + 瓶颈类型 |
| 16 | reduction / scan / warp shuffle | 四版优化路径表 |
| 17 | Tensor Core 与 CUTLASS | 三层 tile 划分 |
| 18 | 把 matmul 推到 cuBLAS 的百分之几 | 一个数字 + 差距归因 |
| 19 | Triton 入门：fused softmax | 手写 CUDA vs Triton 的成本对比 |
| 20 | Triton matmul + autotune | 与 day 14 对比 |
| 21 | Flash Attention 原理 | 手推 online softmax |
| 22 | 改一个 attention kernel | 数值误差 < 1e-2 + 速度比 |
| 23 | Thor 的 roofline | roofline 图 + 关键算子定位 |
| 24 | Thor 上什么 memory-bound、什么 compute-bound | 指导后续优化的一页纸 |

</details>

<details>
<summary><b>Phase 3 · Training</b>（day 25 到 36）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| 25 | tokenizer 与数据 | BPE + 切词结果 |
| 26 | 手写一个 20M GPT | loss 从约 10 开始下降 |
| 27 | 优化器与 lr sweep | loss 曲线族，找到发散的边界 |
| 28 | 混合精度与显存 | 显存与吞吐的权衡表 |
| 29 | 评估与故意过拟合 | 让它模仿自己说话 |
| 30 | 第一个会说人话的模型 | checkpoint + 全部超参 |
| 31 | LoRA 原理 | 手算加了多少参数 |
| 32 | Full SFT / LoRA / QLoRA 在 Thor 上实跑 | 三档显存、时间、效果 |
| 33 | 数据集构造与 loss mask | 干净的 SFT 数据集 |
| 34 | 微调到一个真实下游任务 | 与通用 LLM 的指标对比 |
| 35 | 可信的评测链 | 评测脚本 |
| 36 | FSDP / ZeRO 原理与多卡实测 | scaling 效率 |

</details>

<details>
<summary><b>Phase 4 · VLM</b>（day 37 到 48）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| 37 | VLM 架构谱系 | 三派对比图 |
| 38 | 拆一个真模型的 forward | 张量 shape 流水账 |
| 39 | 视觉 token 的代价 | token 数到延迟的曲线 |
| 40 | 手训一个 projector | 训练前后描述质量 |
| 41 | 空间关系 / 计数 / OCR 评测 | benchmark 分数 |
| 42 | Cosmos Reason2 在 Thor 上的可用性 | 延迟、显存、精度报告 |
| 43 | VLM 当感知前端 vs 传统检测管线 | 三项对比 |
| 44 | 结构化输出与 constrained decoding | 100% 合法 JSON + 开销 |
| 45 | 多帧输入与 KV 复用 | 帧数到效果、延迟的曲线 |
| 46 | VLM-as-judge | 可信的自动评测器 |
| 47 | 控制回路里 VLM 的延迟预算 | 预算表 + 模型规模上限 |
| 48 | 给自己的模型接上眼睛 | demo |

</details>

<details>
<summary><b>Phase 5 · VLA</b>（day 49 到 60）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| 49 | VLA 谱系与三种动作头 | 谱系图 |
| 50 | π₀.₅ 上 Thor（TensorRT NVFP4） | 控制频率 + 温度曲线 |
| 51 | GR00T 1.7 上 Thor（混合 NVFP4） | 同上，对比 |
| 52 | 四个模型放同一 LIBERO 评测 | 成功率 × 频率 × 显存 × 功耗 |
| 53 | action chunking 与异步推理 | chunk 长度权衡曲线 |
| 54 | 边缘 VLA 的可行域 | 一张图 |
| 55 | LeRobot 数据格式 | 转换成功的小数据集 |
| 56 | 微调 GR00T N1.5 / 1.7 | loss 曲线 + rollout 视频 |
| 57 | 微调 SmolVLA / π₀.₅ | 两者对比 |
| 58 | 微调有没有用 | 哪些任务涨、哪些退 |
| 59 | 量化后的策略掉多少 | 精度与速度的权衡 |
| 60 | 全流程可复现脚本 | `vla_finetune.sh` |

</details>

<details>
<summary><b>Phase 6 · WAM</b>（day 61 到 72）</summary>

| Day | 主题 | 产出 |
|---|---|---|
| 61 | 世界模型谱系，WAM 与 VLA 的区别 | 谱系图 |
| 62 | DreamZero 与 Cosmos Policy | 方法对比笔记 |
| 63 | 视频扩散基础 | 手推 flow matching 目标 |
| 64 | 跑一个小 world model | 预测帧 vs 真实帧 |
| 65 | “想象”一次要多少毫秒 | 开销表 |
| 66 | WAM 在边缘可行吗 | 用数字回答 |
| 67 | WAM 的用法综述 | 一页笔记 |
| 68 | 用 world model 评策略 | 可行性结论 |
| 69 | WAM 的数据管线：视频怎么变成训练样本 | 一个能跑的转换脚本 |
| 70 到 71 | 最小闭环：感知 + VLA + WAM | 一段视频 |
| 72 | 收官与下一步 | 产出清单 + 3 个方向 |

</details>

## 开始

```bash
git clone https://github.com/enkerewpo/tokens-to-torque
cd tokens-to-torque
git config core.hooksPath .githooks   # 隐私守卫，只需一次
cat SETUP.md                           # 环境、依赖、Jetson 安全脚本
cd days/day00_lora-quickstart && cat README.md
```

每一天的目录都是自洽的：`README.md` 是教程，`code/` 是能跑的代码，`results/` 是数字和图。

## 约定

**每一天六节，缺一节不算写完**：为什么学、背景、动手、结果、踩坑、延伸。结果一节必须有实测数字。踩坑一节记录装不上、跑崩了和结论被推翻的过程。模板在 [templates/day.md](templates/day.md)。

**私人数据不进仓库。** 跟做时会产生只属于自己的东西：语料、聊天记录、数据集、checkpoint、流水账。它们放在各天的 `private/` 和根目录的 `LOCAL.md`，都在 `.gitignore` 里。`.githooks/pre-commit` 拦截 `private/` 路径、`*.jsonl`、IP 和邮箱；自己的私有关键词写进 `.git/private-patterns`。它拦不住正文里的人名和内部术语，提交前要自己看一遍。

**用 agent 跟做。** 课表默认开着 Claude Code 或 Codex 学。[AGENTS.md](AGENTS.md)（`CLAUDE.md` 是它的软链接）规定了 agent 的角色：先讲概念再动手，讲不清楚的地方补进当天教程的 §2。`.claude/skills/` 提供 `day-start`、`day-wrap`、`privacy-check`、`thor-guard` 四个 skill。

## 仓库结构

```
tokens-to-torque/
├── README.md                    # 本文件
├── ROADMAP.md                   # 72 天完整课表（每天的目标 / 动手 / 产出）
├── SETUP.md                     # 环境搭建 + Jetson 安全规程
├── RESOURCES.md                 # 精选材料，每条线只挑一份主材料
├── AGENTS.md                    # 给 AI agent 的工作说明（CLAUDE.md 是它的软链接）
├── appendix/                    # 附录：Transformer、线性代数、优化器、数值格式速查
├── templates/day.md             # 每日教程模板
├── common/                      # Jetson 预检 / 遥测 / 看门狗 / 优雅停止
├── scripts/                     # 站点构建、图表生成、文本检查
├── site_src/                    # 文档站（Quarto）模板与主题，内容由脚本生成
└── days/
    └── dayNN_topic-name/
        ├── README.md            # 教程正文
        ├── code/                # 可运行代码
        ├── results/             # 数字、图、日志摘要
        └── private/             # 个人数据，gitignore，永不上传
```

## 致谢

结构和写法参考了这几个仓库：

- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)：章节化、`setup/`、可复用包、引用规范
- [elizabetht/100-days-of-inference](https://github.com/elizabetht/100-days-of-inference)：phase 分段与进度表
- [bikrammajhi/100-days-of-GPU](https://github.com/bikrammajhi/100-days-of-GPU)：进度表当主导航、`dayNN_topic/` 命名
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/)：Thor 上大部分实验的起点

## License

[MIT](LICENSE) · wheatfox
