# RESOURCES — 精选材料

标 ✅ 的是 2026-09-04 这次会话里实际检索确认存在的；其余是长期稳定的经典材料。**原则：每条线只挑一份主材料跟到底**，链接多了等于没有。

## 0. Thor 平台（先看这个，不然后面装不上）

- ✅ [**Jetson AI Lab — Tutorials**](https://www.jetson-ai-lab.com/tutorials/) ← **主材料，几乎覆盖你六条线的一半**
  - Introduction to GenAI on Jetson（Ollama 快试 / vLLM 求性能）
  - GenAI Benchmarking: LLMs and VLMs on Jetson（day 05 用）
  - Speculative Decoding on Jetson（MTP / DFlash / DSpark，day 09 用）
  - TensorRT Edge-LLM on Jetson（量化 → ONNX → C++ 推理，day 08 用）
  - **Fine-tune LLMs on Jetson**（Full SFT 4B / LoRA 9B / QLoRA 27B，day 00 和 day 32 用）
  - Cosmos Reason2 on Jetson（VLM，day 42 用）
  - **OpenPi π₀.₅ on Jetson Thor**（TensorRT NVFP4，day 50 用）
  - **Isaac GR00T 1.7 on Jetson Thor**（混合 NVFP4，day 51 用）
  - GTC 2026 Workshop — Generative AI on Jetson Thor（100 分钟动手课）
- ✅ [vLLM 对 Jetson Thor 的支持追踪](https://github.com/vllm-project/vllm/issues/31269)
- ✅ [JetPack 7.1 / T4000 发布说明](https://jetsonhacks.com/2026/01/12/jetpack-7-1-and-jetson-t4000-now-available/)（Thor 出厂常见是 JetPack 7.0；7.1 = Jetson Linux 38.4 / kernel 6.8 / Ubuntu 24.04。**升级前先想清楚，远程刷机风险高**）
- ✅ [JetPack 7.2 平台级 reset](https://www.seeedstudio.com/blog/2026/07/09/jetpack-7-2-platform-level-reset-and-the-new-era-of-agentic-ai/)

> Thor 的 vLLM / SGLang / PyTorch 官方 wheel 和容器走 **NGC**，不要用 pypi 上的通用包。

## 1. Serving（Phase 1）

- vLLM 官方文档 + 源码：`vllm/v1/core/sched/`（scheduler）、`vllm/v1/worker/`（执行）
- PagedAttention 论文（Efficient Memory Management for LLM Serving, SOSP'23）— day 03 的理论来源
- Orca: 连续批处理的原始论文（OSDI'22）— day 04
- SGLang + RadixAttention 论文 — day 11，agent 场景的 prefix 复用
- ✅ [How to Run vLLM on Jetson AGX Thor](https://blog.aetherix.com/how-to-run-vllm-on-jetson-agx-thor/)

## 2. CUDA（Phase 2）

- **PMPP**（*Programming Massively Parallel Processors*, 4th ed.）— 前 6 章足够，别通读
- **GPU MODE**（原 CUDA MODE）讲座系列 + 习题 — 社区最好的实践课
- CUDA C++ Programming Guide — 当字典用，不要顺序读
- ✅ [**Colfax Research 的 CUTLASS/Blackwell 系列**](https://research.colfax-intl.com/) — Blackwell Tensor Memory 和 5th-gen MMA 写得最清楚
- ✅ [CUTLASS](https://github.com/NVIDIA/cutlass) （4.x 有 Python DSL，比纯 C++ 模板好入门）
- Triton 官方 tutorials（vector add → fused softmax → matmul → flash attention，正好是 Phase 2 的路线）
- Nsight Systems / Nsight Compute 官方 quickstart

> Thor 是 sm_110，编译时 `-arch=sm_110`；无 RT core；统一内存（无独立显存拷贝）——很多 x86 GPU 教程里的 H2D/D2H 优化在这里不适用，这本身是个好研究点。

## 3. 训练与微调（Phase 3）

- **nanoGPT** / **modded-nanogpt** — day 26 手敲的参考，后者是速度优化的军备竞赛，很好玩
- HuggingFace **TRL**（SFTTrainer）+ **PEFT** — Phase 3 后半的主力工具
- ✅ Jetson AI Lab «Fine-tune LLMs on Jetson» — Thor 上的实跑路径
- LoRA 论文 + QLoRA 论文 — 各读半小时够了
- FSDP 官方教程 / DeepSpeed ZeRO 论文 — day 36
- Chinchilla scaling law — 建立“多少数据配多少参数”的直觉

## 4. VLM（Phase 4）

- LLaVA 系列论文（1 → 1.5 → NeXT）— projector 路线的教科书
- Qwen-VL / InternVL 技术报告 — 动态分辨率、AnyRes 的工程细节
- ✅ Cosmos Reason2 on Jetson（Jetson AI Lab）— 物理常识推理 VLM
- xgrammar / outlines — 结构化输出，day 44

## 5. VLA（Phase 5）

- OpenVLA、π0、π0.5、GR00T N1.5/N2 的论文与代码
- ✅ [**Fine-tune Isaac GR00T N1.5 for LeRobot SO-101 + Deploy on Jetson Thor**（Seeed Wiki）](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/)
- LeRobot 仓库 + 数据格式文档
- LIBERO benchmark（你已有基础）
- RoboArena 排行榜 — 看谁真的强

## 6. WAM（Phase 6）

- ✅ [**NVIDIA: What Is a World Action Model**](https://www.nvidia.com/en-us/glossary/world-action-model/) — 先读定义
- ✅ [**Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models**（NVIDIA 技术博客）](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
- ✅ [**Awesome-WAM**](https://github.com/OpenMOSS/Awesome-WAM) — 论文清单
- ✅ [**WAM Survey（交互式分类 + 每周更新）**](https://world-action-models.github.io/)
- ✅ [World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/html/2605.00080v1)
- ✅ [World Action Models are Zero-shot Policies](https://arxiv.org/html/2602.15922v1)
- **DreamZero**（Wan 2.1-I2V-14B backbone 的联合预测 WAM；2026-04 RoboArena 1750 Elo vs π0.5 的 1622）
- **Cosmos Policy**（Kim et al., 2026；把动作/未来状态/value 当作 latent frame 塞进视频扩散序列）
- Dreamer V3、Genie 3、V-JEPA 2 — 背景谱系

## 用法提醒

- 每天只开**一个**链接。开三个等于零个。
- 论文先读 abstract + 图 + 实验表，觉得值得再读方法。
- 装不上的轮子不要死磕超过 40 分钟——记下来，换台机器或者换条路。
