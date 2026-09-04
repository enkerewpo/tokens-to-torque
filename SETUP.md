# SETUP

## 每天的 2 小时怎么切

```
00:00–00:25  读    当天的概念 / 一段源码 / 半篇 paper
00:25–01:40  做    跑实验、改代码、profile
01:40–02:00  写    days/dayNN/README.md：数字 + 一句话结论 + 下一个疑问
```

**没跑出数字的一天不算完成。**

## Jetson 环境

JetPack 7.0（Jetson Linux / Ubuntu 22.04 系）。**PyTorch / vLLM / SGLang 一律用 NGC 的 Jetson wheel 或容器**，不要装 pypi 上的通用包——通用 wheel 没有 `sm_110` 的 kernel，装上也跑不了。

```bash
# 每次起 Python GPU 任务前
source common/env.sh
```

`common/env.sh` 设置：

| 变量 | 为什么 |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` | 统一内存下减少碎片 |
| `HF_ENDPOINT=https://hf-mirror.com` | Thor 没有 HF 直连，不设会静默超时 |
| `MUJOCO_GL=egl` | headless 渲染（LIBERO 等） |

> 统一内存架构下 `nvidia-smi` 的 per-process 显存显示 `[N/A]` 是**正常的**，不是 bug，别去查。

### Jetson 的几个反直觉之处（以 Thor 为例）

- **`sm_110`（Blackwell）**，编译要 `-arch=sm_110`。很多教程写死 `sm_80`/`sm_90`。
- **没有 RT core。**
- **统一内存 122 GB**：CPU 和 GPU 共享，没有独立显存。x86 教程里大段的 H2D/D2H 拷贝优化在这里不适用——这个差异本身就是 day 23 的研究点。
- **反过来，122 GB 的统一内存让 Thor 塞得下同价位独显塞不下的模型**，微调时它反而是优势那一方。

## Jetson 安全

Thor 是一块**单独的、不可替换的**板子，而且经常是远程操作、够不着电源。**junction 温度 118 °C 会触发硬件热保护复位**——不是降频，是复位。远程时复位 = 丢会话、可能起不来。

所以 85 °C 就停，33 °C 的余量一分不花。

### 四件套

```bash
# 1. 跑前预检（任何一项 FAIL 都不要起跑）
ssh "$THOR" 'bash -s' < common/jetson_preflight.sh

# 2. 起遥测（后台常驻，30s 一次）
ssh "$THOR" 'nohup bash -s > /dev/null 2>&1 &' < common/jetson_telemetry.sh

# 3. 起看门狗（>=85C 自动优雅停）
ssh "$THOR" "nohup bash common/jetson_watchdog.sh 'train_|vllm' 85 &"

# 4. 要停的时候（永远不要 kill -9）
ssh "$THOR" "bash common/jetson_stop.sh 'train_|vllm'"
```

### 绝对不做

1. **不改功耗 / 电流 / 温度限制**（sysfs 或 `nvpmodel`）。NVIDIA 明确写了放宽这些“可能造成永久性损坏”。不切 MAXN，保持 120 W（id 1）。
2. **不对碰 `/dev/nvidia*` 的进程用 `kill -9`。** SIGKILL 打断 ioctl 会留下收不掉的 D 状态僵尸，抓着设备 FD 不放，远程救不回来。
3. **不重启，也不跑任何需要重启才能恢复的东西。**
4. **每个 job 都要有界。** 不开放式循环，长 sweep 必须挂看门狗。
5. **不开 persistence mode**（`nvidia-smi -pm 1`）——会跨进程泄漏 context。
6. **远程连接会断。** 长 job 一律 `nohup` / `setsid` 起，日志落盘，别指望前台任务活过断线。

### 温度档位

| tj | 怎么办 |
|---|---|
| < 70 °C | 正常，可以起跑 |
| 70–80 °C | 盯着，短时可以 |
| **≥ 85 °C** | **优雅停止** |
| ≥ 95 °C | 出问题了（风扇？风道？）—— 全停，查原因 |

空载基线约 45 °C。

### 出问题的征兆

- `nvidia-smi` 卡住 → GPU 子系统楔死，**不要再加负载**。
- tj 逼近 85 °C 而风扇已经满转（pwm1≈255）→ 散热问题，停机。
- 重负载中突然连不上 → 可能是热复位。别猛重连，等，回来后查 `uptime` / `last reboot`。

### 依据

- [Jetson Thor Platform Power and Performance — Jetson Linux Developer Guide r38.2.1](https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html)
- [Jetson Thor Series Modules Thermal Design Guide (TDG-12271-001)](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/thor/docs/jetson_thor_thermal_dg_tdg12271001.pdf)


## 本地

读代码、画图、写 `days/*/README.md`。不跑 GPU 实验。
