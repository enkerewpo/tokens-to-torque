---
name: jetson-guard
description: 在 Jetson 上跑 GPU 任务前后的安全规程。任何涉及 Jetson、Orin、Thor、tegra、jtop、nvpmodel、训练或推理任务的操作前使用；用独显则不需要。
---

# Jetson 安全规程

**Jetson 到达热保护点是硬件复位或关机，不是降频。** 远程操作时一次复位就可能失去这台机器，直到有人物理断电。

各型号临界温度不同（AGX Thor 是 118 °C），所以脚本**从 `/sys/class/thermal/thermal_zone*/trip_point_*_temp` 读这块板子自己的 trip point**，再留 30 °C 余量作为停机阈值。不要写死数字，也不要花这个余量。

## 起跑前（必做）

```bash
bash common/jetson_preflight.sh
```

检查：板子型号、GPU 是否响应、起始温度低于 `trip - 45 °C`、功耗模式不是不受限模式、风扇 pwm1 不为 0、没有遗留的同类进程。**任一项 FAIL 就不要起跑。**

## 跑起来之后

```bash
nohup bash common/jetson_telemetry.sh ~/telemetry/$(date +%F_%H%M).log &
nohup bash common/jetson_watchdog.sh '<进程正则>' 85 &
```

超过几分钟的任务**必须**挂看门狗。远程链路会断，长任务一律 `nohup`/`setsid` 起、日志落盘。

## 停止

```bash
bash common/jetson_stop.sh '<进程正则>'
```

顺序是 SIGTERM → docker stop（20s 宽限）→ 查孤儿进程 → 复查 GPU 和温度。

## 绝对不做

1. 改功耗/电流/温度限制（sysfs 或 `nvpmodel`）——NVIDIA 写明可能造成**永久损坏**。不切到不受限模式。
2. 对碰 `/dev/nvidia*` 的进程用 `kill -9`——SIGKILL 打断 ioctl 会留下收不掉的 D 状态僵尸，抓着设备 FD 不放，远程救不回来。
3. 重启，或跑任何需要重启才能恢复的东西。
4. 无上界的 job。
5. `nvidia-smi -pm 1`（persistence mode）——跨进程泄漏 context。

## 温度档位

| 温度 | 动作 |
|---|---|
| < `trip - 45` | 正常 |
| 逼近阈值 | 盯着 |
| **≥ `trip - 30`** | **优雅停止**（看门狗自动执行） |
| 更高 | 散热出问题，全停查原因 |

以 AGX Thor（trip 118 °C）为例：< 73 °C 正常，≥ 88 °C 停。空载基线约 45 °C。

## 出问题的征兆

- `nvidia-smi` 卡住 → GPU 子系统楔死，**不要再加负载**
- 温度逼近阈值而风扇已满转（pwm1≈255）→ 散热问题，停机
- 重负载中失联 → 可能已热复位。别猛重连，等，回来查 `uptime` / `last reboot`

## 环境

```bash
source common/env.sh
```

统一内存下 `nvidia-smi` 的 per-process 显存显示 `[N/A]` 是**正常的**，别去查。
