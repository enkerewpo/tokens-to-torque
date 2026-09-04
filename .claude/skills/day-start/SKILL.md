---
name: day-start
description: 开始课表里的某一天。当用户说 "day NN"、"开始今天的"、"下一天" 时使用。会从 ROADMAP 取出当天任务、按模板建好目录、做好 GPU 预检，然后先讲概念再动手。
---

# 开始一天

## 1. 取出任务

从仓库根目录的 `ROADMAP.md` 里找到 `| **NN** |` 那一行，读出三列：**目标 / 动手 / 产出**。同时看一眼 `RESOURCES.md` 里这条线的主材料——**只给一份**，不要堆链接。

## 2. 建目录

```bash
NN=14; SLUG=memory-hierarchy-tiled-matmul     # slug 用英文小写连字符，能概括当天主题
mkdir -p days/day${NN}_${SLUG}/{code,results,private}
cp templates/day.md days/day${NN}_${SLUG}/README.md
```

填好 README 顶部的 Phase / 日期 / 机器 / 耗时。

## 3. GPU 预检（涉及 GPU 时）

见 `thor-guard` skill。**没过预检不要起跑。**

## 4. 先讲概念，再动手

当天的 25 分钟“读”要真的讲明白，而不是甩链接：

- 数学用 LaTeX（`$$...$$`），**不要用表格糊弄矩阵运算**
- 用户说“没懂”时，把补充解释**写进当天 README 的 §2**，不要只在对话里说一遍
- 概念讲完再进 §3 动手

## 5. 动手时

- 命令逐条给，能复现
- 代码写进 `code/`，不要整段贴进 README
- 个人数据一律进 `private/`（已 gitignore）

## 收尾

用 `day-wrap` skill。
