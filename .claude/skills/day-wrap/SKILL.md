---
name: day-wrap
description: 收尾课表里的一天。当用户说 "今天完了"、"收尾"、"提交今天的" 时使用。检查六节是否写全、数字是否落地、站点能否构建，然后安全提交。
---

# 收尾一天

## 1. 六节检查

`days/dayNN_*/README.md` 必须齐：

| 节 | 检查点 |
|---|---|
| 1 为什么要学这个 | 接上了前一天，不是“课表上是今天” |
| 2 背景 | 有自己的理解；公式用 LaTeX |
| 3 动手 | 逐条命令，别人能复现 |
| 4 结果 | **有实测数字**，不是占位符 |
| 5 踩坑 | 装不上/报错/假设被推翻，没有就写“无” |
| 6 延伸 | 一到两个真读过的链接 + 明天的问题 |

**§4 还是空的就别提交。** 没跑出数字的一天不算完成——如实说，不要替他填。

## 2. 站点能构建

```bash
python scripts/check_footnotes.py     # 引用与定义一一对应
python scripts/build_docs.py
quarto render site_src
```

脚注这一步单独查，是因为整段替换文本时很容易把文末的定义一起删掉，而 Quarto 只会把 `[^key]` 原样渲染成字面量，不报错。

提示框在源码里用 **GitHub alert 语法**（`> [!NOTE]` / `> [!WARNING]` / `> [!CAUTION]` / `> [!TIP]` / `> [!IMPORTANT]`），这样直接在 GitHub 上看仓库也能正常渲染；构建时 `build_docs.py` 会把它转成 Quarto 的 callout。想给提示框加标题，就在 `> [!NOTE]` 下一行写 `> **标题**`。

## 3. 提交前逐条看

```bash
git status --short
git diff --cached --stat
```

**重点确认没有 `private/` 里的东西溜出来**，也没有把机器名/IP/邮箱/真名/内部项目名写进公开 md。hook 拦得住路径和 IP/邮箱，**拦不住正文里的中文人名和内部术语**。

```bash
git add days/dayNN_*/README.md days/dayNN_*/code days/dayNN_*/results
git commit -m "day NN: <一句话说清做了什么、出了什么数>"
git push
```

push 到 main 后 GitHub Actions 自动部署文档站。

## 4. 更新进度

`README.md` 里 Phase 表的 `X / 12` 和总数 `X / 73` 加一。
