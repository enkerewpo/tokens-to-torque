---
name: privacy-check
description: 提交前检查有没有个人数据或基础设施信息要进公开仓库。当准备 git commit/push、或用户提供了聊天记录、笔记、语料、机器信息时使用。
---

# 提交前的隐私检查

这个仓库是**公开**的，而使用者会把私人数据放进来当训练材料。一旦推上去，force-push 也抹不掉——GitHub 上旧 commit 仍能按 SHA 访问。

## 绝不能进仓库

| 类别 | 例子 |
|---|---|
| **个人数据** | 聊天记录、私人笔记、SFT 数据集、checkpoint、adapter、个人流水账 |
| **基础设施** | 主机名、IP、ssh 别名、内网地址、共享服务器的使用惯例和事故记录 |
| **身份** | 邮箱、真名、称谓、个人站点域名 |
| **组织内部** | 内部项目代号、未公开的实验结论、他人的原话 |

公开文档里硬件只写**通用型号**（“Jetson AGX Thor”可以，具体是谁的哪台不行）。示例命令一律用 `you@example.com`、`~/path/to/repo`。

## 检查步骤

```bash
# 1. 暂存了什么
git diff --cached --name-only

# 2. private/ 有没有溜出来
git diff --cached --name-only | grep -E 'private/|\.jsonl$|LOCAL\.md' && echo "STOP"

# 3. 正文扫一遍（hook 也会做，但先自己看）
git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|@[a-z0-9.-]+\.[a-z]{2,}'
```

## hook 拦不住什么

`.githooks/pre-commit` 检查路径、IP、邮箱，加上 `.git/private-patterns` 里的自定义词。它**拦不住**你新写进正文的中文人名、称谓、内部术语——因为词表里没有。

所以：**发现新的私有词，先加进 `.git/private-patterns` 再继续**（那个文件在 `.git/` 里，不可能被提交）。

## 装 hook

clone 之后要执行一次：

```bash
git config core.hooksPath .githooks
```

## 已经推上去了怎么办

1. 立刻改掉内容、`git commit --amend`、`git push --force-with-lease`（止血）
2. 告诉使用者：**旧 commit 仍能按 SHA 访问**，彻底清除要删库重建
3. 泄露的是凭证就必须轮换，不是改文件能解决的
