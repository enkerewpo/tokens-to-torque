# Day 01 · 把模型变成一个服务

> **Phase** 1 · serving
> **日期** 2026-09-05 · **机器** Jetson AGX Thor · **耗时** ~2h

把 day 00 那个“跑一次就退出”的脚本换成一个**常驻的 HTTP 服务**：权重只加载一次，之后用 `curl` 就能问它问题。跑完你会有一条能复现的启动命令，和这块板子上的第一组延迟数字。

你会做这几件事：

- 用 NGC 的 vLLM 容器起一个 OpenAI 兼容的 server
- 读懂启动命令里每一个参数在管什么
- `curl` 通一次，看清请求和响应的格式
- 量出这块板子上的 TTFT、TPOT、端到端延迟

## 1. 为什么要学这个

day 00 的 `chat.py` 每次运行都要重新加载 18 GB 权重（约 40 秒），而且**一次只服务你一个人**。真实的用法是反过来的：模型常驻在机器上，很多请求陆续到达，服务要决定谁先算、能不能几个人拼在一起算。

起来的这个服务是 Phase 1 剩下五天的实验台：day 02 读它的源码看一个请求怎么走完全程，day 03 算它的 KV cache 占多少内存，day 04 加并发看吞吐怎么涨，day 05 做可复现的 benchmark。所以目标不是“跑起来就行”，而是**起得可复现、并且知道每个参数在干什么**。

## 2. 背景

### 2.1 一次 `generate` 和一个服务差在哪

day 00 的脚本是这样的：启动进程 → 加载权重 → 回答一个问题 → 退出。三件事里，加载占了 40 秒，回答占了几秒。

服务把它拆开了：

| | day 00 的脚本 | 一个 serving 服务 |
|---|---|---|
| 权重加载 | 每次运行都来一遍 | 启动时一次，之后常驻内存 |
| 请求来源 | 命令行参数 | HTTP 请求，随时可以来 |
| 同时几个请求 | 一个 | 多个，服务自己排队和拼批 |
| 结束 | 答完就退出 | 一直等下一个请求 |

“拼批”（batching）是关键：GPU 一次算一个请求时，大部分算力是闲的——矩阵乘法的形状太瘦。把几个请求的 token 拼成一个更大的矩阵一起算，几乎不增加时间，吞吐却成倍涨。vLLM 的做法叫**连续批处理**（continuous batching），day 04 会量它的效果。

### 2.2 OpenAI 兼容 API 是什么

**OpenAI 兼容**（OpenAI-compatible）指的是一组约定好的 HTTP 接口：路径、请求体字段、响应格式都和 OpenAI 的 API 一样。下面用到三个：

| 端点 | 干什么 |
|---|---|
| `GET /health` | 活着没有。返回 200 就是就绪 |
| `GET /v1/models` | 这个服务在跑哪个模型 |
| `POST /v1/chat/completions` | 真正的问答接口 |

**为什么几乎所有推理引擎都抄这套接口。** 因为客户端生态已经写好了：官方 `openai` SDK、各种聊天前端、agent 框架，全都会说这套协议。你的服务只要长得一样，别人改一个 `base_url` 就能指过来，代码一行不用动。这是事实标准，不是技术上的必然。

请求体里会用到的字段：

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "messages": [{"role": "user", "content": "用三句话解释什么是 KV cache。"}],
  "max_tokens": 128,
  "temperature": 0,
  "stream": true
}
```

`messages` 就是 [附录 D.1](../../appendix/transformer.md) 说的那串对话，服务端会用模型自带的 chat template 把它拼成 token（day 00 §2.10 讲过这个模板）。`stream: true` 让服务端**边生成边推**，而不是等整段写完再一次返回——下一节说明为什么这件事对测延迟是必需的。

### 2.3 延迟的三个数

**定义。** 设一个请求生成了 $N$ 个 token：

- **TTFT**（time to first token，首 token 延迟）：从发出请求到收到**第一个** token 的时间。
- **TPOT**（time per output token，每个后续 token 的时间）：第一个 token 之后，平均每多生成一个 token 要多久。它的倒数就是常说的"每秒多少 token"。
- **端到端延迟**（end-to-end latency）：整条请求从发出到收完的总时间。

三者的近似关系：

$$
\text{E2E} \;\approx\; \text{TTFT} + \text{TPOT} \times (N - 1)
$$

**为什么要分开看。** 因为它们由不同的东西决定（[附录 D.8](../../appendix/transformer.md)）：TTFT 主要花在 **prefill**——把你的整段提示一次算完，是大矩阵乘法，吃算力；TPOT 花在 **decode**——每步只算一个新位置，是矩阵乘向量，算得少、读得多，吃内存带宽。所以提示越长 TTFT 越大，而 TPOT 基本和提示长度无关。

**为什么必须开 `stream: true` 才能测 TTFT。** 不流式的话，服务端会等整段生成完再一次性返回，你测到的“第一个 token 到达时间”其实是最后一个 token 的时间，TTFT 会等于端到端延迟。

### 2.4 `--gpu-memory-utilization` 在统一内存上的含义

> [!WARNING]
> **[Thor] 这个参数在独显和 Jetson 上切的不是同一块东西**
>
> 在独显上，它是**显存**的比例：`0.9` 表示 vLLM 最多用掉 90% 的显存，剩下的留给别的进程。
>
> Thor 是**统一内存**（unified memory）：CPU 和 GPU 共用同一块 122 GB。这个比例切的是这块共享内存的一部分，切太狠会把系统本身挤爆——别的容器、页缓存、你的 ssh 会话都在同一块内存里。这里用 `0.30`（约 36 GB）：18 GB 权重 + KV cache + 工作区，够用且留足余量。

vLLM 拿到这块内存后，先放权重，剩下的**全部**拿去当 KV cache。所以启动日志里会打印一行"KV cache 能存多少 token"，这个数决定了服务能同时处理多长、多少条请求——day 03 专门算这笔账。

## 3. 动手

### 3.0 前置

- Thor 上已经有 NGC 的 vLLM 容器镜像 `nvcr.io/nvidia/vllm:26.06-py3`（32.7 GB）。没有的话 `sudo docker pull` 一次。
- 模型权重在 `~/.cache/huggingface`，day 00 下过了。
- 起跑前照例过一遍预检：

```bash
bash ../../common/jetson_preflight.sh
```

### 3.1 起服务

```bash
bash code/serve.sh                    # 默认 Qwen/Qwen3.5-9B，端口 8000
sudo docker logs -f t2t-vllm          # 跟启动日志，Ctrl-C 只是停止跟随，不影响服务
```

`serve.sh` 干的事就是一条 `docker run`，逐个参数说明：

| 参数 | 作用 |
|---|---|
| `--runtime nvidia` | 把 GPU 交给容器。Jetson 上不用 `--gpus all` |
| `--ipc=host` | vLLM 内部是多进程的，进程间要用共享内存；不给的话容器默认只有 64 MB，会莫名其妙挂掉 |
| `--network host` | 容器直接用宿主机网络，容器里的 8000 就是宿主机的 8000，省掉端口映射 |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | 把已经下好的权重挂进去，不用再下一遍 |
| `-e HF_HUB_OFFLINE=1` | 明确禁止联网拉模型。没网的时候它会立刻报错，而不是卡住重试 |
| `--entrypoint vllm` | 这个镜像默认入口是 NVIDIA 的初始化脚本，要显式改成 `vllm` 命令 |
| `serve <模型>` | vLLM 的子命令，起 OpenAI 兼容 server |
| `--max-model-len 8192` | 单个请求最长多少 token（提示 + 生成）。KV cache 随它线性增长 |
| `--gpu-memory-utilization 0.30` | 见 §2.4 |

### 3.2 等它就绪

第一次启动要编译 kernel（torch.compile）并捕获 CUDA graph，比后面几次慢很多。用 `/health` 轮询，不要靠猜：

```bash
until curl -sf localhost:8000/health >/dev/null; do sleep 5; done && echo 就绪
```

### 3.3 curl 通一次

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool | head -12

curl -s localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3.5-9B",
         "messages":[{"role":"user","content":"用一句话说明你是谁。"}],
         "max_tokens":64, "temperature":0}' | python3 -m json.tool
```

### 3.4 把 day 00 的 adapter 挂上去

`serve.sh` 默认会顺手挂上 day 00 训出来的 adapter：

```bash
LORA=<adapter 目录> LORA_NAME=day00-demo bash code/serve.sh
```

挂上之后，**base 和 adapter 在同一个服务里共存**，`/v1/models` 会列出两个名字，请求里换 `model` 就切换：

```bash
curl -s localhost:8000/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
# ['Qwen/Qwen3.5-9B', 'day00-demo']
```

这是 LoRA 在部署侧的价值：**一份 18 GB 的底座，挂 N 个几百 MB 的 adapter**，显存里只多出 N × 43 M 个参数，而不是 N 份完整模型。

> [!WARNING]
> **adapter 可能被静默忽略——服务照常起，回答和 base 一模一样**
>
> 直接把 day 00 的 adapter 挂上去是不生效的，而且**不报错**：`/v1/models` 里有它，请求也路由过去了（响应的 `model` 字段就是 adapter 名），但输出和 base 逐字节相同。
>
> 原因是权重的键名对不上。训练时用 `AutoModelForCausalLM` 加载的是纯文本模型，键名长这样：
>
> ```text
> base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
> ```
>
> 而 vLLM 把同一个模型当**多模态模型**实例化（它带一座视觉塔，见 day 00 §2.9），文本塔挂在 `language_model` 下面。名字对不上的权重被跳过，一个都没生效。
>
> 修法是改键名，顺便只保留 vLLM 支持的那几类模块：
>
> ```bash
> python code/relabel_adapter.py \
>     --in  <训练出来的 adapter> \
>     --out <改好的 adapter> \
>     --prefix language_model. \
>     --keep q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
> ```
>
> 496 个张量里保留 256 个：丢掉的是线性注意力层里的 `in_proj_*`、`out_proj`——vLLM 的 GatedDeltaNet 实现不接 LoRA。**训练时能挂的模块，部署时不一定能挂。**

### 3.5 量延迟

```bash
python3 code/latency.py --model Qwen/Qwen3.5-9B --runs 5
```

脚本做三件事：先跑一次热身（第一次请求会触发编译和缓存分配，不能计进统计）；然后开 `stream: true` 发 5 次，记录第一个数据块到达的时刻当作 TTFT；最后把中位数打出来。它只用标准库，在宿主机上跑，不进容器。

### 3.6 用浏览器聊天

`code/ui/index.html` 是一个自包含的客户端：没有构建、没有依赖，一个文件。它做的事就是 §2.2 说的那句话——接口长得一样，客户端直接指过来。

```bash
bash code/ui.sh          # 在跑模型的那台机器上起个静态服务
# 浏览器打开 http://<那台机器的地址>:8181
```

页面默认把服务地址猜成"同一台机器的 8000 端口"，所以在板子上托管时不用改任何配置。它会从 `/v1/models` 把 base 和 adapter 都拉出来做成按钮，点一下就切；每条回答下面实时显示 TTFT、TPOT 和 token/s——§2.3 那三个数，你自己聊天的时候就能看见。

> [!NOTE]
> **页面和模型不在同一台机器上时**
>
> 浏览器不允许一个网页随便去请求别的地址，除非对方明确说“我放行”。vLLM 默认对所有来源放行，所以这一层通常不挡事。
>
> 但如果你把页面放在自己电脑上打开、去连另一台机器上的服务，Chrome 还有一层限制：不让一个网页去摸内网地址。它会直接拒掉，控制台只留一句 `Failed to fetch`，看不出原因。
>
> 所以就按上面那样做：**页面托管在跑模型的那台机器上**，两边同一台机器，什么都不用配。

### 3.7 停

```bash
bash code/stop.sh
```

用的是 `docker stop`（先 SIGTERM，再等 20 秒宽限），不是 `kill -9`——对正在碰 `/dev/nvidia*` 的进程用 SIGKILL 会留下清不掉的 D 状态进程，在远程板子上等于把 GPU 弄丢。

## 4. 结果

Jetson AGX Thor（120 W），vLLM 0.22.1（NGC `26.06-py3` 容器），Qwen3.5-9B bf16，`--max-model-len 8192 --gpu-memory-utilization 0.30`。延迟是 `code/latency.py` 跑 5 次取中位数，提示固定为“用三句话解释什么是 KV cache。”，生成 128 个 token。

| | TTFT | TPOT | 吞吐 | 端到端（128 token） |
|---|---|---|---|---|
| base | **80.8 ms** | 62.0 ms | **16.1 token/s** | 7.96 s |
| base + day00 adapter | 89.9 ms | 72.9 ms | **13.7 token/s** | 9.35 s |

### 比不用 serving 引擎快多少

同一台机器、同一个模型、同一个问题，用 day 00 那种写法（`transformers.generate`，`code/baseline_hf.py`）作对照：

| | TTFT | TPOT | 吞吐 | 每次运行还要等 |
|---|---|---|---|---|
| `transformers.generate` | 264.0 ms | 101.5 ms | 9.8 token/s | 加载权重 15.4 s（页缓存热了之后 3.9 s） |
| vLLM | **80.8 ms** | **62.0 ms** | **16.1 token/s** | 0（常驻） |
| 快了多少 | **3.3×** | **1.6×** | **1.6×** | — |

这还只是**单请求**。vLLM 在这里赢的是实现细节：CUDA graph 把每步 decode 的启动开销压掉、算子做过融合、注意力用分页的 KV cache。真正拉开差距的是并发——多个请求拼成一批一起算，day 04 会把这条曲线量出来。

验算 §2.3 那个式子：$0.0808 + 0.0620 \times 127 = 7.95$ s，实测 7.96 s。

启动过程（第一次，之后有编译缓存会快很多）：

| 阶段 | 耗时 |
|---|---|
| torch.compile | 56 s |
| 显存探测 + warmup | 87 s |
| 从敲下命令到 `/health` 返回 200 | 约 4 min |

vLLM 报的 KV cache 容量：**base 895 946 token，挂 adapter 后 879 130 token**（adapter 的权重也占那块内存）。day 03 会拿这个数和手算的公式对账。

**挂 adapter 的代价是 15% 的吞吐**（16.1 → 13.7 token/s）。day 00 §2.6 说 LoRA“推理时零开销”，那句话的前提是**把 adapter 合并进权重**；这里是把它当独立分支挂着，每层都要多算一次 $BA$，所以不是零。要零开销就得合并，代价是失去“一个底座挂多个 adapter”的能力。

**风格确实跟着 adapter 走了。** 同一个问题、同样 `temperature=0`：

| | 回答开头 |
|---|---|
| base | 抱抱你！辛苦了～ 🌙 有时候累到连话都不想多说…… |
| day00-demo | **嗯…**那就允许自己彻底瘫一会儿，不用急着把今天补回来。 |

## 5. 踩坑

1. **adapter 被静默忽略**（详见 §3.4 那个警告框）。服务正常起、`/v1/models` 里有名字、请求也路由过去，就是不生效。判断方法很简单：**同一个问题、`temperature=0`，base 和 adapter 的输出如果逐字节相同，那就是没生效**。
2. **vLLM 默认开着 thinking。** 不传参数时，这个模型会先写一大段英文的思考过程再回答，把 `max_tokens` 吃光。要关掉得在请求里加 `"chat_template_kwargs": {"enable_thinking": false}`——注意这不是 OpenAI 官方字段，是 vLLM 的扩展。day 00 是在代码里调 `apply_chat_template(enable_thinking=False)`，服务端换了个入口。
3. **热身不止一次。** 前两次请求明显更慢（端到端 11.6 s、11.1 s），第三次开始才稳定在 8.0 s。只热身一次就开测会把数字拉高 40%。day 05 做正经 benchmark 时要专门处理这件事。
4. **serving 必须另起一个容器。** `t2t` 是 pytorch 镜像，vLLM 在 Jetson 上要用 NGC 的 vllm 镜像，一个容器只能有一个镜像。约定和理由写进了 [SETUP](../../setup.md#为什么-serving-要另起一个容器)。
5. **浏览器走系统代理时，页面打不开还只给一个 502。** 如果你开着 Clash 这类代理客户端，浏览器会把请求交给代理，而代理连不到你内网或 Tailscale 上的地址，回给你一个 502——看起来像服务挂了，其实服务好好的（用 `curl` 一试就通，因为 curl 默认不走系统代理）。把对应网段加进代理的绕过列表即可——家用内网一般是 `192.168.x.x`，Tailscale 分配的地址都在 `100.64.x.x`–`100.127.x.x` 这一段（CGNAT 网段），很多代理客户端的默认绕过规则里没有它。
6. **`--gpu-memory-utilization` 在统一内存上切的是整块内存**（§2.4），不是独显那种"反正显存是我的"。这台机器上还跑着别的容器，切太狠会把系统挤爆。

## 6. 延伸

- vLLM 的 [OpenAI 兼容 server 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)——把上面用到的三个端点之外的东西过一眼，尤其是 `/v1/completions` 和 `/metrics`。
- 想要一个功能更全的客户端（会话历史、多模型管理、RAG）可以起 [Open WebUI](https://github.com/open-webui/open-webui)，它就是指到 OpenAI 兼容接口上工作的。这里不用它，因为一个不到 200 行的 HTML 更能说清"接口一样就能换客户端"这件事。

**明天要回答的问题：** 一个请求从 HTTP 进来，到第一个 token 出去，在 vLLM 里到底经过了哪些对象？调度器凭什么决定这一步先算谁？→ day 02 读源码。
