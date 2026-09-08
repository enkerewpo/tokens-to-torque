# Day 02 · 一个请求在 vLLM 里经历了什么

> **Phase** 1 · serving
> **日期** 2026-09-08 · **机器** Jetson AGX Thor · **耗时** ~2h

day 01 起的服务是个黑盒：`curl` 进去，token 出来。今天把它打开，跟着一条请求从 HTTP 进入，经过分词、排队、调度、前向、采样，再变回文本发出去。读的是 vLLM 0.22 的 v1 引擎源码，同时用服务自己报的指标验证每一段。

你会做这几件事：

- 找到源码在容器里的位置，读四个关键函数
- 画清楚请求路径：哪些步骤在 API 进程，哪些在引擎进程，中间怎么通信
- 用 `/metrics` 把一条请求拆成排队、prefill、decode 三段，各自量出毫秒数
- 观察引擎每一步处理多少 token，确认 prefill 和 decode 在调度器眼里是同一件事
- 发多条并发请求，看调度器如何决定这一轮算谁

结束时手上会有：一张请求路径图、三个能自己定位的类名、一条请求的分段耗时。

## 1. 为什么要学这个

day 03 到 day 06 都要改这个服务的行为：算 KV cache 占用、加并发看吞吐、找可用工作点。改之前得知道每个数字由谁产生。KV cache 由 `KVCacheManager` 分配，并发上限由 `Scheduler` 执行，延迟分成几段也是引擎内部的分工决定的。读一遍主循环，后面几天的实验才有解释的落点。

## 2. 背景

### 2.1 两个进程，一条 ZMQ 通道

v1 引擎把工作分给两个进程。API 进程处理 HTTP、分词、拼 chat template、把生成的 token 变回文本；引擎进程只做一件事，循环调度和前向。两者之间用 ZMQ 传消息，消息体由 msgspec 编码。

分开的理由是 Python 的全局解释器锁。HTTP 解析、分词、SSE 编码都是 CPU 活，和引擎主循环放在同一个解释器里会互相抢锁，GPU 因此空等。分成两个进程后，引擎那一侧只剩下调度和前向。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../site_src/assets/fig-request-path-dark.svg">
  <img class="fig" alt="一个请求在 vLLM 里的路径：API 进程做分词和发送，EngineCore 进程里调度、执行、回写输出，两个进程之间用 ZMQ 传消息" src="../../site_src/assets/fig-request-path-light.svg">
</picture>

### 2.2 三个关键的类

读源码时先认这三个，其余都是它们的辅助。

| 类 | 文件 | 负责什么 |
|---|---|---|
| `AsyncLLM` | `vllm/v1/engine/async_llm.py` | API 进程这一侧的入口。每条请求在这里拿到一个输出队列，后台任务把引擎返回的 token 放进队列 |
| `EngineCore` | `vllm/v1/engine/core.py` | 引擎主循环。`step()` 一次做三件事：调度、执行、回写 |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py` | 每一步决定算哪些请求、各算几个 token，并向 `KVCacheManager` 要块 |

另外两个会反复出现：`Request`（`vllm/v1/request.py`）是引擎里一条请求的全部状态；`OutputProcessor`（`vllm/v1/engine/output_processor.py`）把 token id 变回文本并判断停止条件。

### 2.3 主循环：`EngineCore.step()`

主循环的正体只有十几行，结构一目了然：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

三步各自的职责：

1. `schedule()` 产出一个 `SchedulerOutput`：这一轮有哪些请求、每条算多少个 token、各自的 KV 块在哪。
2. `execute_model()` 把这些信息交给 worker，做一次前向并采样。它先返回一个 future，中间那行 `get_grammar_bitmask` 就是趁 GPU 在算时做的 CPU 活。
3. `update_from_output()` 把采样出的 token 追加到各条请求上，判断谁结束了，产出要发回 API 进程的输出。

外层的 `run_busy_loop()` 只是反复调用 `step()`，并在没有请求时阻塞等待新请求。

### 2.4 调度器眼里没有 prefill 和 decode

`Scheduler.schedule()` 开头的注释直接说明了它的模型：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`.

每条请求带两个数：已经算过多少个 token，一共需要算到多少。调度器每一步的工作就是给各请求分配一些 token 配额，让前者追上后者。

这个写法把几种情况统一了。刚到达的请求已算 0 个、需要算 20 个，于是这一步给它 20 个，这就是 prefill。已经在生成的请求每步只差 1 个，于是给它 1 个，这就是 decode。提示很长而配额不够时，这一步只给一部分，下一步继续，这就是分块预填充（chunked prefill）。三种情况在代码里是同一条路径。

配额来自 `token_budget = self.max_num_scheduled_tokens`。调度器先遍历 `running` 队列，再从 `waiting` 队列取新请求，每分配一条就从预算里扣掉。预算耗尽或 KV 块不够时，这一轮就到此为止。

### 2.5 KV 块不够时会抢占

从 `waiting` 取出一条请求，要先向 `KVCacheManager` 申请块（`allocate_slots`）。申请不到时，调度器不会让新请求插队，而是从 `running` 队列尾部往回抢占（`_preempt_request`）：把某条请求的块全部释放，状态改回 `PREEMPTED`，它之前算过的 token 作废，之后重新排队从头 prefill。

被抢占的次数记在 `vllm:num_preemptions_total` 里。这个数不为零，说明 KV cache 相对负载太小，day 03 会把这条关系算清楚。

### 2.6 一条请求的状态机

`RequestStatus`（`vllm/v1/request.py`）列出了全部状态。主要的几个：

| 状态 | 含义 |
|---|---|
| `WAITING` | 已进入引擎，还没被调度过 |
| `RUNNING` | 正在被调度，每步分到一些 token |
| `PREEMPTED` | 被抢占，块已释放，重新排队 |
| `FINISHED_STOPPED` | 遇到结束符或停止字符串 |
| `FINISHED_LENGTH_CAPPED` | 到达 `max_tokens` 或模型长度上限 |
| `FINISHED_ABORTED` | 客户端断开或显式取消 |

枚举里有一行注释值得注意：`PREEMPTED` 之后的所有值都算「已结束」，判断函数就是 `status > RequestStatus.PREEMPTED` 一个比较。所以往这个枚举中间插值会改变语义。

## 3. 动手

### 3.0 前置

day 01 的服务能起来即可。本节的实验用的是同一套脚本，只是换了小模型：机器上还跑着别的服务时，起 9B 会把系统内存挤爆（见 §5）。

```bash
MODEL=<模型路径> PORT=8100 UTIL=0.12 MAXLEN=4096 LORA= NAME=t2t-vllm-day02 \
    bash ../day01_vllm-first-serve/code/serve.sh
until curl -sf localhost:8100/health >/dev/null; do sleep 5; done && echo 就绪
```

`LORA=` 表示不挂 adapter。`NAME` 换一个，免得覆盖 day 01 的容器。

### 3.1 找到源码

源码在容器里，不用另外 clone：

```bash
sudo docker exec t2t-vllm-day02 python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)"
# /usr/local/lib/python3.12/dist-packages/vllm

sudo docker exec t2t-vllm-day02 sed -n '428,460p' \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py
```

要在本机编辑器里读，就把这几个文件复制出来：

```bash
sudo docker cp t2t-vllm-day02:/usr/local/lib/python3.12/dist-packages/vllm/v1 ./vllm-v1
```

§2 引用的行号对应 vLLM 0.22.1，换版本会变，用函数名搜索更稳。

### 3.2 把一条请求拆成三段

`code/trace_one.py` 发一条请求，在请求前后各抓一次 `/metrics`，相减得到这一条的值。vLLM 的指标是 Prometheus 格式的累计量：直方图给 `_sum` 和 `_count`，两次快照的差相除就是这一条请求的平均值。

```bash
python3 code/trace_one.py --url http://localhost:8100 --model <模型名> --max-tokens 64
```

核心是这几行：

```python
def scrape(url):
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"([a-z_:]+(?:_sum|_count|_total)?)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        ...
```

要注意的是单位。`vllm:iteration_tokens_total` 名字里有 `_total`，但它是直方图，值是每一步处理的 token 数，不是时间。脚本按指标分别标了单位，否则会把 1.31 个 token 打印成 1310 毫秒。

### 3.3 看调度器怎么排

`code/watch_sched.py` 同时发 N 条请求，每 100 毫秒读一次两个瞬时值：`vllm:num_requests_running` 和 `vllm:num_requests_waiting`。这两个数直接来自调度器的两个队列长度。

```bash
python3 code/watch_sched.py --url http://localhost:8100 --model <模型名> -n 8 --max-tokens 96
```

采样在主线程，请求在后台线程，输出是一条时间线。想看到排队，就把服务的并发上限调小：`MAXSEQS=4 bash ../day01_vllm-first-serve/code/serve.sh`。

### 3.4 停

```bash
bash ../day01_vllm-first-serve/code/stop.sh    # 或 NAME=t2t-vllm-day02 时用 docker stop
```

## 4. 结果

Jetson AGX Thor（120 W），vLLM 0.22.1（NGC `26.06-py3` 容器），Qwen3.5-0.8B bf16，`--max-model-len 4096 --gpu-memory-utilization 0.12`。用小模型是因为机器上同时跑着别的服务，只剩约 20 GB 可用（见 §5）。请求路径与模型大小无关，换 9B 数值会变，各段的比例关系不变。

启动日志里的三个数：

| | 值 |
|---|---|
| 权重加载 | 1.72 GiB，0.81 s |
| KV cache | 12.02 GiB，718 661 token |
| 4096 token 每条时的最大并发 | 175× |
| torch.compile | 44.0 s |

一条请求（提示 20 个 token，生成 64 个）的分段，来自 `code/trace_one.py`：

| 指标 | 值 | 对应路径上的哪一步 |
|---|---|---|
| `request_queue_time_seconds` | 0.0 ms | 进 `waiting` 到第一次被 `schedule()` 选中 |
| `request_prefill_time_seconds` | 23.4 ms | 提示一次算完 |
| `time_to_first_token_seconds` | 28.2 ms | 客户端看到第一个 token |
| `request_decode_time_seconds` | 676.9 ms | 第一个 token 到最后一个 |
| `request_inference_time_seconds` | 700.3 ms | prefill + decode |
| `inter_token_latency_seconds` | 10.7 ms | 相邻两个 token 的间隔 |
| 客户端量到的端到端 | 705.9 ms | 含 HTTP 和 SSE 的开销 |
| `num_preemptions_total` | 0 | 没有发生抢占 |

空闲的服务上排队时间是 0，这条请求一进来就被下一步调度选中。decode 占了总时间的 96%，与 day 01 §2.3 的结论一致：生成越长，TPOT 越主导。

`vllm:iteration_tokens_total` 的平均值是 **1.31 个 token**。这个数把 §2.4 说的调度模型验证了：一次 prefill 步处理 20 个 token，之后 64 步各处理 1 个，$(20 + 64) / 65 = 1.29$，与实测相符。引擎绝大多数步只算一个 token，这正是 decode 阶段受内存带宽限制的原因。

并发 8 条、每条 96 个 token 时的时间线（`code/watch_sched.py`）：

```text
  时刻 (s)   在跑   在等
    0.44    8    0  ████████
    1.09    8    0  ████████
    1.62    8    0  ████████
```

8 条全部进入同一批，等待队列始终是空的。KV cache 能装 718 661 个 token，而 8 条请求最多用 8 × 4096 ≈ 33 000 个，`allocate_slots` 从不失败，`max_num_seqs` 的默认值也远大于 8。

代价体现在延迟上：

| | 端到端 | 每条的输出 token |
|---|---|---|
| 单条请求 | 0.71 s | 64 |
| 并发 8 条 | 1.69 s（8 条相差不到 10 ms） | 96 |

按每 token 折算，单条是 11.0 ms，并发 8 条是 17.6 ms。每条请求慢了 60%，而单位时间的总产出涨到 4.5 倍。这就是连续批处理的取舍，day 04 会把这条曲线完整扫出来。

8 条请求的端到端时间相差不到 10 毫秒，因为它们每一步都被放进同一个批次，同步前进。

## 5. 踩坑

1. **`--gpu-memory-utilization` 在共享的机器上要按当前空闲量算，不是按总量。** Thor 是统一内存，这个比例乘的是 122 GB 总量。机器上还跑着别的服务时，0.06 换算成 7.3 GB，扣掉权重和 CUDA graph 之后 vLLM 报 `Available KV cache memory: -1.14 GiB` 并退出。改成 0.12 才有 12 GiB 的 KV cache。反过来，空闲内存降到 6.7 GB 之后，同样的 0.12 又变成启动即失败：`Free memory on device cuda:0 (6.71/122.83 GiB) on startup is less than desired GPU memory utilization`。起之前先 `free -g` 看一眼。
2. **`serve.sh` 里 `LORA=` 原来不生效。** 脚本写的是 `${LORA:-默认路径}`，这个写法在变量为空时也会落回默认值，于是显式写 `LORA=` 仍然会去挂 day 00 的 adapter，而那个路径在别的机器上不存在，服务直接起不来。改成 `${LORA-默认路径}`（少一个冒号）之后，只有完全不设这个变量才用默认值。
3. **模型目录必须在挂进容器的路径下。** 把权重放在 `~/models/` 下、用绝对路径传给 `vllm serve`，容器里看不到这个目录，vLLM 会把它当成 Hugging Face 仓库名，报 `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`。放进已经挂载的缓存目录即可。
4. **名字里有 `_total` 的不一定是计数器。** `vllm:iteration_tokens_total` 是直方图，单位是 token 数。按时间打印会得到「1310 毫秒」这种数字，看起来还挺合理，所以特别容易错。判断方法是看 `/metrics` 里有没有对应的 `_bucket` 行。
5. **容器内没有到 Hugging Face 的路由。** 宿主机能连通，桥接网络里的容器连不上，容器里预设的代理又指向 `127.0.0.1`，在容器里就是它自己。结论是模型要么提前下好，要么在宿主机上下载再挂进去。
6. **指标要等请求结束后才写入。** 请求返回和指标更新之间有一小段延迟，紧接着抓 `/metrics` 会拿到旧值，导致差值为零。脚本里等了 0.5 秒。

## 6. 延伸

- vLLM 的 [V1 架构设计文档](https://docs.vllm.ai/en/latest/design/arch_overview.html)，看完源码再读一遍，能确认自己没有理解偏。
- 想看调度器的完整决策过程，把日志级别调到 DEBUG：`-e VLLM_LOGGING_LEVEL=DEBUG`。输出很多，建议只在单条请求时开。

day 03 要回答的问题：KV cache 那 12.02 GiB 是怎么算出来的？改 `--max-model-len` 和 `--gpu-memory-utilization`，预测的容量和实测差多少？
