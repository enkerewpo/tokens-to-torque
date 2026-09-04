# 每次在 Thor 上起 Python GPU 任务前 source 一下。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_DISABLE_XET=1                # 新的 xet 传输不走 HTTPS_PROXY，走代理时会静默 0 MB/s
# 按自己的网络二选一：直连/代理用官方站，否则用镜像
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export MUJOCO_GL=egl                       # headless 渲染（LIBERO 等）
# 统一内存下 nvidia-smi 的 per-process 显存显示 [N/A] 是正常的，不是 bug，别查。
