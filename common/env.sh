# 每次在 Thor 上起 Python GPU 任务前 source 一下。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export HF_HOME=$HOME/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像；Thor 没有直连 HF
export MUJOCO_GL=egl                       # headless 渲染（LIBERO 等）
# 统一内存下 nvidia-smi 的 per-process 显存显示 [N/A] 是正常的，不是 bug，别查。
