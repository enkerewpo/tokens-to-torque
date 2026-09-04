# Source before launching any Python GPU job.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export HF_HOME=$HOME/.cache/huggingface
# The newer xet transport ignores HTTPS_PROXY and silently stalls at 0 MB/s
# behind a proxy. Plain HTTP download works.
export HF_HUB_DISABLE_XET=1
# Pick one to match your network: the official site (direct or via proxy),
# or a mirror.
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export MUJOCO_GL=egl                       # headless rendering (LIBERO and friends)
# On unified memory, nvidia-smi reports [N/A] for per-process memory.
# That is normal on tegra, not a bug.
