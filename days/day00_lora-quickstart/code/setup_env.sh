#!/usr/bin/env bash
# 容器内环境。版本是 2026-09 在 Jetson AGX Thor 上实跑通的组合。
# 装好一次就够，重复跑会秒退。
set -e
PKGS=("trl==1.12.0" "peft==0.20.0" "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0")

echo "检查依赖…"
if python - "${PKGS[@]}" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version, PackageNotFoundError
for spec in sys.argv[1:]:
    name, want = spec.split("==")
    try:
        if version(name) != want:
            sys.exit(1)
    except PackageNotFoundError:
        sys.exit(1)
PY
then
  echo "依赖已就绪，跳过安装。"
else
  echo
  echo "开始安装，需要几分钟，中途不要 Ctrl+C —— 装到一半退出会留下缺包的环境。"
  echo "要装：${PKGS[*]}"
  echo
  pip install --progress-bar on "${PKGS[@]}"
  # NVIDIA 容器自带 torchao 0.12，peft 0.20 要求 >=0.16 否则直接报错；训练用不到，卸掉
  pip uninstall -y -q torchao 2>/dev/null || true
fi

echo
python -c 'import torch, transformers, trl, peft; print("torch", torch.__version__, "| cuda", torch.cuda.is_available()); print("transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__)'
echo "环境就绪。"
