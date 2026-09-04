#!/usr/bin/env bash
# Install the pinned dependency set inside the container.
# Verified 2026-09 on Jetson AGX Thor with nvcr.io/nvidia/pytorch:25.08-py3.
# Safe to re-run: it checks versions first and exits immediately if satisfied.
set -e
G=$'\033[38;5;148m'; D=$'\033[2m'; B=$'\033[1m'; Y=$'\033[33m'; R=$'\033[0m'
PKGS=("trl==1.12.0" "peft==0.20.0" "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0")

printf '\n%s▸ Checking dependencies%s\n' "$B" "$R"
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
  printf '  %s✓%s already satisfied, nothing to install\n' "$G" "$R"
else
  printf '  %s!%s installing — takes a few minutes, do NOT Ctrl+C%s\n' "$Y" "$R" "$R"
  printf '  %s(interrupting leaves a half-installed environment)%s\n\n' "$D" "$R"
  for p in "${PKGS[@]}"; do printf '  %s· %s%s\n' "$D" "$p" "$R"; done
  echo
  pip install --progress-bar on "${PKGS[@]}"
  # The NVIDIA image ships torchao 0.12; peft 0.20 refuses anything below 0.16.
  # Training does not use it, so remove it rather than upgrade.
  pip uninstall -y -q torchao 2>/dev/null || true
fi

printf '\n%s▸ Versions%s\n' "$B" "$R"
python - <<'PY'
import torch, transformers, trl, peft
print(f"  torch        {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
print(f"  trl          {trl.__version__}")
print(f"  peft         {peft.__version__}")
PY
printf '\n%s%sEnvironment ready%s\n\n' "$G" "$B" "$R"
