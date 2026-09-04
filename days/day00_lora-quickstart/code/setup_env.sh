#!/usr/bin/env bash
# Day 00 容器内环境（在 nvcr.io/nvidia/pytorch:25.08-py3 里跑一次）。
# 版本是 2026-09 在 Jetson AGX Thor 上实跑通的组合。
set -e
pip install -q "trl==1.12.0" "peft==0.20.0" "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0"
# NVIDIA 容器自带 torchao 0.12，peft 0.20 要求 >=0.16 否则直接报错；训练不用它，卸掉
pip uninstall -y -q torchao 2>/dev/null || true
python -c 'import torch, transformers, trl, peft; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)'
