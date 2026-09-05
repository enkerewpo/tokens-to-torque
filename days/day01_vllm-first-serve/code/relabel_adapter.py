#!/usr/bin/env python3
"""把 adapter 的权重键名改成推理引擎认得的形式。

    python code/relabel_adapter.py --in <训练出来的 adapter> --out <新目录> \
        --prefix language_model. --keep q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

为什么需要这一步：训练时用 AutoModelForCausalLM 加载的是纯文本模型，权重叫
`...model.layers.N.self_attn.q_proj`；而 vLLM 把同一个模型当多模态模型实例化，
文本塔挂在 `language_model` 下面。名字对不上，adapter 会被静默忽略——服务照常
起来，回答和 base 一模一样。
"""
import argparse, json, pathlib, re, shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="language_model.",
                    help="插在 model. 之后的前缀；空字符串表示不加")
    ap.add_argument("--keep", default="",
                    help="逗号分隔的模块名，只保留这些；留空表示全保留")
    a = ap.parse_args()

    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = load_file(str(src / "adapter_model.safetensors"))
    keep = [k for k in a.keep.split(",") if k]

    new, dropped = {}, 0
    for k, v in w.items():
        if keep and not any(f".{m}." in k for m in keep):
            dropped += 1
            continue
        nk = k.replace("base_model.model.model.", f"base_model.model.model.{a.prefix}") \
             if a.prefix else k
        new[nk] = v
    save_file(new, str(out / "adapter_model.safetensors"))

    cfg = json.loads((src / "adapter_config.json").read_text())
    if keep:
        cfg["target_modules"] = keep
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    for extra in ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)

    print(f"保留 {len(new)} 个张量，丢掉 {dropped} 个")
    print("新键名示例：", next(iter(new)))


if __name__ == "__main__":
    main()
