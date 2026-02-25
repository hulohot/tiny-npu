"""Export a tiny GPT-2 family model into demo_data artifacts for tiny-npu flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "sshleifer/tiny-gpt2"


def export_gpt2_weights(model_name: str = DEFAULT_MODEL, outdir: str = "demo_data") -> Path:
    """Export a minimal set of real GPT-2 weights and tokenizer assets.

    Artifacts written:
      - model_meta.json
      - *.npy tensors (fp32)
      - tokenizer/*
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Missing dependency. Install with: pip install torch transformers"
        ) from exc

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token exists for robust batch tokenization in downstream scripts.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    state = model.state_dict()

    # Minimal viable tensor set for first-token path + future expansion hooks.
    tensor_map = {
        "wte": state["transformer.wte.weight"],
        "wpe": state["transformer.wpe.weight"],
        "ln_f_weight": state["transformer.ln_f.weight"],
        "ln_f_bias": state["transformer.ln_f.bias"],
    }

    # lm_head may be tied to wte, but export explicitly for clarity.
    if "lm_head.weight" in state:
        tensor_map["lm_head"] = state["lm_head.weight"]
    else:
        tensor_map["lm_head"] = state["transformer.wte.weight"]

    # Export first transformer block for future real RTL wiring.
    for key in [
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.ln_1.bias",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.attn.c_proj.bias",
        "transformer.h.0.ln_2.weight",
        "transformer.h.0.ln_2.bias",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_fc.bias",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.0.mlp.c_proj.bias",
    ]:
        if key in state:
            safe_name = key.replace("transformer.h.0.", "block0_").replace(".", "_")
            tensor_map[safe_name] = state[key]

    exported = {}
    for name, tensor in tensor_map.items():
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        np.save(out / f"{name}.npy", arr)
        exported[name] = {
            "shape": list(arr.shape),
            "dtype": "float32",
        }

    tok_dir = out / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    meta = {
        "model_name": model_name,
        "n_layer": int(model.config.n_layer),
        "n_head": int(model.config.n_head),
        "n_embd": int(model.config.n_embd),
        "vocab_size": int(model.config.vocab_size),
        "exported_tensors": exported,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Exported {len(exported)} tensors to {out.resolve()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tiny GPT-2 weights to demo_data")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--outdir", default="demo_data", help="Output directory")
    args = parser.parse_args()
    export_gpt2_weights(model_name=args.model, outdir=args.outdir)


if __name__ == "__main__":
    main()
