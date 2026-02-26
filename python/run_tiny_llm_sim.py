"""Minimal tiny-npu real-weights first-token demo.

Flow:
  1) (optional) export HF tiny GPT-2 weights
  2) (optional) quantize + pack to INT8
  3) tokenize prompt
  4) compute reference next token (full HF model)
  5) compute simulated next token using INT8 projection path
  6) (optional) run existing Verilator smoke binary (test_gpt2_block)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from python.tools.export_gpt2_weights import export_gpt2_weights
from python.tools.quantize_pack import quantize_and_pack


def _q_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def _sim_logits_from_hidden(last_hidden: np.ndarray, lm_head_w: np.ndarray) -> np.ndarray:
    """INT8 simulated projection (pre-softmax logits approximation).

    This models the GEMM accumulator path: int8 x int8 -> int32, then dequant.
    """
    x_q, sx = _q_int8(last_hidden.astype(np.float32))          # [H]
    w_q, sw = _q_int8(lm_head_w.astype(np.float32))            # [V, H]
    acc_i32 = x_q.astype(np.int32) @ w_q.astype(np.int32).T    # [V]
    logits = acc_i32.astype(np.float32) * (sx * sw)
    return logits


def run_demo(
    prompt: str,
    model_name: str,
    datadir: Path,
    prepare: bool,
    run_verilator: bool,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency. Install with: pip install torch transformers") from exc

    if prepare:
        export_gpt2_weights(model_name=model_name, outdir=str(datadir))
        quantize_and_pack(indir=str(datadir), outdir=str(datadir))

    if not (datadir / "lm_head.npy").exists():
        raise FileNotFoundError(
            f"{datadir}/lm_head.npy missing. Run with --prepare or run export_gpt2_weights.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    # Reference (full model)
    ref_logits = out.logits[0, -1, :].detach().cpu().numpy()
    ref_token_id = int(np.argmax(ref_logits))
    ref_token = tokenizer.decode([ref_token_id])

    # Simulated path (INT8 projection from real hidden state + real lm_head)
    last_hidden = out.hidden_states[-1][0, -1, :].detach().cpu().numpy()
    lm_head = np.load(datadir / "lm_head.npy")
    sim_logits = _sim_logits_from_hidden(last_hidden, lm_head)
    sim_token_id = int(np.argmax(sim_logits))
    sim_token = tokenizer.decode([sim_token_id])

    result = {
        "model": model_name,
        "prompt": prompt,
        "prompt_token_ids": inputs["input_ids"][0].tolist(),
        "reference": {
            "token_id": ref_token_id,
            "token": ref_token,
            "note": "Full HF model forward pass (ground truth reference)",
        },
        "simulated": {
            "token_id": sim_token_id,
            "token": sim_token,
            "note": "INT8 simulated projection only (real hidden state + real lm_head)",
        },
    }

    if run_verilator:
        repo_root = Path(__file__).resolve().parent.parent
        sim_bin = repo_root / "sim/verilator/build/test_gpt2_block"
        if sim_bin.exists():
            proc = subprocess.run(
                [str(sim_bin)],
                capture_output=True,
                text=True,
                cwd=str(sim_bin.parent),
            )
            result["verilator_smoke"] = {
                "returncode": proc.returncode,
                "cwd": str(sim_bin.parent),
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
            }
        else:
            result["verilator_smoke"] = {
                "skipped": True,
                "reason": f"{sim_bin} not found (build sim/verilator first)",
            }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny-npu minimal real-weights LLM simulation")
    parser.add_argument("--prompt", default="Hello tiny NPU", help="Prompt text")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="HF model id")
    parser.add_argument("--datadir", default="demo_data", help="Path for exported/quantized data")
    parser.add_argument("--prepare", action="store_true", help="Export + quantize before inference")
    parser.add_argument(
        "--run-verilator-smoke",
        action="store_true",
        help="Run existing test_gpt2_block binary if available",
    )
    args = parser.parse_args()

    result = run_demo(
        prompt=args.prompt,
        model_name=args.model,
        datadir=Path(args.datadir),
        prepare=args.prepare,
        run_verilator=args.run_verilator_smoke,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
