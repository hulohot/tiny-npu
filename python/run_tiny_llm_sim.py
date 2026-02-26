"""tiny-npu real-weights demo runner.

Capabilities:
- Export + quantize + pack real tiny GPT-2 weights (optional --prepare)
- Reference generation via full HF model (1+ tokens)
- Simulated token via INT8 projection-only path (first generated token)
- Optional Verilator smoke execution
- Optional interactive REPL mode
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:  # module mode: python -m python.run_tiny_llm_sim
    from python.tools.export_gpt2_weights import export_gpt2_weights
    from python.tools.quantize_pack import quantize_and_pack
except ModuleNotFoundError:  # script mode: python python/run_tiny_llm_sim.py
    from tools.export_gpt2_weights import export_gpt2_weights
    from tools.quantize_pack import quantize_and_pack


def _q_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def _sim_logits_from_hidden(last_hidden: np.ndarray, lm_head_w: np.ndarray) -> np.ndarray:
    """INT8 simulated projection (pre-softmax logits approximation).

    Models GEMM accumulator path: int8 x int8 -> int32, then dequant.
    """
    x_q, sx = _q_int8(last_hidden.astype(np.float32))
    w_q, sw = _q_int8(lm_head_w.astype(np.float32))
    acc_i32 = x_q.astype(np.int32) @ w_q.astype(np.int32).T
    logits = acc_i32.astype(np.float32) * (sx * sw)
    return logits


def _apply_repetition_penalty(logits: np.ndarray, seen_ids: list[int], penalty: float) -> np.ndarray:
    if penalty is None or penalty <= 1.0 or not seen_ids:
        return logits
    out = logits.copy()
    for tid in set(seen_ids):
        if out[tid] > 0:
            out[tid] /= penalty
        else:
            out[tid] *= penalty
    return out


def _sample_from_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits / max(temperature, 1e-6)

    # top-k filter
    if top_k and top_k > 0 and top_k < scaled.shape[0]:
        idx = np.argpartition(scaled, -top_k)[-top_k:]
        mask = np.full_like(scaled, -np.inf)
        mask[idx] = scaled[idx]
        scaled = mask

    # softmax
    m = np.max(scaled)
    probs = np.exp(scaled - m)
    probs_sum = np.sum(probs)
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return int(np.argmax(logits))
    probs /= probs_sum

    # top-p nucleus filter
    if top_p is not None and 0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        csum = np.cumsum(sorted_probs)
        keep_n = int(np.searchsorted(csum, top_p, side="left")) + 1
        keep = sorted_idx[:keep_n]
        new_probs = np.zeros_like(probs)
        new_probs[keep] = probs[keep]
        z = np.sum(new_probs)
        if z > 0:
            probs = new_probs / z

    return int(rng.choice(len(probs), p=probs))


def _run_reference_generation(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> tuple[list[int], str, np.ndarray, list[int]]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seen_ids = input_ids[0].tolist()
    rng = np.random.default_rng(seed)

    generated: list[int] = []
    first_hidden: np.ndarray | None = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

        logits = out.logits[0, -1, :].detach().cpu().numpy()
        if first_hidden is None:
            first_hidden = out.hidden_states[-1][0, -1, :].detach().cpu().numpy()

        adjusted = _apply_repetition_penalty(logits, seen_ids + generated, repetition_penalty)
        next_id = _sample_from_logits(
            adjusted,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated.append(next_id)
        next_tok = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    text = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
    return generated, text, first_hidden, seen_ids


def run_demo(
    prompt: str,
    model_name: str,
    datadir: Path,
    prepare: bool,
    run_verilator: bool,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
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

    gen_ids, gen_text, first_hidden, prompt_token_ids = _run_reference_generation(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )

    # Simulated path (INT8 projection from real hidden state + real lm_head)
    lm_head = np.load(datadir / "lm_head.npy")
    sim_logits = _sim_logits_from_hidden(first_hidden, lm_head)
    sim_token_id = int(np.argmax(sim_logits))
    sim_token = tokenizer.decode([sim_token_id], clean_up_tokenization_spaces=False)

    result = {
        "model": model_name,
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "decoding": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        },
        "reference": {
            "token_id": int(gen_ids[0]),
            "token": tokenizer.decode([int(gen_ids[0])], clean_up_tokenization_spaces=False),
            "generated_token_ids": [int(x) for x in gen_ids],
            "generated_text": gen_text,
            "note": "Full HF model generation (ground truth reference)",
        },
        "simulated": {
            "token_id": sim_token_id,
            "token": sim_token,
            "note": "INT8 simulated projection only (first-token projection from real hidden state + real lm_head)",
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


def _run_interactive(args: argparse.Namespace) -> None:
    print("tiny-npu interactive demo (type 'exit' or Ctrl-D to quit)")
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        result = run_demo(
            prompt=prompt,
            model_name=args.model,
            datadir=Path(args.datadir),
            prepare=False,
            run_verilator=args.run_verilator_smoke,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

        print(f"ref> {result['reference']['generated_text']!r}")
        print(f"sim> {result['simulated']['token']!r}  (first-token projection-only)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny-npu real-weights LLM simulation demo")
    parser.add_argument("--prompt", default="Hello tiny NPU", help="Prompt text")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="HF model id")
    parser.add_argument("--datadir", default="demo_data", help="Path for exported/quantized data")
    parser.add_argument("--prepare", action="store_true", help="Export + quantize before inference")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--max-new-tokens", type=int, default=12, help="Reference generation length")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff (0 disables)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty >1.0 discourages repeats")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument(
        "--run-verilator-smoke",
        action="store_true",
        help="Run existing test_gpt2_block binary if available",
    )
    args = parser.parse_args()

    if args.prepare:
        export_gpt2_weights(model_name=args.model, outdir=args.datadir)
        quantize_and_pack(indir=args.datadir, outdir=args.datadir)

    if args.interactive:
        _run_interactive(args)
        return

    result = run_demo(
        prompt=args.prompt,
        model_name=args.model,
        datadir=Path(args.datadir),
        prepare=False,
        run_verilator=args.run_verilator_smoke,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
