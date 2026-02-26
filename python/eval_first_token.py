"""Evaluate first-token agreement between reference and simulated paths.

Writes:
- CSV rows per prompt
- Summary JSON with match rate
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from python.run_tiny_llm_sim import _sim_logits_from_hidden
from python.tools.export_gpt2_weights import export_gpt2_weights
from python.tools.quantize_pack import quantize_and_pack


def read_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate first-token reference vs simulated agreement")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--datadir", default="demo_data")
    parser.add_argument("--prompts", default="benchmarks/prompts/first_token_prompts.txt")
    parser.add_argument("--outdir", default="benchmarks/results/first_token_eval")
    parser.add_argument("--prepare", action="store_true", help="Export+quantize weights before evaluation")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency. Install with: pip install torch transformers") from exc

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        export_gpt2_weights(model_name=args.model, outdir=str(datadir))
        quantize_and_pack(indir=str(datadir), outdir=str(datadir))

    lm_head_path = datadir / "lm_head.npy"
    if not lm_head_path.exists():
        raise FileNotFoundError(f"{lm_head_path} missing. Run with --prepare first.")

    prompts = read_prompts(Path(args.prompts))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    lm_head = np.load(lm_head_path)

    rows: list[dict] = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)

        ref_logits = out.logits[0, -1, :].detach().cpu().numpy()
        ref_id = int(np.argmax(ref_logits))
        ref_tok = tokenizer.decode([ref_id], clean_up_tokenization_spaces=False)

        last_hidden = out.hidden_states[-1][0, -1, :].detach().cpu().numpy()
        sim_logits = _sim_logits_from_hidden(last_hidden, lm_head)
        sim_id = int(np.argmax(sim_logits))
        sim_tok = tokenizer.decode([sim_id], clean_up_tokenization_spaces=False)

        rows.append(
            {
                "prompt": p,
                "reference_token_id": ref_id,
                "reference_token": ref_tok,
                "simulated_token_id": sim_id,
                "simulated_token": sim_tok,
                "match": int(ref_id == sim_id),
            }
        )

    csv_path = outdir / "first_token_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "reference_token_id",
                "reference_token",
                "simulated_token_id",
                "simulated_token",
                "match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    matches = sum(r["match"] for r in rows)
    summary = {
        "model": args.model,
        "prompts_file": str(args.prompts),
        "num_prompts": total,
        "matches": matches,
        "match_rate": (matches / total) if total else 0.0,
        "csv": str(csv_path),
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
