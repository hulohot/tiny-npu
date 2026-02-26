"""Prompt variation harness for interactive tiny-npu demo quality checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from python.run_tiny_llm_sim import run_demo


def _load_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate response variation across a prompt set")
    ap.add_argument("--prompts", default="benchmarks/prompts/first_token_prompts.txt", help="Prompt file path")
    ap.add_argument("--outdir", default="benchmarks/results/prompt_variation", help="Output directory")
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--datadir", default="demo_data")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompts = _load_prompts(Path(args.prompts))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    ref_first_tokens: list[str] = []

    for i, prompt in enumerate(prompts):
        result = run_demo(
            prompt=prompt,
            model_name=args.model,
            datadir=Path(args.datadir),
            prepare=False,
            run_verilator=False,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed + i,
        )

        ref_tok = result["reference"]["token"]
        sim_tok = result["simulated"]["token"]
        ref_first_tokens.append(ref_tok)
        rows.append({"prompt": prompt, "reference_token": ref_tok, "simulated_token": sim_tok})

    csv_path = outdir / "prompt_variation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "reference_token", "simulated_token"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "prompt_count": len(prompts),
        "unique_reference_tokens": len(set(ref_first_tokens)),
        "variation_ratio": (len(set(ref_first_tokens)) / len(prompts)) if prompts else 0.0,
        "csv": str(csv_path),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
