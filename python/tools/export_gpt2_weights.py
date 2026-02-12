"""
Export GPT-2 weights from HuggingFace and prepare for NPU quantization.
"""

import numpy as np
import os

def export_gpt2_weights(outdir="demo_data"):
    """Export GPT-2 weights from HuggingFace."""
    print("Exporting GPT-2 weights...")
    os.makedirs(outdir, exist_ok=True)
    
    # TODO: Implement actual weight export from transformers
    print(f"TODO: Export weights to {outdir}")
    
if __name__ == "__main__":
    import sys
    outdir = os.environ.get("DEMO_OUTDIR", "demo_data")
    export_gpt2_weights(outdir)
