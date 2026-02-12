"""
Quantize FP32 weights to INT8 and pack into binary format for NPU.
"""

import numpy as np
import os

def quantize_and_pack(indir="demo_data", outdir="demo_data"):
    """Quantize weights to INT8 and pack."""
    print("Quantizing weights to INT8...")
    
    # TODO: Implement INT8 quantization
    print(f"TODO: Read FP32 weights from {indir}")
    print(f"TODO: Write quantized weights to {outdir}/weights.bin")
    
if __name__ == "__main__":
    outdir = os.environ.get("DEMO_OUTDIR", "demo_data")
    quantize_and_pack(outdir, outdir)
