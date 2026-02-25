"""Quantize exported FP32 tensors to INT8 and pack into a single binary blob."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"TNPUWGT1"


def quantize_int8_symmetric(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8 quantization."""
    x = x.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_and_pack(indir: str = "demo_data", outdir: str = "demo_data") -> Path:
    """Read *.npy tensors, write *.int8.npy + manifest + packed weights.bin.

    Pack format assumptions:
      - little-endian
      - one contiguous int8 payload region
      - per-tensor symmetric scale in manifest (no zero-point)
      - header: magic(8), version(u32), tensor_count(u32), manifest_len(u32)
    """
    src = Path(indir)
    dst = Path(outdir)
    dst.mkdir(parents=True, exist_ok=True)

    tensor_files = sorted(p for p in src.glob("*.npy") if not p.name.endswith(".int8.npy"))
    if not tensor_files:
        raise FileNotFoundError(f"No .npy tensors found in {src}")

    manifest = {
        "format": "tiny-npu-int8-pack-v1",
        "assumptions": {
            "quantization": "per-tensor symmetric int8",
            "zero_point": 0,
            "endianness": "little",
        },
        "tensors": {},
    }

    payload = bytearray()
    offset = 0
    for tpath in tensor_files:
        name = tpath.stem
        arr = np.load(tpath)
        q, scale = quantize_int8_symmetric(arr)

        q_path = dst / f"{name}.int8.npy"
        np.save(q_path, q)

        raw = q.tobytes(order="C")
        payload.extend(raw)

        manifest["tensors"][name] = {
            "shape": list(arr.shape),
            "orig_dtype": str(arr.dtype),
            "packed_dtype": "int8",
            "scale": scale,
            "offset": offset,
            "nbytes": len(raw),
        }
        offset += len(raw)

    manifest_json = json.dumps(manifest, indent=2)
    (dst / "quant_manifest.json").write_text(manifest_json + "\n")

    weights_bin = dst / "weights.bin"
    header = struct.pack("<8sIII", MAGIC, 1, len(tensor_files), len(manifest_json.encode("utf-8")))
    with weights_bin.open("wb") as f:
        f.write(header)
        f.write(manifest_json.encode("utf-8"))
        f.write(payload)

    print(f"Quantized {len(tensor_files)} tensors")
    print(f"Wrote manifest: {(dst / 'quant_manifest.json').resolve()}")
    print(f"Wrote packed weights: {weights_bin.resolve()}")
    return weights_bin


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and pack tiny GPT-2 weights")
    parser.add_argument("--indir", default="demo_data", help="Input directory with FP32 .npy tensors")
    parser.add_argument("--outdir", default="demo_data", help="Output directory")
    args = parser.parse_args()
    quantize_and_pack(indir=args.indir, outdir=args.outdir)


if __name__ == "__main__":
    main()
