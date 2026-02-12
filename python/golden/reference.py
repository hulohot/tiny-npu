"""
Golden reference implementations for NPU verification.
All functions produce bit-exact INT8 results matching the hardware.
"""

import numpy as np
from typing import Optional, Tuple


def quantize_tensor(x: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
    """
    Symmetric per-tensor INT8 quantization.
    
    Args:
        x: FP32 tensor to quantize
        scale: Quantization scale (if None, computed from max abs value)
    
    Returns:
        (quantized INT8 tensor, scale used)
    """
    if scale is None:
        max_abs = np.max(np.abs(x))
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
    
    x_q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return x_q, scale


def dequantize_tensor(x_q: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 tensor back to FP32."""
    return x_q.astype(np.float32) * scale


def gemm_golden(
    A: np.ndarray,  # [M, K] INT8
    B: np.ndarray,  # [K, N] INT8  
    scale: int = 1,
    shift: int = 0,
    accumulate: bool = False,
    C_prev: Optional[np.ndarray] = None  # [M, N] INT8 for accumulation
) -> np.ndarray:
    """
    Golden INT8 GEMM with INT32 accumulation and requantization.
    
    Matches hardware: accumulator = A @ B (INT32)
                     output = (accumulator * scale) >> shift
    
    Args:
        A: Left matrix [M, K] INT8
        B: Right matrix [K, N] INT8
        scale: Requantization scale (multiply before shift)
        shift: Right shift amount (typically ceil(log2(K)) + extra)
        accumulate: Add to C_prev instead of zero
        C_prev: Previous output for accumulation [M, N] INT8
    
    Returns:
        C: Output matrix [M, N] INT8
    """
    assert A.dtype == np.int8, f"A must be INT8, got {A.dtype}"
    assert B.dtype == np.int8, f"B must be INT8, got {B.dtype}"
    
    # INT32 accumulation
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    acc = A_i32 @ B_i32
    
    # Accumulate if requested
    if accumulate and C_prev is not None:
        acc = acc + C_prev.astype(np.int32)
    
    # Requantization: (acc * scale) >> shift
    # Hardware uses round-half-up: (acc * scale + (1 << (shift-1))) >> shift
    if shift > 0:
        rounded = (acc * scale + (1 << (shift - 1))) >> shift
    else:
        rounded = acc * scale
    
    # Clamp to INT8 range and return
    return np.clip(rounded, -128, 127).astype(np.int8)


def softmax_golden(
    x: np.ndarray,  # [M, N] INT8
    causal: bool = False
) -> np.ndarray:
    """
    Golden fixed-point softmax with optional causal mask.
    
    Hardware uses INT8 input -> FP32 compute -> INT8 output.
    This is acceptable for attention where precision is less critical.
    
    Args:
        x: Input tensor [M, N] INT8
        causal: If True, apply causal (lower-triangular) mask
    
    Returns:
        probs: Softmax probabilities [M, N] INT8 (sum to ~1 per row)
    """
    assert x.dtype == np.int8, f"x must be INT8, got {x.dtype}"
    
    # Convert to FP32 for stable computation
    x_f = x.astype(np.float32)
    
    # Apply causal mask if requested
    if causal:
        M, N = x.shape
        mask = np.triu(np.ones((M, N)), k=1) * -1e9
        x_f = x_f + mask
    
    # Numerically stable softmax
    x_max = np.max(x_f, axis=-1, keepdims=True)
    exp_x = np.exp(x_f - x_max)
    probs_f = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    # Quantize back to INT8 (range [0, 1] maps to [0, 127])
    # We use 127 as max to avoid overflow in later GEMM
    return np.clip(np.round(probs_f * 127), 0, 127).astype(np.int8)


def layernorm_golden(
    x: np.ndarray,  # [M, N] INT8
    gamma: np.ndarray,  # [N] INT8 (scale)
    beta: np.ndarray,   # [N] INT8 (shift)
    eps: float = 1e-5
) -> np.ndarray:
    """
    Golden layer normalization.
    
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Args:
        x: Input tensor [M, N] INT8
        gamma: Scale parameter [N] INT8
        beta: Shift parameter [N] INT8
        eps: Small constant for numerical stability
    
    Returns:
        y: Normalized output [M, N] INT8
    """
    assert x.dtype == np.int8, f"x must be INT8, got {x.dtype}"
    
    # Convert to FP32 for stable mean/variance computation
    x_f = x.astype(np.float32)
    
    # Compute mean and variance per row
    mean = np.mean(x_f, axis=-1, keepdims=True)
    var = np.var(x_f, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x_f - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    gamma_f = gamma.astype(np.float32)
    beta_f = beta.astype(np.float32)
    y_f = x_norm * gamma_f + beta_f
    
    # Quantize back to INT8
    return np.clip(np.round(y_f), -128, 127).astype(np.int8)


def gelu_golden(x: np.ndarray) -> np.ndarray:
    """
    Golden GELU activation.
    
    Uses tanh approximation:
    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Args:
        x: Input tensor INT8
    
    Returns:
        y: GELU output INT8
    """
    assert x.dtype == np.int8, f"x must be INT8, got {x.dtype}"
    
    x_f = x.astype(np.float32)
    
    # GELU approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x_f + 0.044715 * x_f**3)))
    y_f = x_f * cdf
    
    return np.clip(np.round(y_f), -128, 127).astype(np.int8)


def vec_add_golden(
    a: np.ndarray,  # [M, N] INT8
    b: np.ndarray   # [M, N] INT8
) -> np.ndarray:
    """Saturating element-wise addition."""
    assert a.dtype == np.int8 and b.dtype == np.int8
    result = a.astype(np.int16) + b.astype(np.int16)
    return np.clip(result, -128, 127).astype(np.int8)


def vec_mul_golden(
    a: np.ndarray,  # [M, N] INT8
    b: np.ndarray   # [M, N] INT8
) -> np.ndarray:
    """Element-wise multiplication (Q7.8 fixed-point, result Q0.7)."""
    assert a.dtype == np.int8 and b.dtype == np.int8
    # Multiply and shift right by 7 (Q7 * Q7 = Q14, shift to Q7)
    result = (a.astype(np.int16) * b.astype(np.int16)) >> 7
    return np.clip(result, -128, 127).astype(np.int8)


def attention_head_golden(
    x: np.ndarray,      # [seq_len, hidden] INT8 - input
    w_q: np.ndarray,    # [hidden, head_dim] INT8
    w_k: np.ndarray,    # [hidden, head_dim] INT8
    w_v: np.ndarray,    # [hidden, head_dim] INT8
    seq_len: int,
    head_dim: int,
    causal: bool = True
) -> np.ndarray:
    """
    Golden single-head attention computation.
    
    Args:
        x: Input activations [seq_len, hidden]
        w_q, w_k, w_v: Weight matrices
        seq_len: Sequence length
        head_dim: Head dimension
        causal: Apply causal mask
    
    Returns:
        context: Attention output [seq_len, head_dim] INT8
    """
    # Q, K, V projections
    q = gemm_golden(x, w_q, scale=1, shift=7)  # [seq_len, head_dim]
    k = gemm_golden(x, w_k, scale=1, shift=7)  # [seq_len, head_dim]
    v = gemm_golden(x, w_v, scale=1, shift=7)  # [seq_len, head_dim]
    
    # Attention scores: Q @ K^T / sqrt(head_dim)
    scores = gemm_golden(q, k.T, scale=1, shift=4)  # [seq_len, seq_len]
    # Note: shift includes sqrt scaling approximation
    
    # Softmax
    probs = softmax_golden(scores, causal=causal)  # [seq_len, seq_len]
    
    # Context: probs @ V
    context = gemm_golden(probs, v, scale=1, shift=7)  # [seq_len, head_dim]
    
    return context


# =============================================================================
# Test utilities
# =============================================================================

def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    name: str = "tensor",
    tolerance: int = 0
) -> bool:
    """
    Compare two tensors and report differences.
    
    Args:
        a, b: Tensors to compare
        name: Name for error reporting
        tolerance: Allowed difference (0 for bit-exact)
    
    Returns:
        True if match within tolerance
    """
    if a.shape != b.shape:
        print(f"FAIL {name}: Shape mismatch {a.shape} vs {b.shape}")
        return False
    
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    max_diff = np.max(diff)
    mismatches = np.sum(diff > tolerance)
    total = a.size
    
    if max_diff <= tolerance:
        print(f"PASS {name}: {total}/{total} match (max_diff={max_diff})")
        return True
    else:
        print(f"FAIL {name}: {mismatches}/{total} mismatch (max_diff={max_diff})")
        # Print first few mismatches
        mismatch_idx = np.where(diff > tolerance)
        for i in range(min(5, len(mismatch_idx[0]))):
            idx = tuple(m[idx] for m in mismatch_idx)
            print(f"  [{idx}]: {a[idx]} vs {b[idx]} (diff={diff[idx]})")
        return False


if __name__ == "__main__":
    # Quick self-test
    print("Testing golden reference implementations...")
    
    # Test GEMM
    A = np.random.randint(-10, 10, (4, 8), dtype=np.int8)
    B = np.random.randint(-10, 10, (8, 4), dtype=np.int8)
    C = gemm_golden(A, B, scale=1, shift=7)
    print(f"GEMM: {A.shape} @ {B.shape} = {C.shape}")
    
    # Test softmax
    S = np.random.randint(-20, 20, (4, 4), dtype=np.int8)
    P = softmax_golden(S, causal=True)
    print(f"Softmax: {S.shape} -> {P.shape}, row sums ~{np.sum(P, axis=1)}")
    
    # Test GELU
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int8)
    y = gelu_golden(x)
    print(f"GELU: {x} -> {y}")
    
    print("\nAll golden functions loaded successfully!")
