# Tiny NPU Architecture Specification

## 1. Overview

Tiny NPU is a minimal neural processing unit designed for transformer inference. It executes quantized INT8 neural networks with INT32 accumulation, targeting small LLMs (GPT-2 small, MicroLlama, etc.).

### 1.1 Design Principles

1. **Testability First**: Every module must have a standalone testbench
2. **Modularity**: Independent engines communicate via well-defined interfaces
3. **Determinism**: Bit-exact results vs reference Python implementation
4. **Simplicity**: Optimize for understanding, not peak performance

### 1.2 Target Model Specifications

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 64 |
| Attention heads | 4 |
| Head dimension | 16 |
| FFN dimension | 256 (4x hidden) |
| Layers | 4 |
| Max sequence length | 16 |
| Vocabulary size | 256 |
| Quantization | INT8 per-tensor symmetric |

---

## 2. Top-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Tiny NPU Top                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              AXI4-Lite Control Interface                     │   │
│  │   • Configuration registers                                  │   │
│  │   • Start/reset/status                                       │   │
│  └───────────────────────┬─────────────────────────────────────┘   │
│                          │                                          │
│  ┌───────────────────────▼─────────────────────────────────────┐   │
│  │              Microcode Controller                            │   │
│  │   • Instruction fetch from SRAM                              │   │
│  │   • Decode and dispatch to engines                           │   │
│  │   • Scoreboard tracking                                      │   │
│  │   • Barrier synchronization                                  │   │
│  └──────┬──────┬──────┬──────┬──────┬──────┬──────────────────┘   │
│         │      │      │      │      │      │                        │
│  ┌──────▼──┐ ┌─▼────┐ ┌▼─────┐ ┌▼──────┐ ┌▼─────┐ ┌▼──────────┐   │
│  │  DMA    │ │ GEMM │ │SOFT  │ │ LAYER │ │ GELU │ │   VEC     │   │
│  │ Engine  │ │Engine│ │MAX   │ │ NORM  │ │Engine│ │  Engine   │   │
│  │ (DDR↔   │ │16×16 │ │Engine│ │Engine │ │      │ │(Add/Mul/  │   │
│  │  SRAM)  │ │syst. │ │      │ │      │ │      │ │ Copy/     │   │
│  │         │ │array)│ │      │ │      │ │      │ │ Clamp)    │   │
│  └────┬────┘ └──┬───┘ └──┬───┘ └───┬───┘ └──┬───┘ └─────┬─────┘   │
│       │         │        │         │        │           │          │
│  ┌────┴─────────┴────────┴─────────┴────────┴───────────┴─────┐   │
│  │                      Crossbar Switch                        │   │
│  │              (Arbitrated access to SRAM)                    │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│  ┌───────────────────────────▼─────────────────────────────────┐   │
│  │                      On-Chip SRAM                           │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │   │
│  │  │    SRAM0 (64KB)     │  │   SRAM1 (8KB)       │          │   │
│  │  │  • Weights          │  │  • LayerNorm beta   │          │   │
│  │  │  • Activations      │  │  • Residuals        │          │   │
│  │  │  • Microcode        │  │  • Scratch          │          │   │
│  │  │  • KV-cache         │  │                     │          │   │
│  │  └─────────────────────┘  └─────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              AXI4 DDR Interface                              │   │
│  │   • Burst reads/writes for weight loading                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Instruction Set Architecture (ISA)

### 3.1 Instruction Format

Fixed 128-bit instruction width for simple decode logic:

```
 127 112 111 96 95   80 79   64
 +-------+-------+--------+--------+
 |  imm  |   K   |   N    |   M    |
 |16 bits|16 bits|16 bits |16 bits |
 +-------+-------+--------+--------+

 63   48 47   32 31   16 15   8 7  0
 +--------+--------+--------+-------+------+
 |  src1  |  src0  |  dst   | flags |opcode|
 |16 bits |16 bits |16 bits | 8 bits|8 bits|
 +--------+--------+--------+-------+------+
```

### 3.2 Opcodes

| Code | Mnemonic | Engine | Description | Fields Used |
|------|----------|--------|-------------|-------------|
| 0x00 | NOP | - | No operation | - |
| 0x01 | DMA_LOAD | DMA | DDR → SRAM | dst=SRAM, src0=DDR, M=bytes |
| 0x02 | DMA_STORE | DMA | SRAM → DDR | src0=SRAM, M=bytes |
| 0x03 | GEMM | GEMM | Matrix multiply | dst, src0, src1, M, N, K, imm=scale/shift |
| 0x04 | SOFTMAX | Softmax | Row-wise softmax | dst, src0, M=rows, N=cols, flags=causal |
| 0x05 | LAYERNORM | LayerNorm | Layer normalization | dst, src0, src1=beta, M, N |
| 0x06 | GELU | GELU | GELU activation | dst, src0, M, N |
| 0x07 | VEC_ADD | Vector | Element-wise add | dst, src0, src1, M, N |
| 0x08 | VEC_MUL | Vector | Element-wise mul | dst, src0, src1, M, N |
| 0x09 | VEC_COPY | Vector | 2D strided copy | dst, src0, M, K=src_stride, imm=dst_stride |
| 0xFE | BARRIER | - | Wait all engines | - |
| 0xFF | END | - | End of program | - |

### 3.3 GEMM Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | TRANSPOSE_B | Transpose weight matrix |
| 1 | REQUANT | Apply requantization (imm = scale\|shift) |
| 2 | ACCUMULATE | Accumulate with existing output |

---

## 4. Memory Architecture

### 4.1 SRAM0 Memory Map (64KB)

```
Address     Size    Name            Description
────────────────────────────────────────────────────────────
0x0000      12KB    QKV_WEIGHTS     Query/Key/Value weights (3 × 4KB)
0x3000      4KB     OUT_PROJ        Output projection weights
0x4000      16KB    FFN_UP          FFN up-projection weights
0x8000      16KB    FFN_DOWN        FFN down-projection weights
0xC000      1KB     INPUT           Input activations [seq_len, hidden]
0xC400      1KB     LN1_OUT         Post-LayerNorm activations
0xC800      256B    Q_HEAD          Per-head query buffer (reused)
0xC900      256B    K_HEAD          Per-head key buffer (reused)
0xCA00      256B    V_HEAD          Per-head value buffer (reused)
0xCB00      256B    SCORES          Attention scores Q×K^T
0xCC00      256B    PROBS           Softmax output (reused)
0xCD00      256B    CONTEXT         Attention context (reused)
0xCE00      1KB     ATTENTION       Concatenated attention output
0xD200      1KB     PROJ_OUT        Output projection result
0xD600      1KB     RESIDUAL1       First residual (X + attention)
0xDA00      1KB     LN2_OUT         Post-FFN LayerNorm
0xDE00      4KB     FFN_INTER       FFN intermediate activations
0xEE00      1KB     FFN_OUT         FFN output
0xF200      1KB     OUTPUT          Block output
0xF600      2.5KB   UCODE           Microcode storage (~100 instr)
```

### 4.2 SRAM1 Memory Map (8KB)

```
Address     Size    Name            Description
────────────────────────────────────────────────────────────
0x0000      64B     LN1_BETA        LayerNorm 1 beta/gamma
0x0040      64B     LN2_BETA        LayerNorm 2 beta/gamma
0x0100      1KB     RESIDUAL        Residual connection buffer
0x0500      6KB     KV_CACHE        Key/value cache for generation
```

---

## 5. Engine Specifications

### 5.1 GEMM Engine (Systolic Array)

**Architecture**: 16×16 weight-stationary systolic array

```
         Activations flow DOWN (one row per cycle)
              │      │      │      │
              ▼      ▼      ▼      ▼
           ┌────┐ ┌────┐ ┌────┐ ┌────┐
    ──▶    │MAC │▶│MAC │▶│MAC │▶│MAC │──▶ Output
           │ 0,0│ │ 0,1│ │ 0,2│ │ 0,F│
           └────┘ └────┘ └────┘ └────┘
              │      │      │      │
           ┌────┐ ┌────┐ ┌────┐ ┌────┐
    ──▶    │MAC │▶│MAC │▶│MAC │▶│MAC │──▶ Output
           │ 1,0│ │ 1,1│ │ 1,2│ │ 1,F│
           └────┘ └────┘ └────┘ └────┘
              │      │      │      │
              :      :      :      :
           ┌────┐ ┌────┐ ┌────┐ ┌────┐
    ──▶    │MAC │▶│MAC │▶│MAC │▶│MAC │──▶ Output
           │ F,0│ │ F,1│ │ F,2│ │ F,F│
           └────┘ └────┘ └────┘ └────┘
```

**Operation**:
1. Load weights into array (stationary)
2. Stream activations through (one per cycle per row)
3. Partial sums propagate right
4. Results accumulate in output registers

**Tiling**: For [M,K] × [K,N] where K > 16:
- Break into 16×16 tiles
- Accumulate partial sums across K dimension

**Requantization**:
```
output = clamp(round((accumulator * scale) >> shift), -128, 127)
```

### 5.2 Softmax Engine

Three-pass fixed-point algorithm:

**Pass 1: Find Max**
- Row-wise maximum for numerical stability

**Pass 2: Exp + Sum**
- Compute exp(x - max) via 256-entry LUT
- Accumulate sum

**Pass 3: Normalize**
- Multiply by reciprocal (via LUT)
- Output INT8 probabilities

**Causal Mask**: Optional flag to mask future positions (for autoregressive attention)

### 5.3 LayerNorm Engine

Two-pass algorithm:

**Pass 1: Compute Mean + Variance**
- Sum(x) → mean
- Sum((x - mean)²) → variance

**Pass 2: Normalize + Scale**
- x_norm = (x - mean) × rsqrt(var + ε)
- y = x_norm × gamma + beta

Uses inverse square root LUT for rsqrt.

### 5.4 GELU Engine

Approximate GELU via 256-entry LUT:
```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

Precomputed for INT8 input range [-128, 127].

### 5.5 Vector Engine

Element-wise operations:
- VEC_ADD: Saturating addition (for residuals)
- VEC_MUL: Fixed-point multiplication
- VEC_COPY: 2D strided copy (for attention head scatter/gather)
- VEC_CLAMP: Clamp to range

### 5.6 DMA Engine

AXI4 master for DDR transfers:
- Configurable burst length
- Address alignment handling
- Completion signaling

---

## 6. Control Flow

### 6.1 Microcode Controller

Three-stage pipeline:
1. **Fetch**: Read 128-bit instruction from SRAM
2. **Decode**: Extract opcode, addresses, dimensions
3. **Dispatch**: Send to target engine if available

**Scoreboard**: Track which engines are busy
- 6 slots (one per engine)
- Stall fetch if target engine busy
- Barrier instruction waits for all slots clear

### 6.2 Execution Example: Single Attention Head

```asm
; Load Q, K, V weights for this head via DMA (done before)
; Input activations in SRAM0 @ 0xC000

; LayerNorm
LAYERNORM dst=0xC400 src0=0xC000 src1=SRAM1:0x0000 M=16 N=64
BARRIER

; Q projection: LN_out[16,64] × Wq[64,16] → Q[16,16]
GEMM dst=0xC800 src0=0xC400 src1=SRAM0:Wq M=16 N=16 K=64 imm=0x0701
BARRIER

; K projection
GEMM dst=0xC900 src0=0xC400 src1=SRAM0:Wk M=16 N=16 K=64 imm=0x0701
BARRIER

; V projection
GEMM dst=0xCA00 src0=0xC400 src1=SRAM0:Wv M=16 N=16 K=64 imm=0x0701
BARRIER

; Attention scores: Q[16,16] × K^T[16,16] → S[16,16]
GEMM dst=0xCB00 src0=0xC800 src1=0xC900 M=16 N=16 K=16 flags=TRANSPOSE_B imm=0x0701
BARRIER

; Softmax with causal mask
SOFTMAX dst=0xCC00 src0=0xCB00 M=16 N=16 flags=CAUSAL
BARRIER

; Context: P[16,16] × V[16,16] → Context[16,16]
GEMM dst=0xCD00 src0=0xCC00 src1=0xCA00 M=16 N=16 K=16 imm=0x0701
BARRIER

; Scatter context into concatenated attention buffer
VEC_COPY dst=0xCE00 src0=0xCD00 M=16 K=16 imm=64  ; stride=64 for concat
```

---

## 7. Test Strategy

### 7.1 Testing Philosophy

Every module must be testable standalone with:
1. **Unit tests**: Individual engine correctness
2. **Integration tests**: Multi-engine pipelines
3. **Golden reference**: Python/C++ reference for bit-exact comparison
4. **Regression tests**: CI runs on every commit

### 7.2 Test Pyramid

```
                    ┌─────────────────┐
                    │  End-to-End     │  Run GPT-2 inference
                    │  Demo Tests     │  Compare tokens vs PyTorch
                    │  (1 test)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Integration    │  Full transformer block
                    │  Tests          │  Attention + FFN + residuals
                    │  (5 tests)      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│  Engine Tests   │ │  Engine Tests   │ │  Engine Tests   │
│  (GEMM)         │ │  (Softmax)      │ │  (LayerNorm)    │
│  Matrix mult    │ │  Exp/sum/norm   │ │  Mean/var/norm  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│  Unit Tests     │ │  Unit Tests     │ │  Unit Tests     │
│  (MAC array)    │ │  (LUTs)         │ │  (rsqrt)        │
│  Single PE      │ │  Table lookup   │ │  Fixed-point    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 7.3 Test Coverage Requirements

| Module | Test Type | Coverage Criteria |
|--------|-----------|-------------------|
| MAC unit | Unit | All input combinations, overflow, saturation |
| Systolic array | Unit | 16×16 matmul, tiling, boundary conditions |
| GEMM engine | Engine | Various matrix shapes, requantization, tiling |
| Softmax engine | Engine | Numerical stability, causal mask, INT8 range |
| LayerNorm | Engine | Mean/variance accuracy, rsqrt LUT accuracy |
| Vector engine | Engine | All operations, strided copy, saturation |
| DMA engine | Engine | Burst transfers, alignment, completion |
| Controller | Integration | Instruction fetch, scoreboard, barrier |
| Attention | Integration | Full head compute, bit-exact vs reference |
| Transformer block | Integration | Layer-by-layer verification |
| Full NPU | E2E | Token generation matches PyTorch |

### 7.4 Golden Reference

Python reference implementation:
```python
def gemm_golden(A, B, scale=1.0, shift=0):
    """Reference INT8 GEMM with INT32 accumulation."""
    A_i32 = A.astype(np.int32)
    B_i32 = B.astype(np.int32)
    C = A_i32 @ B_i32  # INT32 accumulation
    C = (C * scale) >> shift  # Requantization
    return np.clip(C, -128, 127).astype(np.int8)

def softmax_golden(x, causal=False):
    """Reference fixed-point softmax."""
    x = x.astype(np.float32)  # Use FP32 for reference
    if causal:
        mask = np.triu(np.ones_like(x), k=1) * -1e9
        x = x + mask
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True))
```

### 7.5 CI/CD Testing

GitHub Actions workflow:
1. **Build**: Compile all simulation targets
2. **Unit Tests**: Run individual engine tests
3. **Integration Tests**: Run multi-engine tests
4. **Golden Comparison**: Verify bit-exact vs Python reference
5. **Lint**: Verible linting for style

---

## 8. Verification Plan

### 8.1 Verification Levels

| Level | Description | Method |
|-------|-------------|--------|
| L0 | Module unit tests | Verilator + C++ testbench |
| L1 | Engine integration | Verilator + Python golden |
| L2 | Full chip simulation | Verilator + GPT-2 weights |
| L3 | FPGA prototyping | AWS F1 or local FPGA |

### 8.2 Success Criteria

1. **Bit-exact**: NPU output matches golden reference (max error = 0)
2. **Performance**: Reasonable cycle counts for each operation
3. **Resource**: Fits in target FPGA (if prototyping)
4. **Power**: Not a primary concern for this educational design

---

## 9. Development Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] MAC unit + basic testbench
- [ ] Systolic array (16×16) + testbench
- [ ] GEMM engine + testbench with golden comparison

### Phase 2: Nonlinear Engines (Week 3)
- [ ] Softmax engine + testbench
- [ ] LayerNorm engine + testbench
- [ ] GELU engine + testbench
- [ ] Vector engine + testbench

### Phase 3: Memory & Control (Week 4)
- [ ] SRAM models
- [ ] DMA engine + testbench
- [ ] Microcode controller + testbench

### Phase 4: Integration (Week 5-6)
- [ ] Full NPU top-level
- [ ] Single attention head integration test
- [ ] Full transformer block test

### Phase 5: Application (Week 7-8)
- [ ] Weight export from HuggingFace
- [ ] Quantization pipeline
- [ ] End-to-end GPT-2 inference demo

---

## 10. References

- Systolic Arrays: Kung & Leiserson, 1979
- Transformer Architecture: "Attention Is All You Need", Vaswani et al., 2017
- INT8 Quantization: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", Jacob et al., 2018
- GPT-2: "Language Models are Unsupervised Multitask Learners", Radford et al., 2019

---

*Document Version: 1.0*
*Last Updated: 2026-02-12*
