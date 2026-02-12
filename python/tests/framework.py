"""
Test framework for NPU verification.
Provides utilities for generating test vectors, running simulations, and comparing results.
"""

import numpy as np
import subprocess
import os
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import json


class TestCase:
    """Base class for NPU test cases."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.passed = False
        self.error_msg = ""
        self.cycles = 0
    
    def generate_inputs(self, output_dir: Path) -> None:
        """Generate input test vectors. Override in subclass."""
        raise NotImplementedError
    
    def run_golden(self) -> np.ndarray:
        """Run golden reference. Override in subclass."""
        raise NotImplementedError
    
    def run_simulation(self, sim_path: Path, output_dir: Path) -> np.ndarray:
        """Run hardware simulation. Override in subclass."""
        raise NotImplementedError
    
    def verify(self, golden: np.ndarray, hw: np.ndarray, tolerance: int = 0) -> bool:
        """Compare golden vs hardware output."""
        if golden.shape != hw.shape:
            self.error_msg = f"Shape mismatch: {golden.shape} vs {hw.shape}"
            return False
        
        diff = np.abs(golden.astype(np.int16) - hw.astype(np.int16))
        max_diff = np.max(diff)
        mismatches = np.sum(diff > tolerance)
        
        if max_diff <= tolerance:
            self.passed = True
            return True
        else:
            self.error_msg = f"{mismatches}/{golden.size} mismatches, max_diff={max_diff}"
            return False
    
    def run(self, sim_path: Path, output_dir: Path, tolerance: int = 0) -> bool:
        """Run complete test: generate inputs, golden, sim, verify."""
        print(f"\n{'='*60}")
        print(f"Test: {self.name}")
        print(f"{'='*60}")
        
        try:
            # Generate inputs
            print("Generating inputs...")
            self.generate_inputs(output_dir)
            
            # Run golden reference
            print("Running golden reference...")
            golden = self.run_golden()
            
            # Run hardware simulation
            print("Running hardware simulation...")
            hw = self.run_simulation(sim_path, output_dir)
            
            # Verify
            print("Verifying results...")
            result = self.verify(golden, hw, tolerance)
            
            if result:
                print(f"✓ PASSED: {self.name}")
            else:
                print(f"✗ FAILED: {self.name}")
                print(f"  Error: {self.error_msg}")
            
            return result
            
        except Exception as e:
            self.error_msg = str(e)
            print(f"✗ FAILED: {self.name}")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


class GEMMTestCase(TestCase):
    """Test case for GEMM engine."""
    
    def __init__(self, M: int, K: int, N: int, name: Optional[str] = None):
        self.M = M
        self.K = K
        self.N = N
        name = name or f"GEMM_{M}x{K}x{N}"
        super().__init__(name, f"GEMM test: [{M},{K}] @ [{K},{N}]")
    
    def generate_inputs(self, output_dir: Path) -> None:
        from python.golden.reference import quantize_tensor
        
        # Generate random FP32 weights and activations
        A_f = np.random.randn(self.M, self.K).astype(np.float32) * 0.5
        B_f = np.random.randn(self.K, self.N).astype(np.float32) * 0.5
        
        # Quantize to INT8
        self.A, _ = quantize_tensor(A_f, scale=None)
        self.B, _ = quantize_tensor(B_f, scale=None)
        
        # Save to binary files
        output_dir.mkdir(parents=True, exist_ok=True)
        self.A.tofile(output_dir / "A.bin")
        self.B.tofile(output_dir / "B.bin")
        
        # Save metadata
        with open(output_dir / "test_config.json", "w") as f:
            json.dump({
                "M": self.M,
                "K": self.K,
                "N": self.N,
                "scale": 1,
                "shift": self.compute_shift()
            }, f)
    
    def compute_shift(self) -> int:
        """Compute appropriate shift for requantization."""
        return int(np.ceil(np.log2(self.K))) + 1
    
    def run_golden(self) -> np.ndarray:
        from python.golden.reference import gemm_golden
        shift = self.compute_shift()
        return gemm_golden(self.A, self.B, scale=1, shift=shift)
    
    def run_simulation(self, sim_path: Path, output_dir: Path) -> np.ndarray:
        # Run Verilator simulation
        result = subprocess.run(
            [str(sim_path / "gemm_test"), str(output_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")
        
        # Read output
        C = np.fromfile(output_dir / "C_hw.bin", dtype=np.int8)
        return C.reshape(self.M, self.N)


class TestSuite:
    """Collection of test cases."""
    
    def __init__(self, name: str):
        self.name = name
        self.tests: List[TestCase] = []
    
    def add_test(self, test: TestCase) -> None:
        self.tests.append(test)
    
    def run_all(self, sim_path: Path, output_dir: Path, tolerance: int = 0) -> Tuple[int, int]:
        """Run all tests. Returns (passed, total)."""
        print(f"\n{'#'*70}")
        print(f"# Test Suite: {self.name}")
        print(f"{'#'*70}")
        
        passed = 0
        failed_tests = []
        
        for test in self.tests:
            if test.run(sim_path, output_dir / test.name, tolerance):
                passed += 1
            else:
                failed_tests.append(test.name)
        
        # Summary
        print(f"\n{'#'*70}")
        print(f"# Results: {passed}/{len(self.tests)} passed")
        print(f"{'#'*70}")
        
        if failed_tests:
            print(f"\nFailed tests:")
            for name in failed_tests:
                print(f"  - {name}")
        
        return passed, len(self.tests)


def generate_gemm_tests() -> TestSuite:
    """Generate comprehensive GEMM test suite."""
    suite = TestSuite("GEMM Engine Tests")
    
    # Basic tests
    suite.add_test(GEMMTestCase(16, 16, 16, "GEMM_16x16x16"))  # Fits exactly in array
    suite.add_test(GEMMTestCase(4, 8, 4, "GEMM_small"))  # Small test
    suite.add_test(GEMMTestCase(16, 64, 16, "GEMM_K_tiling"))  # Requires K tiling
    suite.add_test(GEMMTestCase(16, 16, 64, "GEMM_N_tiling"))  # Requires N tiling
    suite.add_test(GEMMTestCase(64, 64, 64, "GEMM_large"))  # Large test
    
    # Edge cases
    suite.add_test(GEMMTestCase(1, 16, 1, "GEMM_vector"))  # Vector input
    suite.add_test(GEMMTestCase(16, 1, 16, "GEMM_K=1"))  # K=1 (inner product)
    
    return suite


def write_array_to_file(arr: np.ndarray, filepath: Path, hex_format: bool = False) -> None:
    """Write numpy array to binary file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if hex_format:
        # Write as hex for Verilog $readmemh
        with open(filepath, "w") as f:
            for val in arr.flatten():
                # Handle negative numbers (two's complement)
                if val < 0:
                    val = val + 256
                f.write(f"{val:02x}\n")
    else:
        arr.tofile(filepath)


def read_array_from_file(filepath: Path, shape: Tuple[int, ...], dtype=np.int8) -> np.ndarray:
    """Read numpy array from binary file."""
    arr = np.fromfile(filepath, dtype=dtype)
    return arr.reshape(shape)


if __name__ == "__main__":
    print("NPU Test Framework")
    print("Run with: python -m pytest tests/")
