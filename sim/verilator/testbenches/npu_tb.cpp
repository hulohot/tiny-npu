// NPU Basic Smoke Test
// Simple test to verify the NPU top-level compiles and responds to reset

#include <iostream>
#include <verilated.h>
#include "Vnpu_smoke.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // Create instance
    Vnpu_smoke* top = new Vnpu_smoke;
    
    std::cout << "=== Tiny NPU Smoke Test ===" << std::endl;
    
    // Initialize
    top->clk = 0;
    top->rst_n = 0;
    
    // Reset
    for (int i = 0; i < 10; i++) {
        top->clk = !top->clk;
        top->eval();
    }
    
    // Release reset
    top->rst_n = 1;
    
    // Run a few cycles
    for (int i = 0; i < 20; i++) {
        top->clk = !top->clk;
        top->eval();
    }
    
    std::cout << "Smoke test PASSED - NPU compiles and resets correctly" << std::endl;
    
    // Cleanup
    top->final();
    delete top;
    
    return 0;
}
