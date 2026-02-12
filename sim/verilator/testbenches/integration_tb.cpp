// Integration Testbench
// Verifies full NPU operation with memory and controller

#include <iostream>
#include <verilated.h>
#include "Vnpu_top.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vnpu_top* top = new Vnpu_top;

    // Clock and Reset
    top->clk = 0;
    top->rst_n = 0;

    // Run reset
    for (int i = 0; i < 10; i++) {
        top->clk = !top->clk;
        top->eval();
    }
    top->rst_n = 1;

    std::cout << "=== NPU Integration Test ===" << std::endl;
    
    // Test AXI Lite write to CTRL register
    top->s_axi_awvalid = 1;
    top->s_axi_awaddr = 0x00; // CTRL
    top->s_axi_wvalid = 1;
    top->s_axi_wdata = 0x01; // Start
    top->s_axi_wstrb = 0xF;
    top->s_axi_bready = 1;
    
    // Cycle clock
    top->clk = !top->clk; top->eval();
    top->clk = !top->clk; top->eval();
    
    top->s_axi_awvalid = 0;
    top->s_axi_wvalid = 0;

    // Wait for done or timeout
    int cycles = 0;
    while (!top->done && cycles < 1000) {
        top->clk = !top->clk; top->eval();
        top->clk = !top->clk; top->eval();
        cycles++;
    }

    if (top->done) {
        std::cout << "PASS: NPU finished execution." << std::endl;
    } else {
        std::cout << "FAIL: Timeout waiting for NPU done." << std::endl;
    }

    top->final();
    delete top;
    return 0;
}
