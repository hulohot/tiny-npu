#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <verilated.h>

#include "Vgelu_engine.h"

static void tick(Vgelu_engine* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vgelu_engine;

    dut->clk = 0;
    dut->rst_n = 0;
    dut->start = 0;
    dut->num_elements = 4;
    dut->data_valid = 0;
    dut->data_in = 0;

    tick(dut);
    tick(dut);
    dut->rst_n = 1;
    tick(dut);

    dut->start = 1;
    tick(dut);
    dut->start = 0;

    // Input: [-2, -1, 0, 2] (as unsigned bytes: 254, 255, 0, 2)
    // Expected GELU outputs (tanh approximation, truncated to INT8 via $rtoi):
    //   GELU(-2) ≈ -0.045 → 0
    //   GELU(-1) ≈ -0.159 → 0
    //   GELU( 0) =  0.000 → 0
    //   GELU( 2) ≈  1.955 → 1
    const int8_t in_vals[4]       = {-2, -1, 0, 2};
    const int8_t expected_vals[4] = { 0,  0, 0, 1};
    std::vector<int8_t> outs;

    // Feed all 4 inputs; pipeline produces outputs starting on the second input cycle.
    for (int i = 0; i < 4; ++i) {
        dut->data_valid = 1;
        dut->data_in = static_cast<uint8_t>(in_vals[i]);
        tick(dut);
        if (dut->out_valid) outs.push_back(static_cast<int8_t>(dut->data_out));
    }
    dut->data_valid = 0;

    // Drain the last element from the FIFO.
    for (int i = 0; i < 16; ++i) {
        tick(dut);
        if (dut->out_valid) outs.push_back(static_cast<int8_t>(dut->data_out));
    }

    assert(outs.size() == 4 && "expected exactly 4 GELU outputs");
    for (int i = 0; i < 4; ++i) {
        assert(outs[i] == expected_vals[i] && "GELU output mismatch");
    }

    std::cout << "gelu_engine_tb: PASS" << std::endl;

    dut->final();
    delete dut;
    return 0;
}
