#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <verilated.h>

#include "Vvec_engine.h"

static void tick(Vvec_engine* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vvec_engine;

    dut->clk = 0;
    dut->rst_n = 0;
    dut->start = 0;
    dut->operation = 0b001;  // VEC_ADD
    dut->num_elements = 3;
    dut->immediate = 0;
    dut->data_a_valid = 0;
    dut->data_b_valid = 0;
    dut->data_a_in = 0;
    dut->data_b_in = 0;

    tick(dut);
    tick(dut);
    dut->rst_n = 1;
    tick(dut);

    dut->start = 1;
    tick(dut);
    dut->start = 0;

    // VEC_ADD: [10,20,30] + [1,2,3] = [11,22,33] (all within INT8 range, no saturation).
    // With the single-stage pipeline fix, data_out is computed in the same cycle as
    // the input arrives, so outputs are captured correctly during the feed loop.
    const int8_t a_vals[3]        = {10, 20, 30};
    const int8_t b_vals[3]        = { 1,  2,  3};
    const int8_t expected_vals[3] = {11, 22, 33};
    std::vector<int8_t> outs;

    for (int i = 0; i < 3; ++i) {
        dut->data_a_valid = 1;
        dut->data_b_valid = 1;
        dut->data_a_in = static_cast<uint8_t>(a_vals[i]);
        dut->data_b_in = static_cast<uint8_t>(b_vals[i]);
        tick(dut);
        if (dut->out_valid) outs.push_back(static_cast<int8_t>(dut->data_out));
    }

    dut->data_a_valid = 0;
    dut->data_b_valid = 0;
    for (int i = 0; i < 8; ++i) {
        tick(dut);
        if (dut->out_valid) outs.push_back(static_cast<int8_t>(dut->data_out));
    }

    assert(outs.size() == 3 && "expected 3 output samples");
    for (int i = 0; i < 3; ++i) {
        assert(outs[i] == expected_vals[i] && "VEC_ADD output mismatch");
    }

    std::cout << "vec_engine_tb: PASS" << std::endl;

    dut->final();
    delete dut;
    return 0;
}
