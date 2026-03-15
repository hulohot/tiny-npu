#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <verilated.h>

#include "Vlayernorm_engine.h"

static void tick(Vlayernorm_engine* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vlayernorm_engine;

    dut->clk = 0;
    dut->rst_n = 0;
    dut->start = 0;
    dut->hidden_dim = 4;
    dut->data_valid = 0;
    dut->param_valid = 0;
    dut->data_in = 0;
    dut->gamma_in = 0;
    dut->beta_in = 0;

    tick(dut);
    tick(dut);
    dut->rst_n = 1;
    tick(dut);

    // Load gamma=127 (~1.0 in Q7), beta=0 while in IDLE.
    for (int i = 0; i < 4; ++i) {
        dut->param_valid = 1;
        dut->gamma_in = 127;
        dut->beta_in = 0;
        tick(dut);
    }
    dut->param_valid = 0;

    // Start layernorm and feed 4 values: [-2, -1, 1, 2].
    dut->start = 1;
    tick(dut);
    dut->start = 0;

    // Expected outputs (analytically computed, Q16.16 arithmetic):
    //   inputs  = [-2, -1, 1, 2], mean = 0, variance = 2
    //
    //   Note: in COMPUTE_STATS, `inv_sqrt_var <= rsqrt_lut[variance[9:0]]` reads the
    //   OLD (reset=0) value of `variance` due to non-blocking assignment semantics,
    //   so inv_sqrt_var = rsqrt_lut[0] = floor(1/sqrt(1) * 65536) = 65536  (Q16.16 ≡ 1.0)
    //
    //   x_norm[i]  = (x[i] * 65536) >>> 16  →  [-2, -1, 1, 2]
    //   scaled[i]  = (x_norm[i] * 127) >>> 7
    //     -2 → -2,  -1 → -1,  1 → 0 (127/128 truncated),  2 → 1
    //   shifted[i] = scaled[i] + beta=0  →  [-2, -1, 0, 1]
    const int8_t input_vals[4]    = {-2, -1, 1, 2};
    const int8_t expected_vals[4] = {-2, -1, 0, 1};

    for (int i = 0; i < 4; ++i) {
        dut->data_valid = 1;
        dut->data_in = static_cast<uint8_t>(input_vals[i]);
        tick(dut);
    }
    dut->data_valid = 0;

    std::vector<int8_t> outputs;
    bool saw_done = false;
    for (int i = 0; i < 128; ++i) {
        tick(dut);
        if (dut->done) saw_done = true;
        if (dut->out_valid) outputs.push_back(static_cast<int8_t>(dut->data_out));
    }

    assert(saw_done && "layernorm_engine never reached done");
    assert(outputs.size() == 4 && "expected all 4 hidden_dim outputs");
    for (int i = 0; i < 4; ++i) {
        assert(outputs[i] == expected_vals[i] && "layernorm output mismatch");
    }

    std::cout << "layernorm_engine_tb: PASS" << std::endl;

    dut->final();
    delete dut;
    return 0;
}
