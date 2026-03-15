#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <verilated.h>

#include "Vsoftmax_engine.h"

static void tick(Vsoftmax_engine* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vsoftmax_engine;

    dut->clk = 0;
    dut->rst_n = 0;
    dut->start = 0;
    dut->data_valid = 0;
    dut->seq_len = 2;
    dut->causal_mask = 0;
    dut->col_in = 0;
    dut->row_in = 0;
    dut->data_in = 0;

    tick(dut);
    tick(dut);
    dut->rst_n = 1;
    tick(dut);

    // Load a deterministic 2x2 attention matrix: [[1, 2], [3, 4]].
    // Rows have the same relative ordering, so within each row the second
    // element should have a strictly larger softmax output than the first.
    const int vals[2][2] = {{1, 2}, {3, 4}};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            dut->row_in = r;
            dut->col_in = c;
            dut->data_in = vals[r][c] & 0xFF;
            dut->data_valid = 1;
            tick(dut);
        }
    }
    dut->data_valid = 0;

    // Start processing.
    dut->start = 1;
    tick(dut);
    dut->start = 0;

    // Collect all outputs emitted during DONE_STATE drain.
    // For seq_len=2 we expect exactly 2×2=4 outputs.
    struct Output { int8_t val; uint8_t row; uint8_t col; };
    std::vector<Output> outputs;
    bool saw_done = false;
    for (int i = 0; i < 256; ++i) {
        tick(dut);
        if (dut->done) saw_done = true;
        if (dut->out_valid) {
            outputs.push_back({static_cast<int8_t>(dut->data_out),
                               static_cast<uint8_t>(dut->row_out),
                               static_cast<uint8_t>(dut->col_out)});
        }
    }

    assert(saw_done && "softmax_engine never reached done");
    assert(outputs.size() == 4 && "expected seq_len*seq_len=4 outputs");

    // Softmax probabilities are non-negative; all INT8 outputs must be >= 0.
    for (auto& o : outputs) {
        assert(o.val >= 0 && "softmax output must be non-negative");
    }

    // Within each row, col=1 had a larger input than col=0, so its output
    // must be strictly greater (both rows use the same relative ordering).
    int8_t row0_col0 = -1, row0_col1 = -1;
    int8_t row1_col0 = -1, row1_col1 = -1;
    for (auto& o : outputs) {
        if (o.row == 0 && o.col == 0) row0_col0 = o.val;
        if (o.row == 0 && o.col == 1) row0_col1 = o.val;
        if (o.row == 1 && o.col == 0) row1_col0 = o.val;
        if (o.row == 1 && o.col == 1) row1_col1 = o.val;
    }
    assert(row0_col1 > row0_col0 && "row 0: col1 should be larger (input 2 > 1)");
    assert(row1_col1 > row1_col0 && "row 1: col1 should be larger (input 4 > 3)");

    std::cout << "softmax_engine_tb: PASS" << std::endl;

    dut->final();
    delete dut;
    return 0;
}
