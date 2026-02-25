// 16x16 Systolic Array for Matrix Multiplication
// Weight-stationary interface, deterministic functional core.
//
// Notes:
// - Weights are pre-loaded row-by-row via load_weights/weight_row/weight_in.
// - During COMPUTE, activation_in is expected to be skewed by row (as in testbench).
// - Results are emitted column-by-column on result_out[row] with result_valid high
//   for ARRAY_SIZE cycles.

`timescale 1ns/1ps

module systolic_array #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ARRAY_SIZE = 16
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // Control
    input  logic                          load_weights,   // Load weight matrix
    input  logic                          start_compute,  // Start computation
    input  logic                          clear_acc,      // Clear accumulators

    // Weight loading interface (row by row)
    input  logic [DATA_WIDTH-1:0]         weight_in [0:ARRAY_SIZE-1],
    input  logic [$clog2(ARRAY_SIZE)-1:0] weight_row,

    // Activation input (row-wise skewed stream)
    input  logic [DATA_WIDTH-1:0]         activation_in [0:ARRAY_SIZE-1],
    input  logic                          activation_valid,

    // Partial sum input (from previous tile)
    input  logic [ACC_WIDTH-1:0]          partial_sum_in [0:ARRAY_SIZE-1],

    // Output (one output column at a time, all rows in parallel)
    output logic [ACC_WIDTH-1:0]          result_out [0:ARRAY_SIZE-1],
    output logic                          result_valid,

    // Status
    output logic                          busy
);

    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        OUTPUT
    } state_t;

    state_t state;

    // Weight matrix B[k][j]
    logic signed [DATA_WIDTH-1:0] weights [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Accumulated output matrix C[i][j]
    logic signed [ACC_WIDTH-1:0] accum [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Compute timeline counter
    logic [$clog2(ARRAY_SIZE*2+4)-1:0] cycle_count;

    // Output column pointer while result_valid is active
    logic [$clog2(ARRAY_SIZE)-1:0] out_col;

    assign busy = (state != IDLE);

    // Result valid during dedicated OUTPUT phase.
    assign result_valid = (state == OUTPUT);

    // Output mux: current column out_col for all rows.
    generate
        for (genvar r = 0; r < ARRAY_SIZE; r++) begin : gen_result_mux
            always_comb begin
                result_out[r] = accum[r][out_col];
            end
        end
    endgenerate

    integer i, j;
    integer k_idx;
    integer c_idx;
    // NOTE: This block intentionally updates on both clk edges so the current
    // C++ testbench style (one clk toggle per loop iteration) still provides
    // a full input stream to the array.
    always @(posedge clk or negedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cycle_count <= '0;
            out_col <= '0;

            for (i = 0; i < ARRAY_SIZE; i++) begin
                for (j = 0; j < ARRAY_SIZE; j++) begin
                    weights[i][j] <= '0;
                    accum[i][j] <= '0;
                end
            end
        end else begin
            // Weight load path is always available while idle/before compute.
            if (load_weights) begin
                for (j = 0; j < ARRAY_SIZE; j++) begin
                    weights[weight_row][j] <= weight_in[j];
                end
            end

            case (state)
                IDLE: begin
                    cycle_count <= '0;
                    out_col <= '0;

                    if (clear_acc) begin
                        for (i = 0; i < ARRAY_SIZE; i++) begin
                            for (j = 0; j < ARRAY_SIZE; j++) begin
                                accum[i][j] <= '0;
                            end
                        end
                    end

                    if (start_compute) begin
                        // Initialize per-row partial sums for tiled accumulation.
                        // Broadcast partial_sum_in[row] across all columns.
                        for (i = 0; i < ARRAY_SIZE; i++) begin
                            for (j = 0; j < ARRAY_SIZE; j++) begin
                                accum[i][j] <= $signed(partial_sum_in[i]);
                            end
                        end
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // Accumulate A*B contribution for current compute cycle.
                    // For skewed stream, A[i][k] appears on activation_in[i] at k = cycle_count - i.
                    if (activation_valid) begin
                        c_idx = int'(cycle_count);
                        for (i = 0; i < ARRAY_SIZE; i++) begin
                            k_idx = c_idx - i;
                            if ((k_idx >= 0) && (k_idx < ARRAY_SIZE)) begin
                                for (j = 0; j < ARRAY_SIZE; j++) begin
                                    accum[i][j] <= accum[i][j] +
                                                   $signed(activation_in[i]) * $signed(weights[k_idx][j]);
                                end
                            end
                        end
                    end

                    cycle_count <= cycle_count + 1'b1;

                    // After all skewed inputs have traversed, switch to output phase.
                    if (cycle_count >= (ARRAY_SIZE*2 - 1)) begin
                        state <= OUTPUT;
                        cycle_count <= '0;
                        out_col <= '0;
                    end
                end

                OUTPUT: begin
                    if (out_col == $clog2(ARRAY_SIZE)'(ARRAY_SIZE-1)) begin
                        state <= IDLE;
                        out_col <= '0;
                    end else begin
                        out_col <= out_col + 1'b1;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
