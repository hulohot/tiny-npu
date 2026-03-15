// Softmax Engine
// Three-pass fixed-point softmax with optional causal mask
// Pass 1: Find max (numerical stability)
// Pass 2: Compute exp(x-max) and sum
// Pass 3: Normalize by dividing by sum

`timescale 1ns/1ps

module softmax_engine #(
    parameter DATA_WIDTH = 8,
    parameter EXP_WIDTH = 16,     // Fixed-point exp result
    parameter SUM_WIDTH = 32,     // Accumulator for sum
    parameter MAX_SEQ_LEN = 16
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control
    input  logic                      start,
    output logic                      busy,
    output logic                      done,
    
    // Configuration
    input  logic [$clog2(MAX_SEQ_LEN)-1:0] seq_len,  // Current sequence length
    input  logic                      causal_mask,   // Apply causal masking
    
    // Data input (row by row)
    input  logic [DATA_WIDTH-1:0]     data_in,
    input  logic                      data_valid,
    input  logic [$clog2(MAX_SEQ_LEN)-1:0] col_in,   // Column index
    input  logic [$clog2(MAX_SEQ_LEN)-1:0] row_in,   // Row index
    
    // Data output
    output logic [DATA_WIDTH-1:0]     data_out,
    output logic                      out_valid,
    output logic [$clog2(MAX_SEQ_LEN)-1:0] col_out,
    output logic [$clog2(MAX_SEQ_LEN)-1:0] row_out
);

    // States
    typedef enum logic [2:0] {
        IDLE,
        PASS1_MAX,      // Find max per row
        PASS2_EXP_SUM,  // Compute exp and sum
        PASS3_NORM,     // Normalize
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Row buffers
    logic [DATA_WIDTH-1:0] input_buffer [0:MAX_SEQ_LEN-1][0:MAX_SEQ_LEN-1];
    logic [DATA_WIDTH-1:0] max_per_row [0:MAX_SEQ_LEN-1];
    logic [SUM_WIDTH-1:0]  sum_per_row [0:MAX_SEQ_LEN-1];
    logic [DATA_WIDTH-1:0] result_buffer [0:MAX_SEQ_LEN-1][0:MAX_SEQ_LEN-1];
    
    // Current processing state
    logic [$clog2(MAX_SEQ_LEN)-1:0] current_row;
    logic [$clog2(MAX_SEQ_LEN)-1:0] current_col;

    // Output drain counters (used in DONE_STATE to walk result_buffer)
    logic [$clog2(MAX_SEQ_LEN)-1:0] drain_row;
    logic [$clog2(MAX_SEQ_LEN)-1:0] drain_col;

    // Combinational exp lookup for the current (row, col) — eliminates pipeline stale
    // in PASS2 accumulation and PASS3 normalization.
    logic [7:0]          exp_idx_comb;
    logic [EXP_WIDTH-1:0] exp_val_comb;
    assign exp_idx_comb = 8'($signed(input_buffer[current_row][current_col]) -
                              $signed(max_per_row[current_row]));
    assign exp_val_comb = exp_lut[exp_idx_comb];
    
    // Exp LUT: maps signed 8-bit difference to exp value
    // Precomputed: exp(x) for x in range [-8, 0] scaled to fit in EXP_WIDTH
    // For x < -8, exp(x) is effectively 0
    logic [EXP_WIDTH-1:0] exp_lut [0:255];
    
    // Initialize LUT (in real implementation, use $readmemh)
    // For now, approximate values
    initial begin
        for (int i = 0; i < 256; i++) begin
            // i is signed 8-bit value
            int signed_val = i < 128 ? i : i - 256;
            if (signed_val > 0) begin
                exp_lut[i] = 16'hFFFF;  // Clamp positive to max
            end else if (signed_val < -8) begin
                exp_lut[i] = 16'h0001;  // Very small, clamp to minimum
            end else begin
                // exp(signed_val) * 4096 (Q12.4 fixed point)
                real exp_val = $exp(signed_val);
                exp_lut[i] = EXP_WIDTH'($rtoi(exp_val * 4096));
            end
        end
    end
    

    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            current_row <= '0;
            current_col <= '0;
            drain_row <= '0;
            drain_col <= '0;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    current_row <= '0;
                    current_col <= '0;
                    drain_row <= '0;
                    drain_col <= '0;
                    if (start) begin
                        for (int i = 0; i < MAX_SEQ_LEN; i++) begin
                            max_per_row[i] <= 8'h80;  // -128
                            sum_per_row[i] <= '0;
                        end
                    end
                end

                PASS1_MAX: begin
                    if (current_row < seq_len) begin
                        if (current_col < seq_len) begin
                            if (!causal_mask || current_col <= current_row) begin
                                if ($signed(input_buffer[current_row][current_col]) >
                                    $signed(max_per_row[current_row])) begin
                                    max_per_row[current_row] <= input_buffer[current_row][current_col];
                                end
                            end
                            current_col <= current_col + 1;
                        end else begin
                            current_col <= '0;
                            current_row <= current_row + 1;
                        end
                    end else begin
                        // Pass complete: reset counters for next pass.
                        current_row <= '0;
                        current_col <= '0;
                    end
                end

                PASS2_EXP_SUM: begin
                    if (current_row < seq_len) begin
                        if (current_col < seq_len) begin
                            if (!causal_mask || current_col <= current_row) begin
                                // Use combinational exp lookup to avoid 1-cycle pipeline stale
                                // that previously caused cross-row sum corruption.
                                sum_per_row[current_row] <= sum_per_row[current_row] +
                                                            SUM_WIDTH'(exp_val_comb);
                            end
                            current_col <= current_col + 1;
                        end else begin
                            current_col <= '0;
                            current_row <= current_row + 1;
                        end
                    end else begin
                        // Pass complete: reset counters for next pass.
                        current_row <= '0;
                        current_col <= '0;
                    end
                end

                PASS3_NORM: begin
                    if (current_row < seq_len) begin
                        if (current_col < seq_len) begin
                            if (!causal_mask || current_col <= current_row) begin
                                // Use combinational exp lookup and inline normalization
                                // to avoid the 2-cycle pipeline stale (exp_result + norm_result).
                                if (sum_per_row[current_row] != 0) begin
                                    logic [EXP_WIDTH+16-1:0] norm_comb;
                                    norm_comb = (SUM_WIDTH'(exp_val_comb) * 32'h0000_007F) /
                                                sum_per_row[current_row];
                                    result_buffer[current_row][current_col] <=
                                        norm_comb > 127 ? 8'd127 : norm_comb[7:0];
                                end else begin
                                    result_buffer[current_row][current_col] <= 8'd0;
                                end
                            end else begin
                                result_buffer[current_row][current_col] <= 8'd0;  // Causal mask
                            end
                            current_col <= current_col + 1;
                        end else begin
                            current_col <= '0;
                            current_row <= current_row + 1;
                        end
                    end else begin
                        // Pass complete: reset counters for drain phase.
                        current_row <= '0;
                        current_col <= '0;
                    end
                end

                DONE_STATE: begin
                    // Walk drain_row/drain_col through result_buffer.
                    if (drain_row < seq_len) begin
                        if (drain_col < seq_len - 1) begin
                            drain_col <= drain_col + 1;
                        end else begin
                            drain_col <= '0;
                            drain_row <= drain_row + 1;
                        end
                    end
                end

                default: begin
                    current_row <= '0;
                    current_col <= '0;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) next_state = PASS1_MAX;
            end
            
            PASS1_MAX: begin
                if (current_row >= seq_len) next_state = PASS2_EXP_SUM;
            end
            
            PASS2_EXP_SUM: begin
                if (current_row >= seq_len) next_state = PASS3_NORM;
            end
            
            PASS3_NORM: begin
                if (current_row >= seq_len) next_state = DONE_STATE;
            end
            
            DONE_STATE: begin
                // Stay until all seq_len×seq_len results have been drained.
                if (drain_row >= seq_len) next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Status
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

    // Input capture
    always_ff @(posedge clk) begin
        if (data_valid) begin
            input_buffer[row_in][col_in] <= data_in;
        end
    end

    // Output generation: walk drain counters through result_buffer in DONE_STATE.
    assign out_valid = (state == DONE_STATE) && (drain_row < seq_len);
    assign row_out   = drain_row;
    assign col_out   = drain_col;
    assign data_out  = result_buffer[drain_row][drain_col];

endmodule
