// 16x16 Systolic Array for Matrix Multiplication
// Weight-stationary design: weights pre-loaded, activations stream through

`timescale 1ns/1ps

module systolic_array #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ARRAY_SIZE = 16
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control
    input  logic                      load_weights,   // Load weight matrix
    input  logic                      start_compute,  // Start computation
    input  logic                      clear_acc,      // Clear accumulators
    
    // Weight loading interface (row by row)
    input  logic [DATA_WIDTH-1:0]     weight_in [0:ARRAY_SIZE-1],  // One row per cycle
    input  logic [$clog2(ARRAY_SIZE)-1:0] weight_row,  // Which row to load
    
    // Activation input (column by column, skewed)
    input  logic [DATA_WIDTH-1:0]     activation_in [0:ARRAY_SIZE-1],
    input  logic                      activation_valid,
    
    // Partial sum input (from previous tile, for accumulation)
    input  logic [ACC_WIDTH-1:0]      partial_sum_in [0:ARRAY_SIZE-1],
    
    // Output (row by row)
    output logic [ACC_WIDTH-1:0]      result_out [0:ARRAY_SIZE-1],
    output logic                      result_valid,
    
    // Status
    output logic                      busy
);

    // Internal 2D arrays for connecting MAC units
    // Activation flows down (vertical)
    logic [DATA_WIDTH-1:0] activation_wire [0:ARRAY_SIZE][0:ARRAY_SIZE];
    
    // Partial sums flow right (horizontal)
    logic [ACC_WIDTH-1:0] partial_sum_wire [0:ARRAY_SIZE][0:ARRAY_SIZE];
    
    // Weight loading distribution
    logic [DATA_WIDTH-1:0] weight_load [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic load_weight_reg [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    // State machine
    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE,
        OUTPUT
    } state_t;
    
    state_t state, next_state;
    logic [$clog2(ARRAY_SIZE*2+10)-1:0] cycle_count;
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cycle_count <= '0;
        end else begin
            state <= next_state;
            
            if (state == COMPUTE) begin
                cycle_count <= cycle_count + 1;
            end else begin
                cycle_count <= '0;
            end
        end
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (load_weights) next_state = LOAD_WEIGHTS;
                else if (start_compute) next_state = COMPUTE;
            end
            
            LOAD_WEIGHTS: begin
                if (!load_weights) next_state = IDLE;
            end
            
            COMPUTE: begin
                // Computation takes ARRAY_SIZE + ARRAY_SIZE - 1 cycles
                // Plus some padding for pipeline flush
                if (cycle_count >= ARRAY_SIZE * 2 + 2) begin
                    next_state = OUTPUT;
                end
            end
            
            OUTPUT: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    assign busy = (state != IDLE);
    
    // Weight loading control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ARRAY_SIZE; i++) begin
                for (int j = 0; j < ARRAY_SIZE; j++) begin
                    weight_load[i][j] <= '0;
                    load_weight_reg[i][j] <= 1'b0;
                end
            end
        end else if (load_weights) begin
            for (int col = 0; col < ARRAY_SIZE; col++) begin
                weight_load[weight_row][col] <= weight_in[col];
                load_weight_reg[weight_row][col] <= 1'b1;
            end
        end else begin
            // Clear load_weight_reg after one cycle
            for (int i = 0; i < ARRAY_SIZE; i++) begin
                for (int j = 0; j < ARRAY_SIZE; j++) begin
                    load_weight_reg[i][j] <= 1'b0;
                end
            end
        end
    end
    
    // Input activations (top edge of array)
    generate
        for (genvar col = 0; col < ARRAY_SIZE; col++) begin : gen_activation_input
            assign activation_wire[0][col] = activation_in[col];
        end
    endgenerate
    
    // Input partial sums (left edge of array)
    generate
        for (genvar row = 0; row < ARRAY_SIZE; row++) begin : gen_partial_input
            assign partial_sum_wire[row][0] = partial_sum_in[row];
        end
    endgenerate
    
    // Systolic array of MAC units
    generate
        for (genvar row = 0; row < ARRAY_SIZE; row++) begin : gen_row
            for (genvar col = 0; col < ARRAY_SIZE; col++) begin : gen_col
                
                mac_unit #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) mac (
                    .clk(clk),
                    .rst_n(rst_n),
                    .en(activation_valid && (state == COMPUTE)),
                    .clr(clear_acc),
                    .load_weight(load_weight_reg[row][col]),
                    .activation_in(activation_wire[row][col]),
                    .weight_in(weight_load[row][col]),
                    .partial_sum_in(partial_sum_wire[row][col]),
                    .activation_out(activation_wire[row+1][col]),
                    .weight_out(),  // Not used in weight-stationary
                    .partial_sum_out(partial_sum_wire[row][col+1])
                );
                
            end
        end
    endgenerate
    
    // Output assignment (right edge of array)
    // Result appears on the right side after ARRAY_SIZE cycles
    generate
        for (genvar row = 0; row < ARRAY_SIZE; row++) begin : gen_output
            assign result_out[row] = partial_sum_wire[row][ARRAY_SIZE];
        end
    endgenerate
    
    // Result valid: assert when results are emerging
    // In a weight-stationary systolic array, results emerge over ARRAY_SIZE cycles
    // starting at cycle ARRAY_SIZE (after the first column propagates through)
    assign result_valid = ((state == COMPUTE) && 
                           (cycle_count >= ARRAY_SIZE) && 
                           (cycle_count < ARRAY_SIZE * 2));

endmodule
