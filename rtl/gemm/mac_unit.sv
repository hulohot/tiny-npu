// Multiply-Accumulate Unit
// Basic building block for systolic array
// Performs: accumulator += activation * weight

`timescale 1ns/1ps

module mac_unit #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control
    input  logic                      en,           // Enable accumulation
    input  logic                      clr,          // Clear accumulator
    input  logic                      load_weight,  // Load new weight
    
    // Data inputs
    input  logic [DATA_WIDTH-1:0]     activation_in,   // From top neighbor
    input  logic [DATA_WIDTH-1:0]     weight_in,       // From load or left neighbor
    input  logic [ACC_WIDTH-1:0]      partial_sum_in,  // From left neighbor
    
    // Data outputs
    output logic [DATA_WIDTH-1:0]     activation_out,  // To bottom neighbor
    output logic [DATA_WIDTH-1:0]     weight_out,      // To right neighbor
    output logic [ACC_WIDTH-1:0]      partial_sum_out  // To right neighbor
);

    // Internal weight register (weight-stationary)
    logic [DATA_WIDTH-1:0] weight_reg;
    
    // Accumulator
    logic [ACC_WIDTH-1:0] accumulator;
    
    // Signed multiplication result
    logic signed [2*DATA_WIDTH-1:0] mult_result;
    logic signed [ACC_WIDTH-1:0] mult_extended;
    
    // Weight loading
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= '0;
        end else if (load_weight) begin
            weight_reg <= weight_in;
        end
    end
    
    // Signed multiplication
    // Both inputs treated as signed 8-bit values
    assign mult_result = signed'(activation_in) * signed'(weight_reg);
    
    // Sign-extend to accumulator width
    assign mult_extended = signed'(mult_result);
    
    // Accumulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else if (clr) begin
            accumulator <= '0;
        end else if (en) begin
            accumulator <= partial_sum_in + mult_extended;
        end else begin
            accumulator <= partial_sum_in;  // Pass through
        end
    end
    
    // Output assignments (registered for timing)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out <= '0;
            weight_out     <= '0;
            partial_sum_out<= '0;
        end else begin
            activation_out <= activation_in;
            weight_out     <= weight_reg;
            partial_sum_out<= accumulator;
        end
    end

endmodule
