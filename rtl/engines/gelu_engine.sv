// GELU Activation Engine
// Lookup-table based GELU approximation
// GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF
// Approximated as: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

`timescale 1ns/1ps

module gelu_engine #(
    parameter DATA_WIDTH = 8,
    parameter MAX_ELEMENTS = 4096  // Max elements to process (e.g., 16*256 for FFN)
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control
    input  logic                      start,
    output logic                      busy,
    output logic                      done,
    
    // Configuration
    input  logic [$clog2(MAX_ELEMENTS)-1:0] num_elements,
    
    // Data input (streaming)
    input  logic [DATA_WIDTH-1:0]     data_in,
    input  logic                      data_valid,
    
    // Data output (streaming)
    output logic [DATA_WIDTH-1:0]     data_out,
    output logic                      out_valid
);

    // GELU lookup table
    // Maps INT8 input to INT8 output
    // Precomputed using the tanh approximation
    logic [DATA_WIDTH-1:0] gelu_lut [0:255];
    
    // Initialize LUT
    initial begin
        for (int i = 0; i < 256; i++) begin
            // Convert to signed value
            int signed_val;
            real x, gelu_val;
            
            if (i < 128) begin
                signed_val = i;  // 0 to 127
            end else begin
                signed_val = i - 256;  // -128 to -1
            end
            
            x = signed_val;
            
            // GELU approximation using tanh
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            real sqrt_2_over_pi = 0.7978845608;
            real coeff = 0.044715;
            real inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            real tanh_val = (exp(inner) - exp(-inner)) / (exp(inner) + exp(-inner));
            gelu_val = 0.5 * x * (1.0 + tanh_val);
            
            // Clamp and convert back to INT8
            if (gelu_val > 127.0) begin
                gelu_lut[i] = 8'd127;
            end else if (gelu_val < -128.0) begin
                gelu_lut[i] = 8'h80;  // -128
            end else begin
                gelu_lut[i] = $rtoi(gelu_val) & 8'hFF;
            end
        end
    end
    
    // State machine
    typedef enum logic [1:0] {
        IDLE,
        PROCESSING,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Counters
    logic [$clog2(MAX_ELEMENTS)-1:0] in_count;
    logic [$clog2(MAX_ELEMENTS)-1:0] out_count;
    
    // Input buffer (small FIFO for pipeline)
    logic [DATA_WIDTH-1:0] input_buffer [0:3];
    logic [1:0] wr_ptr, rd_ptr;
    logic [2:0] fifo_count;
    
    // Output register
    logic [DATA_WIDTH-1:0] result_reg;
    logic result_valid_reg;
    
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            in_count <= '0;
            out_count <= '0;
            wr_ptr <= '0;
            rd_ptr <= '0;
            fifo_count <= '0;
            result_valid_reg <= 1'b0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    in_count <= '0;
                    out_count <= '0;
                    wr_ptr <= '0;
                    rd_ptr <= '0;
                    fifo_count <= '0;
                    result_valid_reg <= 1'b0;
                end
                
                PROCESSING: begin
                    // Input handling
                    if (data_valid && in_count < num_elements) begin
                        input_buffer[wr_ptr] <= data_in;
                        wr_ptr <= wr_ptr + 1;
                        fifo_count <= fifo_count + 1;
                        in_count <= in_count + 1;
                    end
                    
                    // Output handling (LUT lookup)
                    if (fifo_count > 0 && out_count < num_elements) begin
                        // Lookup in GELU table
                        result_reg <= gelu_lut[input_buffer[rd_ptr]];
                        result_valid_reg <= 1'b1;
                        rd_ptr <= rd_ptr + 1;
                        fifo_count <= fifo_count - 1;
                        out_count <= out_count + 1;
                    end else begin
                        result_valid_reg <= 1'b0;
                    end
                end
                
                DONE_STATE: begin
                    result_valid_reg <= 1'b0;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) next_state = PROCESSING;
            end
            
            PROCESSING: begin
                if (out_count >= num_elements && in_count >= num_elements) begin
                    next_state = DONE_STATE;
                end
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output assignments
    assign data_out = result_reg;
    assign out_valid = result_valid_reg;
    
    // Status
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule
