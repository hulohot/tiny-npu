// Vector Engine
// Element-wise operations: ADD, MUL, COPY, CLAMP
// Used for residual connections, scaling, and data movement

`timescale 1ns/1ps

module vec_engine #(
    parameter DATA_WIDTH = 8,
    parameter MAX_ELEMENTS = 4096
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control
    input  logic                      start,
    output logic                      busy,
    output logic                      done,
    
    // Configuration
    input  logic [2:0]                operation,     // Operation code
    input  logic [$clog2(MAX_ELEMENTS)-1:0] num_elements,
    input  logic [DATA_WIDTH-1:0]     immediate,     // Immediate value for scalar ops
    
    // Data input A (streaming)
    input  logic [DATA_WIDTH-1:0]     data_a_in,
    input  logic                      data_a_valid,
    
    // Data input B (streaming, for binary ops)
    input  logic [DATA_WIDTH-1:0]     data_b_in,
    input  logic                      data_b_valid,
    
    // Data output (streaming)
    output logic [DATA_WIDTH-1:0]     data_out,
    output logic                      out_valid
);

    // Operation codes (match ISA)
    localparam VEC_NOP     = 3'b000;
    localparam VEC_ADD     = 3'b001;
    localparam VEC_MUL     = 3'b010;
    localparam VEC_SUB     = 3'b011;
    localparam VEC_SCALE   = 3'b100;  // Multiply by immediate
    localparam VEC_CLAMP   = 3'b101;  // Clamp to range
    localparam VEC_COPY    = 3'b110;  // Copy with stride
    localparam VEC_COPY2D  = 3'b111;  // 2D strided copy
    
    // State machine
    typedef enum logic [1:0] {
        IDLE,
        PROCESSING,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Counters
    logic [$clog2(MAX_ELEMENTS)-1:0] element_count;
    
    // Operation result
    logic signed [DATA_WIDTH:0] op_result;  // Extra bit for overflow detection
    logic [DATA_WIDTH-1:0] final_result;
    
    // Saturation logic
    function automatic [DATA_WIDTH-1:0] saturate(input signed [DATA_WIDTH:0] val);
        if (val > 127) begin
            return 8'd127;
        end else if (val < -128) begin
            return 8'h80;  // -128
        end else begin
            return val[DATA_WIDTH-1:0];
        end
    endfunction
    
    // Q7.8 multiplication (multiply then shift right by 7)
    function automatic [DATA_WIDTH-1:0] qmul(
        input signed [DATA_WIDTH-1:0] a,
        input signed [DATA_WIDTH-1:0] b
    );
        logic signed [2*DATA_WIDTH-1:0] full_product;
        logic signed [DATA_WIDTH:0] rounded;
        
        full_product = a * b;
        // Round: add 0.5 (1 << 6) before shifting
        rounded = (full_product + (1 << 6)) >>> 7;
        return saturate(rounded);
    endfunction
    
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            element_count <= '0;
            out_valid <= 1'b0;
        end else begin
            state <= next_state;
            out_valid <= 1'b0;
            
            case (state)
                IDLE: begin
                    element_count <= '0;
                    if (start) begin
                        // Ready to start
                    end
                end
                
                PROCESSING: begin
                    if (data_a_valid && element_count < num_elements) begin
                        // Perform operation based on opcode
                        case (operation)
                            VEC_ADD: begin
                                // Saturating addition
                                op_result <= $signed(data_a_in) + $signed(data_b_in);
                                final_result <= saturate(op_result);
                            end
                            
                            VEC_SUB: begin
                                // Saturating subtraction
                                op_result <= $signed(data_a_in) - $signed(data_b_in);
                                final_result <= saturate(op_result);
                            end
                            
                            VEC_MUL: begin
                                // Q7.8 multiplication
                                final_result <= qmul($signed(data_a_in), $signed(data_b_in));
                            end
                            
                            VEC_SCALE: begin
                                // Multiply by immediate
                                final_result <= qmul($signed(data_a_in), $signed(immediate));
                            end
                            
                            VEC_CLAMP: begin
                                // Clamp to range [immediate, data_b_in]
                                // immediate = min, data_b_in = max
                                if ($signed(data_a_in) > $signed(data_b_in)) begin
                                    final_result <= data_b_in;
                                end else if ($signed(data_a_in) < $signed(immediate)) begin
                                    final_result <= immediate;
                                end else begin
                                    final_result <= data_a_in;
                                end
                            end
                            
                            VEC_COPY, VEC_COPY2D, VEC_NOP: begin
                                // Passthrough
                                final_result <= data_a_in;
                            end
                            
                            default: begin
                                final_result <= data_a_in;
                            end
                        endcase
                        
                        data_out <= final_result;
                        out_valid <= 1'b1;
                        element_count <= element_count + 1;
                    end
                end
                
                DONE_STATE: begin
                    out_valid <= 1'b0;
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
                if (element_count >= num_elements) begin
                    next_state = DONE_STATE;
                end
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Status
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule
