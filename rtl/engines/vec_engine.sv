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
        logic signed [2*DATA_WIDTH-1:0] rounded_full;
        
        full_product = a * b;
        // Round: add 0.5 (1 << 6) before shifting
        rounded_full = (full_product + (1 << 6)) >>> 7;
        return saturate(rounded_full[DATA_WIDTH:0]);
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
                        // Compute result directly into data_out in a single register stage.
                        // The original 2-stage pipeline (op_result → final_result → data_out)
                        // caused the last two outputs per burst to be silently dropped when the
                        // FSM transitioned to DONE_STATE before the pipeline drained.
                        case (operation)
                            VEC_ADD: begin
                                // Sign-extend to 9 bits before adding to detect overflow.
                                data_out <= saturate(
                                    $signed({data_a_in[DATA_WIDTH-1], data_a_in}) +
                                    $signed({data_b_in[DATA_WIDTH-1], data_b_in}));
                            end

                            VEC_SUB: begin
                                data_out <= saturate(
                                    $signed({data_a_in[DATA_WIDTH-1], data_a_in}) -
                                    $signed({data_b_in[DATA_WIDTH-1], data_b_in}));
                            end

                            VEC_MUL: begin
                                data_out <= qmul($signed(data_a_in), $signed(data_b_in));
                            end

                            VEC_SCALE: begin
                                data_out <= qmul($signed(data_a_in), $signed(immediate));
                            end

                            VEC_CLAMP: begin
                                if ($signed(data_a_in) > $signed(data_b_in))
                                    data_out <= data_b_in;
                                else if ($signed(data_a_in) < $signed(immediate))
                                    data_out <= immediate;
                                else
                                    data_out <= data_a_in;
                            end

                            default: begin  // VEC_COPY, VEC_COPY2D, VEC_NOP
                                data_out <= data_a_in;
                            end
                        endcase

                        out_valid <= 1'b1;
                        element_count <= element_count + 1;
                    end
                end
                
                DONE_STATE: begin
                    out_valid <= 1'b0;
                end

                default: begin
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

            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Status
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule
