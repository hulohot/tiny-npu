// Layer Normalization Engine
// Two-pass algorithm:
// Pass 1: Compute mean and variance across hidden dimension
// Pass 2: Normalize, scale by gamma, add beta

`timescale 1ns/1ps

module layernorm_engine #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter MAX_HIDDEN_DIM = 64,
    parameter EPS = 32'h00001000  // Small epsilon value (fixed-point)
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control
    input  logic                      start,
    output logic                      busy,
    output logic                      done,
    
    // Configuration
    input  logic [$clog2(MAX_HIDDEN_DIM)-1:0] hidden_dim,
    
    // Data input (one element per cycle)
    input  logic [DATA_WIDTH-1:0]     data_in,
    input  logic                      data_valid,
    
    // Gamma/beta parameters (from SRAM1)
    input  logic [DATA_WIDTH-1:0]     gamma_in,      // Scale
    input  logic [DATA_WIDTH-1:0]     beta_in,       // Shift
    input  logic                      param_valid,
    
    // Data output
    output logic [DATA_WIDTH-1:0]     data_out,
    output logic                      out_valid
);

    // States
    typedef enum logic [2:0] {
        IDLE,
        PASS1_MEAN_VAR,   // Accumulate sum and sum of squares
        COMPUTE_STATS,    // Calculate mean and variance
        PASS2_NORM,       // Normalize and apply gamma/beta
        DONE_STATE
    } state_t;
    
    state_t state, next_state;

    logic [ACC_WIDTH-1:0] hidden_dim_ext;
    assign hidden_dim_ext = ACC_WIDTH'(hidden_dim);
    
    // Accumulators for pass 1
    logic signed [ACC_WIDTH-1:0] sum_acc;
    logic signed [ACC_WIDTH-1:0] sum_sq_acc;
    logic [$clog2(MAX_HIDDEN_DIM)-1:0] element_count;
    
    // Computed statistics (pass 1 → pass 2)
    logic signed [ACC_WIDTH-1:0] mean;
    logic signed [ACC_WIDTH-1:0] variance;
    logic signed [ACC_WIDTH-1:0] inv_sqrt_var;  // 1/sqrt(var + eps)
    
    // Input buffer (store for pass 2)
    logic [DATA_WIDTH-1:0] input_buffer [0:MAX_HIDDEN_DIM-1];
    
    // Gamma/beta buffer
    logic [DATA_WIDTH-1:0] gamma_buffer [0:MAX_HIDDEN_DIM-1];
    logic [DATA_WIDTH-1:0] beta_buffer [0:MAX_HIDDEN_DIM-1];
    
    // Current processing index
    logic [$clog2(MAX_HIDDEN_DIM)-1:0] current_idx;
    logic [$clog2(MAX_HIDDEN_DIM)-1:0] param_count;
    
    // Inverse square root LUT
    // Maps variance value to 1/sqrt(var + eps)
    // Using Q16.16 fixed point for precision
    logic [31:0] rsqrt_lut [0:1023];
    
    initial begin
        // Initialize rsqrt LUT
        // For value v, store 1/sqrt(v + eps) * 2^16
        for (int i = 0; i < 1024; i++) begin
            real v = i + 1;  // Avoid division by zero
            real rsqrt_val = 1.0 / $sqrt(v);
            rsqrt_lut[i] = int'(rsqrt_val * 65536) & 32'hFFFFFFFF;
        end
    end
    
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            sum_acc <= '0;
            sum_sq_acc <= '0;
            element_count <= '0;
            current_idx <= '0;
            param_count <= '0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    sum_acc <= '0;
                    sum_sq_acc <= '0;
                    element_count <= '0;
                    current_idx <= '0;
                    
                    if (start) begin
                        // Nothing to initialize yet
                    end
                end
                
                PASS1_MEAN_VAR: begin
                    if (data_valid) begin
                        // Store input
                        input_buffer[element_count] <= data_in;
                        
                        // Accumulate sum
                        sum_acc <= sum_acc + ACC_WIDTH'($signed(data_in));
                        
                        // Accumulate sum of squares
                        sum_sq_acc <= sum_sq_acc + (ACC_WIDTH'($signed(data_in)) * ACC_WIDTH'($signed(data_in)));
                        
                        element_count <= element_count + 1;
                    end
                end
                
                COMPUTE_STATS: begin
                    // Compute mean = sum / N
                    mean <= sum_acc / hidden_dim_ext;
                    
                    // Compute variance = E[x^2] - (E[x])^2
                    // var = sum_sq / N - mean^2
                    variance <= (sum_sq_acc / hidden_dim_ext) - 
                                ((sum_acc / hidden_dim_ext) * (sum_acc / hidden_dim_ext));
                    
                    // Lookup 1/sqrt(var + eps)
                    // Clamp variance to LUT range
                    if (variance > 0 && variance < 1024) begin
                        inv_sqrt_var <= rsqrt_lut[variance[9:0]];
                    end else if (variance >= 1024) begin
                        inv_sqrt_var <= rsqrt_lut[1023];  // Clamp
                    end else begin
                        inv_sqrt_var <= rsqrt_lut[0];  // Minimum variance
                    end
                    
                    current_idx <= '0;
                end
                
                PASS2_NORM: begin
                    if (current_idx < hidden_dim) begin
                        // x_norm = (x - mean) * inv_sqrt_var
                        logic signed [DATA_WIDTH-1:0] x;
                        logic signed [ACC_WIDTH-1:0] x_minus_mean;
                        logic signed [ACC_WIDTH-1:0] x_norm;
                        logic signed [ACC_WIDTH-1:0] scaled;
                        logic signed [ACC_WIDTH-1:0] shifted;
                        
                        x = $signed(input_buffer[current_idx]);
                        x_minus_mean = ACC_WIDTH'($signed(x)) - mean;
                        
                        // Multiply by inv_sqrt_var (Q16.16 format)
                        // Result needs to be shifted right by 16
                        x_norm = (x_minus_mean * inv_sqrt_var) >>> 16;
                        
                        // Apply gamma (scale)
                        scaled = (x_norm * $signed(gamma_buffer[current_idx])) >>> 7;
                        
                        // Apply beta (shift)
                        shifted = scaled + ACC_WIDTH'($signed(beta_buffer[current_idx]));
                        
                        // Clamp to INT8 range
                        if (shifted > 127) begin
                            input_buffer[current_idx] <= 8'd127;
                        end else if (shifted < -128) begin
                            input_buffer[current_idx] <= 8'h80;  // -128
                        end else begin
                            input_buffer[current_idx] <= shifted[7:0];
                        end
                        
                        current_idx <= current_idx + 1;
                    end
                end

                default: begin
                    // no-op
                end
            endcase
        end
    end
    
    // Capture gamma/beta parameters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            param_count <= '0;
        end else if (state == IDLE) begin
            if (param_valid) begin
                gamma_buffer[param_count] <= gamma_in;
                beta_buffer[param_count] <= beta_in;
                param_count <= param_count + 1;
            end
        end else begin
            param_count <= '0;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) next_state = PASS1_MEAN_VAR;
            end
            
            PASS1_MEAN_VAR: begin
                if (element_count >= hidden_dim) next_state = COMPUTE_STATS;
            end
            
            COMPUTE_STATS: begin
                next_state = PASS2_NORM;
            end
            
            PASS2_NORM: begin
                if (current_idx >= hidden_dim) next_state = DONE_STATE;
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output
    logic [$clog2(MAX_HIDDEN_DIM)-1:0] out_idx;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_idx <= '0;
            out_valid <= 1'b0;
        end else if (state == DONE_STATE) begin
            if (out_idx < hidden_dim) begin
                data_out <= input_buffer[out_idx];
                out_valid <= 1'b1;
                out_idx <= out_idx + 1;
            end else begin
                out_valid <= 1'b0;
            end
        end else begin
            out_idx <= '0;
            out_valid <= 1'b0;
        end
    end
    
    // Status
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule
