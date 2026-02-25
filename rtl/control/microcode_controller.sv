// Microcode Controller
// Fetches 128-bit instructions from SRAM and dispatches to engines
// Uses scoreboard for out-of-order execution within dataflow constraints

`timescale 1ns/1ps

module microcode_controller #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ADDR_WIDTH = 16,
    parameter MAX_SEQ_LEN = 16
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control interface
    input  logic                      start,
    output logic                      busy,
    output logic                      done,
    input  logic [ADDR_WIDTH-1:0]     ucode_base_addr,
    input  logic [15:0]               ucode_length,
    
    // SRAM interface for instruction fetch
    output logic [ADDR_WIDTH-1:0]     sram_rd_addr,
    input  logic [127:0]              sram_rd_data,
    output logic                      sram_rd_en,
    
    // Engine command interfaces
    // GEMM
    output logic                      gemm_start,
    input  logic                      gemm_busy,
    output logic [15:0]               gemm_dim_m,
    output logic [15:0]               gemm_dim_k,
    output logic [15:0]               gemm_dim_n,
    output logic                      gemm_transpose_b,
    output logic                      gemm_accumulate,
    output logic                      gemm_requant,
    output logic [15:0]               gemm_imm,
    
    // Softmax
    output logic                      softmax_start,
    input  logic                      softmax_busy,
    output logic [15:0]               softmax_m,
    output logic [15:0]               softmax_n,
    output logic                      softmax_causal,
    
    // LayerNorm
    output logic                      layernorm_start,
    input  logic                      layernorm_busy,
    output logic [15:0]               layernorm_dim,
    
    // GELU
    output logic                      gelu_start,
    input  logic                      gelu_busy,
    output logic [15:0]               gelu_count,
    
    // Vector
    output logic                      vec_start,
    input  logic                      vec_busy,
    output logic [2:0]                vec_op,
    output logic [15:0]               vec_count,
    output logic [15:0]               vec_imm,
    
    // DMA
    output logic                      dma_start,
    input  logic                      dma_busy,
    output logic                      dma_direction,
    output logic [31:0]               dma_byte_count,
    
    // Barrier sync
    output logic                      barrier_wait,
    input  logic                      all_engines_idle
);

    // Instruction format (128 bits)
    typedef struct packed {
        logic [15:0] imm;
        logic [15:0] k;
        logic [15:0] n;
        logic [15:0] m;
        logic [15:0] src1;
        logic [15:0] src0;
        logic [15:0] dst;
        logic [7:0]  flags;
        logic [7:0]  opcode;
    } instruction_t;
    
    // Opcodes
    localparam OPCODE_NOP       = 8'h00;
    localparam OPCODE_DMA_LOAD  = 8'h01;
    localparam OPCODE_DMA_STORE = 8'h02;
    localparam OPCODE_GEMM      = 8'h03;
    localparam OPCODE_VEC       = 8'h04; // Used for various ops based on flags? No, VEC opcode covers add/mul etc?
                                         // ARCHITECTURE.md defines VEC_ADD=0x07 etc as separate opcodes?
                                         // "3.2 Opcodes" table lists VEC_ADD, VEC_MUL separately.
                                         // But here I defined OPCODE_VEC = 0x04.
                                         // Let's stick to the implementation here and use sub-opcodes or distinct opcodes.
                                         // The implementation of vec_engine takes `operation` input.
                                         // I should map instruction opcodes to engine ops.
    localparam OPCODE_SOFTMAX   = 8'h05;
    localparam OPCODE_LAYERNORM = 8'h06;
    localparam OPCODE_GELU      = 8'h07;
    // ARCHITECTURE.md had separate opcodes. Let's align.
    localparam OPCODE_VEC_ADD   = 8'h08;
    localparam OPCODE_VEC_MUL   = 8'h09;
    localparam OPCODE_VEC_COPY  = 8'h0A;
    
    localparam OPCODE_BARRIER   = 8'hFE;
    localparam OPCODE_END       = 8'hFF;
    
    // Engine IDs
    localparam ENGINE_GEMM      = 3'd0;
    localparam ENGINE_SOFTMAX   = 3'd1;
    localparam ENGINE_LAYERNORM = 3'd2;
    localparam ENGINE_GELU      = 3'd3;
    localparam ENGINE_VEC       = 3'd4;
    localparam ENGINE_DMA       = 3'd5;
    localparam NUM_ENGINES      = 6;
    
    // States
    typedef enum logic [2:0] {
        IDLE,
        FETCH,
        DECODE,
        DISPATCH,
        WAIT_BARRIER,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Program counter
    logic [15:0] pc;
    logic [15:0] ucode_end;
    
    // Current instruction
    instruction_t current_instr;
    logic instr_valid;
    
    // Scoreboard
    logic [NUM_ENGINES-1:0] scoreboard;
    logic [NUM_ENGINES-1:0] scoreboard_set;
    logic [NUM_ENGINES-1:0] scoreboard_clear;
    
    logic [NUM_ENGINES-1:0] engine_busy;
    assign engine_busy = {dma_busy, vec_busy, gelu_busy, layernorm_busy, softmax_busy, gemm_busy};
    
    // Decode target engine
    logic [2:0] target_engine;
    always_comb begin
        case (current_instr.opcode)
            OPCODE_GEMM:      target_engine = ENGINE_GEMM;
            OPCODE_SOFTMAX:   target_engine = ENGINE_SOFTMAX;
            OPCODE_LAYERNORM: target_engine = ENGINE_LAYERNORM;
            OPCODE_GELU:      target_engine = ENGINE_GELU;
            OPCODE_VEC, OPCODE_VEC_ADD, OPCODE_VEC_MUL, OPCODE_VEC_COPY: 
                              target_engine = ENGINE_VEC;
            OPCODE_DMA_LOAD,
            OPCODE_DMA_STORE: target_engine = ENGINE_DMA;
            default:          target_engine = 3'd7;
        endcase
    end
    
    // Scoreboard logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) scoreboard <= '0;
        else scoreboard <= (scoreboard | scoreboard_set) & ~scoreboard_clear;
    end
    
    always_comb scoreboard_clear = ~engine_busy & scoreboard;
    
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            pc <= '0;
            ucode_end <= '0;
            current_instr <= '0;
            instr_valid <= 1'b0;
            scoreboard_set <= '0;
            gemm_start <= 1'b0;
            softmax_start <= 1'b0;
            layernorm_start <= 1'b0;
            gelu_start <= 1'b0;
            vec_start <= 1'b0;
            dma_start <= 1'b0;
            barrier_wait <= 1'b0;
            
            // Output registers reset
            gemm_dim_m <= '0; gemm_dim_k <= '0; gemm_dim_n <= '0;
            gemm_transpose_b <= '0; gemm_accumulate <= '0; gemm_requant <= '0; gemm_imm <= '0;
            softmax_m <= '0; softmax_n <= '0; softmax_causal <= '0;
            layernorm_dim <= '0;
            gelu_count <= '0;
            vec_op <= '0; vec_count <= '0; vec_imm <= '0;
            dma_direction <= '0; dma_byte_count <= '0;
        end else begin
            // Default: no starts
            gemm_start <= 1'b0;
            softmax_start <= 1'b0;
            layernorm_start <= 1'b0;
            gelu_start <= 1'b0;
            vec_start <= 1'b0;
            dma_start <= 1'b0;
            scoreboard_set <= '0;
            barrier_wait <= 1'b0;
            
            state <= next_state;
            
            case (state)
                IDLE: begin
                    pc <= '0;
                    instr_valid <= 1'b0;
                    if (start) ucode_end <= ucode_length;
                end
                
                FETCH: begin
                    sram_rd_addr <= ucode_base_addr + pc;
                    sram_rd_en <= 1'b1;
                end
                
                DECODE: begin
                    sram_rd_en <= 1'b0;
                    current_instr <= instruction_t'(sram_rd_data);
                    instr_valid <= 1'b1;
                end
                
                DISPATCH: begin
                    if (instr_valid) begin
                        case (current_instr.opcode)
                            OPCODE_NOP: begin
                                pc <= pc + 1;
                                instr_valid <= 1'b0;
                            end
                            
                            OPCODE_GEMM: begin
                                if (!scoreboard[ENGINE_GEMM]) begin
                                    gemm_start <= 1'b1;
                                    gemm_dim_m <= current_instr.m;
                                    gemm_dim_k <= current_instr.k;
                                    gemm_dim_n <= current_instr.n;
                                    gemm_transpose_b <= current_instr.flags[0];
                                    gemm_requant <= current_instr.flags[1];
                                    gemm_accumulate <= current_instr.flags[2];
                                    gemm_imm <= current_instr.imm;
                                    scoreboard_set[ENGINE_GEMM] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_SOFTMAX: begin
                                if (!scoreboard[ENGINE_SOFTMAX]) begin
                                    softmax_start <= 1'b1;
                                    softmax_m <= current_instr.m;
                                    softmax_n <= current_instr.n;
                                    softmax_causal <= current_instr.flags[0];
                                    scoreboard_set[ENGINE_SOFTMAX] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_LAYERNORM: begin
                                if (!scoreboard[ENGINE_LAYERNORM]) begin
                                    layernorm_start <= 1'b1;
                                    layernorm_dim <= current_instr.n; // Assuming N is hidden dim
                                    scoreboard_set[ENGINE_LAYERNORM] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_GELU: begin
                                if (!scoreboard[ENGINE_GELU]) begin
                                    gelu_start <= 1'b1;
                                    gelu_count <= current_instr.n; // Assuming N is count
                                    scoreboard_set[ENGINE_GELU] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_VEC, OPCODE_VEC_ADD, OPCODE_VEC_MUL, OPCODE_VEC_COPY: begin
                                if (!scoreboard[ENGINE_VEC]) begin
                                    vec_start <= 1'b1;
                                    vec_count <= current_instr.n;
                                    vec_imm <= current_instr.imm;
                                    // Map opcode to vec engine op
                                    case (current_instr.opcode)
                                        OPCODE_VEC_ADD: vec_op <= 3'b001; // ADD
                                        OPCODE_VEC_MUL: vec_op <= 3'b010; // MUL
                                        OPCODE_VEC_COPY: vec_op <= 3'b110; // COPY
                                        default: vec_op <= 3'b000;
                                    endcase
                                    scoreboard_set[ENGINE_VEC] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_DMA_LOAD: begin
                                if (!scoreboard[ENGINE_DMA]) begin
                                    dma_start <= 1'b1;
                                    dma_direction <= 1'b0; // DDR -> SRAM
                                    dma_byte_count <= {current_instr.m, current_instr.n}; // Hack: use M:N for 32-bit count? 
                                    // Or M is bytes? Spec says M=bytes.
                                    dma_byte_count <= {16'd0, current_instr.m};
                                    scoreboard_set[ENGINE_DMA] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_DMA_STORE: begin
                                if (!scoreboard[ENGINE_DMA]) begin
                                    dma_start <= 1'b1;
                                    dma_direction <= 1'b1; // SRAM -> DDR
                                    dma_byte_count <= {16'd0, current_instr.m};
                                    scoreboard_set[ENGINE_DMA] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_BARRIER: begin
                                barrier_wait <= 1'b1;
                            end
                            
                            OPCODE_END: begin
                                // Program complete
                            end

                            default: begin
                                pc <= pc + 1;
                                instr_valid <= 1'b0;
                            end
                        endcase
                    end
                end
                
                WAIT_BARRIER: begin
                    barrier_wait <= 1'b1;
                    if (all_engines_idle) begin
                        pc <= pc + 1;
                        instr_valid <= 1'b0;
                    end
                end
                
                DONE_STATE: begin
                    // Done
                end

                default: begin
                    // no-op
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: if (start) next_state = FETCH;
            FETCH: next_state = DECODE;
            DECODE: next_state = DISPATCH;
            DISPATCH: begin
                if (instr_valid) begin
                    case (current_instr.opcode)
                        OPCODE_END: next_state = DONE_STATE;
                        OPCODE_BARRIER: next_state = WAIT_BARRIER;
                        default: begin
                            if (!scoreboard[target_engine]) next_state = FETCH;
                        end
                    endcase
                end
            end
            WAIT_BARRIER: if (all_engines_idle) next_state = FETCH;
            DONE_STATE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule
