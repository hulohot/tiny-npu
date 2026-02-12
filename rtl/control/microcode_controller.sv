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
    output logic [15:0]               gemm_imm,
    
    // Softmax
    output logic                      softmax_start,
    input  logic                      softmax_busy,
    
    // LayerNorm
    output logic                      layernorm_start,
    input  logic                      layernorm_busy,
    
    // GELU
    output logic                      gelu_start,
    input  logic                      gelu_busy,
    
    // Vector
    output logic                      vec_start,
    input  logic                      vec_busy,
    
    // DMA
    output logic                      dma_start,
    input  logic                      dma_busy,
    
    // Barrier sync
    output logic                      barrier_wait,
    input  logic                      all_engines_idle
);

    // Instruction format (128 bits)
    // [127:112] - imm (16 bits)
    // [111:96]  - K (16 bits)
    // [95:80]   - N (16 bits)
    // [79:64]   - M (16 bits)
    // [63:48]   - src1 (16 bits)
    // [47:32]   - src0 (16 bits)
    // [31:16]   - dst (16 bits)
    // [15:8]    - flags (8 bits)
    // [7:0]     - opcode (8 bits)
    
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
    localparam OPCODE_VEC       = 8'h04;
    localparam OPCODE_SOFTMAX   = 8'h05;
    localparam OPCODE_LAYERNORM = 8'h06;
    localparam OPCODE_GELU      = 8'h07;
    localparam OPCODE_BARRIER   = 8'hFE;
    localparam OPCODE_END       = 8'hFF;
    
    // Engine IDs for scoreboard
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
    
    // Scoreboard (track busy engines)
    logic [NUM_ENGINES-1:0] scoreboard;
    logic [NUM_ENGINES-1:0] scoreboard_set;
    logic [NUM_ENGINES-1:0] scoreboard_clear;
    
    // Engine busy signals
    logic [NUM_ENGINES-1:0] engine_busy;
    assign engine_busy = {dma_busy, vec_busy, gelu_busy, layernorm_busy, softmax_busy, gemm_busy};
    
    // Decode: which engine does this instruction target?
    logic [2:0] target_engine;
    always_comb begin
        case (current_instr.opcode)
            OPCODE_GEMM:      target_engine = ENGINE_GEMM;
            OPCODE_SOFTMAX:   target_engine = ENGINE_SOFTMAX;
            OPCODE_LAYERNORM: target_engine = ENGINE_LAYERNORM;
            OPCODE_GELU:      target_engine = ENGINE_GELU;
            OPCODE_VEC:       target_engine = ENGINE_VEC;
            OPCODE_DMA_LOAD,
            OPCODE_DMA_STORE: target_engine = ENGINE_DMA;
            default:          target_engine = 3'd7;  // Invalid
        endcase
    end
    
    // Scoreboard logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scoreboard <= '0;
        end else begin
            // Set bit when instruction dispatched
            scoreboard <= (scoreboard | scoreboard_set) & ~scoreboard_clear;
        end
    end
    
    // Engine done signals clear scoreboard (in real design, these would be pulses)
    // For now, use inverted busy signals
    always_comb begin
        scoreboard_clear = ~engine_busy & scoreboard;
    end
    
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
                    if (start) begin
                        ucode_end <= ucode_length;
                    end
                end
                
                FETCH: begin
                    // Request instruction from SRAM
                    sram_rd_addr <= ucode_base_addr + pc;
                    sram_rd_en <= 1'b1;
                end
                
                DECODE: begin
                    sram_rd_en <= 1'b0;
                    // Capture instruction
                    current_instr <= instruction_t'(sram_rd_data);
                    instr_valid <= 1'b1;
                end
                
                DISPATCH: begin
                    if (instr_valid) begin
                        case (current_instr.opcode)
                            OPCODE_NOP: begin
                                // Do nothing, advance PC
                                pc <= pc + 1;
                                instr_valid <= 1'b0;
                            end
                            
                            OPCODE_GEMM: begin
                                if (!scoreboard[ENGINE_GEMM]) begin
                                    gemm_start <= 1'b1;
                                    gemm_dim_m <= current_instr.m;
                                    gemm_dim_k <= current_instr.k;
                                    gemm_dim_n <= current_instr.n;
                                    scoreboard_set[ENGINE_GEMM] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_SOFTMAX: begin
                                if (!scoreboard[ENGINE_SOFTMAX]) begin
                                    softmax_start <= 1'b1;
                                    scoreboard_set[ENGINE_SOFTMAX] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_LAYERNORM: begin
                                if (!scoreboard[ENGINE_LAYERNORM]) begin
                                    layernorm_start <= 1'b1;
                                    scoreboard_set[ENGINE_LAYERNORM] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_GELU: begin
                                if (!scoreboard[ENGINE_GELU]) begin
                                    gelu_start <= 1'b1;
                                    scoreboard_set[ENGINE_GELU] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_VEC: begin
                                if (!scoreboard[ENGINE_VEC]) begin
                                    vec_start <= 1'b1;
                                    scoreboard_set[ENGINE_VEC] <= 1'b1;
                                    pc <= pc + 1;
                                    instr_valid <= 1'b0;
                                end
                            end
                            
                            OPCODE_DMA_LOAD, OPCODE_DMA_STORE: begin
                                if (!scoreboard[ENGINE_DMA]) begin
                                    dma_start <= 1'b1;
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
                    // Program complete
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) next_state = FETCH;
            end
            
            FETCH: begin
                // Assume single cycle read for now
                next_state = DECODE;
            end
            
            DECODE: begin
                next_state = DISPATCH;
            end
            
            DISPATCH: begin
                if (instr_valid) begin
                    case (current_instr.opcode)
                        OPCODE_END: begin
                            next_state = DONE_STATE;
                        end
                        OPCODE_BARRIER: begin
                            next_state = WAIT_BARRIER;
                        end
                        default: begin
                            // If engine not available, stay in DISPATCH
                            if (!scoreboard[target_engine]) begin
                                next_state = FETCH;
                            end
                        end
                    endcase
                end
            end
            
            WAIT_BARRIER: begin
                if (all_engines_idle) next_state = FETCH;
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
