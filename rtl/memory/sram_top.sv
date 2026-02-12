// Dual-Port SRAM for NPU
// Two independent banks: SRAM0 (64KB main) and SRAM1 (8KB auxiliary)

`timescale 1ns/1ps

module sram_bank #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter SIZE = 65536  // 64KB default
)(
    input  logic                      clk,
    
    // Port A (read/write)
    input  logic [ADDR_WIDTH-1:0]     addr_a,
    input  logic [DATA_WIDTH-1:0]     wdata_a,
    output logic [DATA_WIDTH-1:0]     rdata_a,
    input  logic                      we_a,      // Write enable
    input  logic                      re_a,      // Read enable
    
    // Port B (read only)
    input  logic [ADDR_WIDTH-1:0]     addr_b,
    output logic [DATA_WIDTH-1:0]     rdata_b,
    input  logic                      re_b
);

    // Memory array
    logic [DATA_WIDTH-1:0] mem [0:SIZE-1];
    
    // Port A operation
    always_ff @(posedge clk) begin
        if (we_a) begin
            mem[addr_a] <= wdata_a;
        end
        if (re_a) begin
            rdata_a <= mem[addr_a];
        end
    end
    
    // Port B operation (read only)
    always_ff @(posedge clk) begin
        if (re_b) begin
            rdata_b <= mem[addr_b];
        end
    end

endmodule


// SRAM Top Level - Both banks with arbitration
module sram_top #(
    parameter DATA_WIDTH = 8
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Engine interfaces (request-based)
    // Each engine can request read/write access
    
    // GEMM engine
    input  logic [15:0]               gemm_rd_addr,
    output logic [DATA_WIDTH-1:0]     gemm_rd_data,
    input  logic                      gemm_rd_en,
    input  logic [15:0]               gemm_wr_addr,
    input  logic [DATA_WIDTH-1:0]     gemm_wr_data,
    input  logic                      gemm_wr_en,
    
    // Softmax engine
    input  logic [15:0]               softmax_rd_addr,
    output logic [DATA_WIDTH-1:0]     softmax_rd_data,
    input  logic                      softmax_rd_en,
    
    // LayerNorm engine
    input  logic [15:0]               layernorm_rd_addr,
    output logic [DATA_WIDTH-1:0]     layernorm_rd_data,
    input  logic                      layernorm_rd_en,
    input  logic [15:0]               layernorm_wr_addr,
    input  logic [DATA_WIDTH-1:0]     layernorm_wr_data,
    input  logic                      layernorm_wr_en,
    input  logic [15:0]               layernorm_rd_addr_b,  // For beta/gamma
    output logic [DATA_WIDTH-1:0]     layernorm_rd_data_b,
    input  logic                      layernorm_rd_en_b,
    
    // GELU engine
    input  logic [15:0]               gelu_rd_addr,
    output logic [DATA_WIDTH-1:0]     gelu_rd_data,
    input  logic                      gelu_rd_en,
    input  logic [15:0]               gelu_wr_addr,
    input  logic [DATA_WIDTH-1:0]     gelu_wr_data,
    input  logic                      gelu_wr_en,
    
    // Vector engine
    input  logic [15:0]               vec_rd_addr,
    output logic [DATA_WIDTH-1:0]     vec_rd_data,
    input  logic                      vec_rd_en,
    input  logic [15:0]               vec_rd_addr_b,  // For binary ops
    output logic [DATA_WIDTH-1:0]     vec_rd_data_b,
    input  logic                      vec_rd_en_b,
    input  logic [15:0]               vec_wr_addr,
    input  logic [DATA_WIDTH-1:0]     vec_wr_data,
    input  logic                      vec_wr_en,
    
    // DMA engine
    input  logic [15:0]               dma_rd_addr,
    output logic [DATA_WIDTH-1:0]     dma_rd_data,
    input  logic                      dma_rd_en,
    input  logic [15:0]               dma_wr_addr,
    input  logic [DATA_WIDTH-1:0]     dma_wr_data,
    input  logic                      dma_wr_en,
    
    // Microcode storage (read only by controller)
    input  logic [15:0]               ucode_rd_addr,
    output logic [127:0]              ucode_rd_data,  // 128-bit instructions
    input  logic                      ucode_rd_en
);

    // Priority arbiter (simple round-robin or fixed priority)
    // For now: fixed priority - DMA > GEMM > Softmax > LayerNorm > GELU > Vec
    
    // SRAM0 (64KB) - Main workspace
    sram_bank #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(16),
        .SIZE(65536)
    ) sram0 (
        .clk(clk),
        // Port A - multiplexed access
        .addr_a(/* multiplexed */),
        .wdata_a(/* multiplexed */),
        .rdata_a(/* multiplexed */),
        .we_a(/* multiplexed */),
        .re_a(/* multiplexed */),
        // Port B - read only
        .addr_b(/* multiplexed */),
        .rdata_b(/* multiplexed */),
        .re_b(/* multiplexed */)
    );
    
    // SRAM1 (8KB) - Auxiliary (LayerNorm beta/gamma, residuals)
    sram_bank #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(16),
        .SIZE(8192)
    ) sram1 (
        .clk(clk),
        .addr_a(/* multiplexed */),
        .wdata_a(/* multiplexed */),
        .rdata_a(/* multiplexed */),
        .we_a(/* multiplexed */),
        .re_a(/* multiplexed */),
        .addr_b(/* multiplexed */),
        .rdata_b(/* multiplexed */),
        .re_b(/* multiplexed */)
    );
    
    // TODO: Implement full arbitration logic
    // For now, this is a structural placeholder
    
    assign gemm_rd_data = '0;
    assign softmax_rd_data = '0;
    assign layernorm_rd_data = '0;
    assign layernorm_rd_data_b = '0;
    assign gelu_rd_data = '0;
    assign vec_rd_data = '0;
    assign vec_rd_data_b = '0;
    assign dma_rd_data = '0;
    assign ucode_rd_data = '0;

endmodule
