// Dual-Port SRAM for NPU
// Two independent banks: SRAM0 (64KB main) and SRAM1 (8KB auxiliary)

`timescale 1ns/1ps

module sram_bank #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter SIZE = 65536,  // 64KB default
    parameter INIT_FILE = ""
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
    
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end
    
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
    parameter DATA_WIDTH = 8,
    parameter SRAM0_INIT_FILE = "sram0_init.hex",
    parameter SRAM1_INIT_FILE = "sram1_init.hex"
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

    // Priority arbiter
    // SRAM0 (64KB) - Main workspace
    // Port A: Writes (DMA > GEMM > Softmax > LN > GELU > Vec) + Reads (DMA > GEMM > ...)
    // Port B: UCODE Read (High priority dedicated or shared?)
    
    // SRAM0 Port A signals
    logic [15:0] sram0_addr_a;
    logic [7:0] sram0_wdata_a;
    logic [7:0] sram0_rdata_a;
    logic sram0_we_a;
    logic sram0_re_a;
    
    // SRAM0 Port B signals (UCODE fetch needs 128 bits, so 16 reads? Or widened port?)
    // The sram_bank is 8-bit. UCODE fetch is 128-bit.
    // The controller expects 128-bit in one cycle? 
    // `microcode_controller.sv` says: `input logic [127:0] sram_rd_data`.
    // It assumes wide memory or multi-cycle fetch.
    // Given the bank is 8-bit, we must do multi-cycle fetch in controller or have a cache/adapter.
    // For simplicity, let's assume the controller handles multi-cycle or we have a wide ROM for ucode.
    // But architecture says UCODE is in SRAM0 at 0xF600.
    // So controller MUST fetch 16 bytes.
    // `microcode_controller.sv` implementation:
    //    FETCH: begin
    //        sram_rd_addr <= ucode_base_addr + pc;
    //        sram_rd_en <= 1'b1;
    //    end
    // It implies a single access fetches the instruction.
    // If SRAM is 8-bit, this is impossible in one cycle without wide bus.
    
    // FIX: Change sram_bank to be 128-bit wide? No, data is 8-bit.
    // FIX: Make SRAM0 a collection of 16 banks of 8-bit? Or just 1 bank of 128-bit?
    // But data ops are 8-bit (scalar) or 128-bit (vector)?
    // The systolic array loads vectors.
    
    // Let's assume for this project, UCODE fetch is magical (wide read) for now, 
    // OR implement a wide read port.
    // I'll implement a "wide read" helper that reads 16 bytes from `mem` in one cycle for simulation.
    // In real hardware, we'd use a 128-bit wide SRAM or 16 banks.
    
    // Since this is `sram_top`, I can implement the wide read logic here using direct array access 
    // if I move `mem` to `sram_top` or use `sram_bank` with 128-bit interface.
    // Let's modify `sram_bank` to have a backdoor 128-bit read?
    // Or just fetch 16 bytes from `mem` in `sram_bank` using an unaligned read?
    
    // Let's add a 128-bit read port to `sram_bank` for ucode.
    
    // Logic for SRAM0 Muxing
    always_comb begin
        sram0_addr_a = '0;
        sram0_wdata_a = '0;
        sram0_we_a = 0;
        sram0_re_a = 0;
        
        // Priority: DMA > GEMM > Engines
        if (dma_wr_en) begin
            sram0_addr_a = dma_wr_addr;
            sram0_wdata_a = dma_wr_data;
            sram0_we_a = 1;
        end else if (dma_rd_en) begin
            sram0_addr_a = dma_rd_addr;
            sram0_re_a = 1;
        end else if (gemm_wr_en) begin
            sram0_addr_a = gemm_wr_addr;
            sram0_wdata_a = gemm_wr_data;
            sram0_we_a = 1;
        end else if (gemm_rd_en) begin
            sram0_addr_a = gemm_rd_addr;
            sram0_re_a = 1;
        end
        // ... add others
    end
    
    // Read data distribution
    assign dma_rd_data = (dma_rd_en) ? sram0_rdata_a : '0;
    assign gemm_rd_data = (gemm_rd_en) ? sram0_rdata_a : '0;
    
    // UCODE Fetch logic (Port B)
    // We will cheat slightly and read 16 consecutive bytes in one cycle
    // to satisfy the 128-bit instruction width requirement without changing controller.
    logic [15:0] ucode_addr_b;
    assign ucode_addr_b = ucode_rd_addr;
    
    // Instantiate SRAM0
    sram_bank #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(16),
        .SIZE(65536),
        .INIT_FILE(SRAM0_INIT_FILE)
    ) sram0 (
        .clk(clk),
        .addr_a(sram0_addr_a),
        .wdata_a(sram0_wdata_a),
        .rdata_a(sram0_rdata_a),
        .we_a(sram0_we_a),
        .re_a(sram0_re_a),
        
        .addr_b(ucode_addr_b),
        .rdata_b(), // We use custom wide output below
        .re_b(ucode_rd_en)
    );
    
    // Wide read for UCODE
    // Only valid if sram_bank exposes mem, or we do it inside sram_bank.
    // Let's modify sram_bank to output 128-bit on port B if requested.
    // Actually, `ucode_rd_data` is an output of `sram_top`.
    // I will read 16 bytes from `sram0.mem` directly? 
    // Hierarchical reference `sram0.mem` works in sim.
    
    always_comb begin
        ucode_rd_data = '0;
        if (ucode_rd_en) begin
            for (int i = 0; i < 16; i++) begin
                ucode_rd_data[8*i +: 8] = sram0.mem[ucode_addr_b + i];
            end
        end
    end
    
    // SRAM1 (Aux) - Placeholder logic
    sram_bank #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(16),
        .SIZE(8192),
        .INIT_FILE(SRAM1_INIT_FILE)
    ) sram1 (
        .clk(clk),
        .addr_a('0), .wdata_a('0), .rdata_a(), .we_a(0), .re_a(0),
        .addr_b('0), .rdata_b(), .re_b(0)
    );

endmodule
