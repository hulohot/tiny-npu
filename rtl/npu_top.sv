// Tiny NPU Top-Level Module
// Integrates Microcode Controller, SRAM, DMA, and Compute Engines

`timescale 1ns/1ps

module npu_top #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ARRAY_SIZE = 16,
    parameter SRAM0_SIZE = 65536,  // 64KB
    parameter SRAM1_SIZE = 8192     // 8KB
)(
    input  logic        clk,
    input  logic        rst_n,
    
    // AXI4-Lite control interface
    input  logic [31:0] s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [31:0] s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,
    
    // AXI4 DDR interface
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,
    
    // Status
    output logic        busy,
    output logic        done
);

    // Internal signals
    
    // Control registers
    logic [31:0] ctrl_reg;
    logic [31:0] status_reg;
    logic [31:0] ucode_base_reg;
    logic [31:0] ucode_len_reg;
    logic [31:0] ddr_base_wgt_reg;
    logic [31:0] exec_mode_reg;
    
    // Start/Busy signals
    logic start_pulse;
    logic controller_busy;
    logic controller_done;
    
    // Engine control signals
    logic gemm_start, gemm_busy, gemm_done;
    logic softmax_start, softmax_busy, softmax_done;
    logic layernorm_start, layernorm_busy, layernorm_done;
    logic gelu_start, gelu_busy, gelu_done;
    logic vec_start, vec_busy, vec_done;
    logic dma_start, dma_busy, dma_done;
    
    // Engine configuration signals
    logic [15:0] gemm_dim_m, gemm_dim_k, gemm_dim_n;
    logic gemm_transpose_b, gemm_accumulate, gemm_requant;
    logic [15:0] gemm_imm;
    
    logic softmax_causal;
    logic [15:0] softmax_m, softmax_n;
    
    logic [15:0] layernorm_dim;
    logic [15:0] gelu_count;
    
    logic [2:0] vec_op;
    logic [15:0] vec_count;
    logic [15:0] vec_imm;
    
    logic dma_direction;
    logic [31:0] dma_byte_count;
    
    // SRAM interfaces
    // GEMM
    logic [15:0] gemm_rd_addr;
    logic [DATA_WIDTH-1:0] gemm_rd_data;
    logic gemm_rd_en;
    logic [15:0] gemm_wr_addr;
    logic [DATA_WIDTH-1:0] gemm_wr_data;
    logic gemm_wr_en;
    
    // Softmax
    logic [15:0] softmax_rd_addr;
    logic [DATA_WIDTH-1:0] softmax_rd_data;
    logic softmax_rd_en;
    
    // LayerNorm
    logic [15:0] layernorm_rd_addr;
    logic [DATA_WIDTH-1:0] layernorm_rd_data;
    logic layernorm_rd_en;
    logic [15:0] layernorm_wr_addr;
    logic [DATA_WIDTH-1:0] layernorm_wr_data;
    logic layernorm_wr_en;
    logic [15:0] layernorm_rd_addr_b;
    logic [DATA_WIDTH-1:0] layernorm_rd_data_b;
    logic layernorm_rd_en_b;
    
    // GELU
    logic [15:0] gelu_rd_addr;
    logic [DATA_WIDTH-1:0] gelu_rd_data;
    logic gelu_rd_en;
    logic [15:0] gelu_wr_addr;
    logic [DATA_WIDTH-1:0] gelu_wr_data;
    logic gelu_wr_en;
    
    // Vector
    logic [15:0] vec_rd_addr;
    logic [DATA_WIDTH-1:0] vec_rd_data;
    logic vec_rd_en;
    logic [15:0] vec_rd_addr_b;
    logic [DATA_WIDTH-1:0] vec_rd_data_b;
    logic vec_rd_en_b;
    logic [15:0] vec_wr_addr;
    logic [DATA_WIDTH-1:0] vec_wr_data;
    logic vec_wr_en;
    
    // DMA
    logic [15:0] dma_rd_addr;
    logic [DATA_WIDTH-1:0] dma_rd_data;
    logic dma_rd_en;
    logic [15:0] dma_wr_addr;
    logic [DATA_WIDTH-1:0] dma_wr_data;
    logic dma_wr_en;
    
    // Microcode
    logic [15:0] ucode_rd_addr;
    logic [127:0] ucode_rd_data;
    logic ucode_rd_en;
    
    // Register file instance
    npu_regs regs (
        .clk(clk),
        .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .ctrl_reg(ctrl_reg),
        .status_reg(status_reg),
        .ucode_base_reg(ucode_base_reg),
        .ucode_len_reg(ucode_len_reg),
        .ddr_base_wgt_reg(ddr_base_wgt_reg),
        .exec_mode_reg(exec_mode_reg),
        .busy(controller_busy),
        .done(controller_done)
    );
    
    // Start pulse generation
    logic start_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) start_r <= 1'b0;
        else start_r <= ctrl_reg[0];
    end
    assign start_pulse = ctrl_reg[0] && !start_r && !busy;
    
    assign busy = controller_busy;
    assign done = controller_done;
    
    // ========================================================================
    // Microcode Controller
    // ========================================================================
    microcode_controller #(.DATA_WIDTH(DATA_WIDTH)) controller (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_pulse),
        .busy(controller_busy),
        .done(controller_done),
        .ucode_base_addr(ucode_base_reg[15:0]),
        .ucode_length(ucode_len_reg[15:0]),
        .sram_rd_addr(ucode_rd_addr),
        .sram_rd_data(ucode_rd_data),
        .sram_rd_en(ucode_rd_en),
        
        .gemm_start(gemm_start),
        .gemm_busy(gemm_busy),
        .gemm_dim_m(gemm_dim_m),
        .gemm_dim_k(gemm_dim_k),
        .gemm_dim_n(gemm_dim_n),
        .gemm_transpose_b(gemm_transpose_b),
        .gemm_accumulate(gemm_accumulate),
        .gemm_requant(gemm_requant),
        .gemm_imm(gemm_imm),
        
        .softmax_start(softmax_start),
        .softmax_busy(softmax_busy),
        .softmax_m(softmax_m),
        .softmax_n(softmax_n),
        .softmax_causal(softmax_causal),
        
        .layernorm_start(layernorm_start),
        .layernorm_busy(layernorm_busy),
        .layernorm_dim(layernorm_dim),
        
        .gelu_start(gelu_start),
        .gelu_busy(gelu_busy),
        .gelu_count(gelu_count),
        
        .vec_start(vec_start),
        .vec_busy(vec_busy),
        .vec_op(vec_op),
        .vec_count(vec_count),
        .vec_imm(vec_imm),
        
        .dma_start(dma_start),
        .dma_busy(dma_busy),
        .dma_direction(dma_direction),
        .dma_byte_count(dma_byte_count),
        
        .barrier_wait(),
        .all_engines_idle(!gemm_busy && !softmax_busy && !layernorm_busy && 
                          !gelu_busy && !vec_busy && !dma_busy)
    );
    
    // ========================================================================
    // SRAM Top
    // ========================================================================
    sram_top #(.DATA_WIDTH(DATA_WIDTH)) sram (
        .clk(clk),
        .rst_n(rst_n),
        .gemm_rd_addr(gemm_rd_addr),
        .gemm_rd_data(gemm_rd_data),
        .gemm_rd_en(gemm_rd_en),
        .gemm_wr_addr(gemm_wr_addr),
        .gemm_wr_data(gemm_wr_data),
        .gemm_wr_en(gemm_wr_en),
        .softmax_rd_addr(softmax_rd_addr),
        .softmax_rd_data(softmax_rd_data),
        .softmax_rd_en(softmax_rd_en),
        .layernorm_rd_addr(layernorm_rd_addr),
        .layernorm_rd_data(layernorm_rd_data),
        .layernorm_rd_en(layernorm_rd_en),
        .layernorm_wr_addr(layernorm_wr_addr),
        .layernorm_wr_data(layernorm_wr_data),
        .layernorm_wr_en(layernorm_wr_en),
        .layernorm_rd_addr_b(layernorm_rd_addr_b),
        .layernorm_rd_data_b(layernorm_rd_data_b),
        .layernorm_rd_en_b(layernorm_rd_en_b),
        .gelu_rd_addr(gelu_rd_addr),
        .gelu_rd_data(gelu_rd_data),
        .gelu_rd_en(gelu_rd_en),
        .gelu_wr_addr(gelu_wr_addr),
        .gelu_wr_data(gelu_wr_data),
        .gelu_wr_en(gelu_wr_en),
        .vec_rd_addr(vec_rd_addr),
        .vec_rd_data(vec_rd_data),
        .vec_rd_en(vec_rd_en),
        .vec_rd_addr_b(vec_rd_addr_b),
        .vec_rd_data_b(vec_rd_data_b),
        .vec_rd_en_b(vec_rd_en_b),
        .vec_wr_addr(vec_wr_addr),
        .vec_wr_data(vec_wr_data),
        .vec_wr_en(vec_wr_en),
        .dma_rd_addr(dma_rd_addr),
        .dma_rd_data(dma_rd_data),
        .dma_rd_en(dma_rd_en),
        .dma_wr_addr(dma_wr_addr),
        .dma_wr_data(dma_wr_data),
        .dma_wr_en(dma_wr_en),
        .ucode_rd_addr(ucode_rd_addr),
        .ucode_rd_data(ucode_rd_data),
        .ucode_rd_en(ucode_rd_en)
    );
    
    // ========================================================================
    // Engines
    // ========================================================================
    
    gemm_engine #(.DATA_WIDTH(DATA_WIDTH)) gemm (
        .clk(clk),
        .rst_n(rst_n),
        .start(gemm_start),
        .busy(gemm_busy),
        .done(gemm_done),
        .src_a_addr('0), // TODO: Wire from controller instr
        .src_b_addr('0),
        .dst_addr('0),
        .dim_m(gemm_dim_m),
        .dim_k(gemm_dim_k),
        .dim_n(gemm_dim_n),
        .transpose_b(gemm_transpose_b),
        .accumulate(gemm_accumulate),
        .scale(gemm_imm[15:8]),
        .shift(gemm_imm[7:0]),
        .requant_en(gemm_requant),
        .sram_rd_addr(gemm_rd_addr),
        .sram_rd_data(gemm_rd_data),
        .sram_rd_en(gemm_rd_en),
        .sram_wr_addr(gemm_wr_addr),
        .sram_wr_data(gemm_wr_data),
        .sram_wr_en(gemm_wr_en),
        .array_load_weights(),
        .array_weight_row(),
        .array_weight_in()
    );
    
    dma_engine #(.DATA_WIDTH(DATA_WIDTH)) dma (
        .clk(clk),
        .rst_n(rst_n),
        .start(dma_start),
        .busy(dma_busy),
        .done(dma_done),
        .direction(dma_direction),
        .ddr_addr('0), // TODO: from instr
        .sram_addr('0),
        .byte_count(dma_byte_count),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .sram_addr_out(dma_rd_addr), // Shared rd/wr port
        .sram_wdata(dma_wr_data),
        .sram_we(dma_wr_en),
        .sram_re(dma_rd_en),
        .sram_rdata(dma_rd_data)
    );
    
    // Placeholders for other engines until fully implemented
    assign softmax_busy = 1'b0;
    assign softmax_done = 1'b0;
    assign layernorm_busy = 1'b0;
    assign layernorm_done = 1'b0;
    assign gelu_busy = 1'b0;
    assign gelu_done = 1'b0;
    assign vec_busy = 1'b0;
    assign vec_done = 1'b0;
    
endmodule

// Register module implementation
module npu_regs (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [31:0] s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,
    output logic [31:0] ctrl_reg,
    output logic [31:0] status_reg,
    output logic [31:0] ucode_base_reg,
    output logic [31:0] ucode_len_reg,
    output logic [31:0] ddr_base_wgt_reg,
    output logic [31:0] exec_mode_reg,
    input  logic        busy,
    input  logic        done
);
    // Simple register implementation
    logic [31:0] regs [0:15];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<16; i++) regs[i] <= '0;
        end else if (s_axi_awvalid && s_axi_wvalid && s_axi_awready && s_axi_wready) begin
            regs[s_axi_awaddr[5:2]] <= s_axi_wdata;
        end
    end
    
    assign ctrl_reg = regs[0];
    assign ucode_base_reg = regs[2];
    assign ucode_len_reg = regs[3];
    assign ddr_base_wgt_reg = regs[5];
    assign exec_mode_reg = regs[14];
    
    assign status_reg = {30'd0, done, busy};
    
    assign s_axi_awready = 1'b1;
    assign s_axi_wready = 1'b1;
    assign s_axi_bresp = 2'b00;
    assign s_axi_bvalid = 1'b1;
    assign s_axi_arready = 1'b1;
    assign s_axi_rdata = (s_axi_araddr[5:2] == 1) ? status_reg : regs[s_axi_araddr[5:2]];
    assign s_axi_rresp = 2'b00;
    assign s_axi_rvalid = 1'b1;

endmodule
