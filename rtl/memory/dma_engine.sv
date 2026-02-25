// DMA Engine
// Transfers data between external DDR and on-chip SRAM
// Supports burst transfers for efficient weight loading

`timescale 1ns/1ps

module dma_engine #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 32,     // DDR address width
    parameter SRAM_ADDR_WIDTH = 16,
    parameter BURST_LEN = 16       // AXI burst length
)(
    input  logic                      clk,
    input  logic                      rst_n,
    
    // Control interface
    input  logic                      start,
    output logic                      busy,
    output logic                      done,
    input  logic                      direction,     // 0: DDR→SRAM, 1: SRAM→DDR
    input  logic [ADDR_WIDTH-1:0]     ddr_addr,
    input  logic [SRAM_ADDR_WIDTH-1:0] sram_addr,
    input  logic [31:0]               byte_count,    // Total bytes to transfer
    
    // AXI4 Read Address Channel
    output logic [ADDR_WIDTH-1:0]     m_axi_araddr,
    output logic [7:0]                m_axi_arlen,
    output logic [2:0]                m_axi_arsize,  // 3'b000 = 1 byte, 3'b011 = 8 bytes
    output logic [1:0]                m_axi_arburst, // 2'b01 = INCR
    output logic                      m_axi_arvalid,
    input  logic                      m_axi_arready,
    
    // AXI4 Read Data Channel
    input  logic [63:0]               m_axi_rdata,
    input  logic [1:0]                m_axi_rresp,
    input  logic                      m_axi_rlast,
    input  logic                      m_axi_rvalid,
    output logic                      m_axi_rready,
    
    // AXI4 Write Address Channel
    output logic [ADDR_WIDTH-1:0]     m_axi_awaddr,
    output logic [7:0]                m_axi_awlen,
    output logic [2:0]                m_axi_awsize,
    output logic [1:0]                m_axi_awburst,
    output logic                      m_axi_awvalid,
    input  logic                      m_axi_awready,
    
    // AXI4 Write Data Channel
    output logic [63:0]               m_axi_wdata,
    output logic [7:0]                m_axi_wstrb,
    output logic                      m_axi_wlast,
    output logic                      m_axi_wvalid,
    input  logic                      m_axi_wready,
    
    // AXI4 Write Response Channel
    input  logic [1:0]                m_axi_bresp,
    input  logic                      m_axi_bvalid,
    output logic                      m_axi_bready,
    
    // SRAM interface
    output logic [SRAM_ADDR_WIDTH-1:0] sram_addr_out,
    output logic [DATA_WIDTH-1:0]      sram_wdata,
    output logic                       sram_we,
    output logic                       sram_re,
    input  logic [DATA_WIDTH-1:0]      sram_rdata
);

    // State machine
    typedef enum logic [3:0] {
        IDLE,
        
        // Read states (DDR → SRAM)
        RD_ADDR,
        RD_DATA,
        RD_WRITE_SRAM,
        
        // Write states (SRAM → DDR)
        WR_ADDR,
        WR_READ_SRAM,
        WR_DATA,
        WR_RESP,
        
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Transfer counters
    logic [31:0] bytes_remaining;
    logic [SRAM_ADDR_WIDTH-1:0] current_sram_addr;
    logic [ADDR_WIDTH-1:0] current_ddr_addr;
    
    // Burst counter
    logic [7:0] burst_count;
    logic [7:0] current_burst_len;
    
    // Read buffer
    logic [63:0] read_buffer;
    logic [2:0] read_buffer_valid_bytes;
    
    // Calculate burst length
    // Minimize bursts while respecting max burst length
    function automatic [7:0] calc_burst_len(input [31:0] remaining);
        if (remaining >= BURST_LEN * 8) begin  // 8 bytes per beat
            return BURST_LEN - 1;  // AXI len = beats - 1
        end else begin
            return 8'((remaining[31:3] - 1));  // Divide by 8, subtract 1
        end
    endfunction
    
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            bytes_remaining <= '0;
            current_sram_addr <= '0;
            current_ddr_addr <= '0;
            burst_count <= '0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        bytes_remaining <= byte_count;
                        current_sram_addr <= sram_addr;
                        current_ddr_addr <= ddr_addr;
                        burst_count <= '0;
                    end
                end
                
                RD_ADDR: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        current_burst_len <= calc_burst_len(bytes_remaining);
                    end
                end
                
                RD_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        read_buffer <= m_axi_rdata;
                        // For 64-bit data, we can extract bytes as needed
                    end
                end
                
                RD_WRITE_SRAM: begin
                    // Write one byte to SRAM per cycle
                    if (bytes_remaining > 0) begin
                        bytes_remaining <= bytes_remaining - 1;
                        current_sram_addr <= current_sram_addr + 1;
                    end
                end
                
                WR_READ_SRAM: begin
                    // Read one byte from SRAM per cycle
                    if (burst_count < current_burst_len) begin
                        burst_count <= burst_count + 1;
                    end
                end
                
                WR_DATA: begin
                    if (m_axi_wvalid && m_axi_wready) begin
                        if (m_axi_wlast) begin
                            bytes_remaining <= bytes_remaining - ((32'(current_burst_len) + 32'd1) * 32'd8);
                            current_ddr_addr <= current_ddr_addr + ((ADDR_WIDTH'(current_burst_len) + ADDR_WIDTH'(1)) * ADDR_WIDTH'(8));
                            current_sram_addr <= current_sram_addr + ((SRAM_ADDR_WIDTH'(current_burst_len) + SRAM_ADDR_WIDTH'(1)) * SRAM_ADDR_WIDTH'(8));
                        end
                    end
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
            IDLE: begin
                if (start) begin
                    if (direction == 1'b0) begin
                        next_state = RD_ADDR;
                    end else begin
                        next_state = WR_ADDR;
                    end
                end
            end
            
            RD_ADDR: begin
                if (m_axi_arvalid && m_axi_arready) begin
                    next_state = RD_DATA;
                end
            end
            
            RD_DATA: begin
                if (m_axi_rvalid && m_axi_rready) begin
                    if (m_axi_rlast) begin
                        next_state = RD_WRITE_SRAM;
                    end
                end
            end
            
            RD_WRITE_SRAM: begin
                if (bytes_remaining <= 1) begin
                    next_state = DONE_STATE;
                end else if (bytes_remaining % ((32'(current_burst_len) + 32'd1) * 32'd8) == 0) begin
                    // Need another burst
                    next_state = RD_ADDR;
                end
            end
            
            WR_ADDR: begin
                if (m_axi_awvalid && m_axi_awready) begin
                    next_state = WR_READ_SRAM;
                end
            end
            
            WR_READ_SRAM: begin
                if (burst_count >= current_burst_len) begin
                    next_state = WR_DATA;
                end
            end
            
            WR_DATA: begin
                if (m_axi_wvalid && m_axi_wready && m_axi_wlast) begin
                    next_state = WR_RESP;
                end
            end
            
            WR_RESP: begin
                if (m_axi_bvalid && m_axi_bready) begin
                    if (bytes_remaining <= ((32'(current_burst_len) + 32'd1) * 32'd8)) begin
                        next_state = DONE_STATE;
                    end else begin
                        next_state = WR_ADDR;
                    end
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
    
    // AXI Read Address Channel
    assign m_axi_araddr = current_ddr_addr;
    assign m_axi_arlen = calc_burst_len(bytes_remaining);
    assign m_axi_arsize = 3'b011;  // 8 bytes per beat
    assign m_axi_arburst = 2'b01;  // INCR
    assign m_axi_arvalid = (state == RD_ADDR);
    
    // AXI Read Data Channel
    assign m_axi_rready = (state == RD_DATA);
    
    // AXI Write Address Channel
    assign m_axi_awaddr = current_ddr_addr;
    assign m_axi_awlen = calc_burst_len(bytes_remaining);
    assign m_axi_awsize = 3'b011;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = (state == WR_ADDR);
    
    // AXI Write Data Channel
    assign m_axi_wdata = {56'd0, sram_rdata};  // Pad to 64 bits
    assign m_axi_wstrb = 8'hFF;  // All bytes valid
    assign m_axi_wlast = (burst_count >= current_burst_len);
    assign m_axi_wvalid = (state == WR_DATA);
    
    // AXI Write Response Channel
    assign m_axi_bready = (state == WR_RESP);
    
    // SRAM interface
    assign sram_addr_out = current_sram_addr;
    assign sram_wdata = read_buffer[7:0];  // Extract byte
    assign sram_we = (state == RD_WRITE_SRAM);
    assign sram_re = (state == WR_READ_SRAM);
    
    // Status
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule
