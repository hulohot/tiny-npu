#include <iostream>
#include <vector>
#include <iomanip>
#include <verilated.h>
#include "Vnpu_block.h"
#include "common/npu_utils.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // Create microcode
    std::vector<Instruction> ucode;
    
    // 1. NOP
    ucode.push_back({OP_NOP, 0, 0, 0, 0, 0, 0, 0, 0});
    // 2. END
    ucode.push_back({OP_END, 0, 0, 0, 0, 0, 0, 0, 0});
    
    // Write microcode to hex file for SRAM initialization
    std::ofstream hex_file("sram0_init.hex");
    uint32_t base_addr = 0xF600; // 63KB offset
    
    // Fill 0s up to base_addr
    for (uint32_t i=0; i<base_addr; i++) hex_file << "00\n";
    
    // Write instructions
    for (const auto& instr : ucode) {
        uint8_t buffer[16];
        instr.pack(buffer);
        for (int b=0; b<16; b++) {
            hex_file << std::hex << std::setw(2) << std::setfill('0') << (int)buffer[b] << "\n";
        }
    }
    hex_file.close();
    
    std::cout << "Generated sram0_init.hex with " << ucode.size() << " instructions." << std::endl;

    Vnpu_block* top = new Vnpu_block;

    std::cout << "=== GPT-2 Block Test ===" << std::endl;

    // Initialize
    top->clk = 0;
    top->rst_n = 0;
    top->eval();

    // Reset
    for (int i = 0; i < 10; i++) {
        top->clk = !top->clk;
        top->eval();
    }
    top->rst_n = 1;
    top->clk = !top->clk;
    top->eval();
    
    // Start NPU
    // Write registers via AXI
    top->s_axi_awvalid = 1;
    top->s_axi_awaddr = 0x08; // UCODE_BASE
    top->s_axi_wvalid = 1;
    top->s_axi_wdata = 0xF600;
    top->s_axi_wstrb = 0xF;
    top->s_axi_bready = 1;
    top->clk = !top->clk; top->eval(); top->clk = !top->clk; top->eval();
    
    top->s_axi_awaddr = 0x0C; // UCODE_LEN
    top->s_axi_wdata = ucode.size();
    top->clk = !top->clk; top->eval(); top->clk = !top->clk; top->eval();
    
    top->s_axi_awaddr = 0x00; // CTRL (Start)
    top->s_axi_wdata = 0x01;
    top->clk = !top->clk; top->eval(); top->clk = !top->clk; top->eval();
    
    top->s_axi_awvalid = 0;
    top->s_axi_wvalid = 0;
    
    // Run until done
    int cycles = 0;
    while (!top->done && cycles < 1000) {
        top->clk = !top->clk; top->eval();
        top->clk = !top->clk; top->eval();
        cycles++;
    }
    
    if (top->done) {
        std::cout << "PASS: NPU finished execution in " << cycles << " cycles." << std::endl;
    } else {
        std::cout << "FAIL: Timeout waiting for NPU done." << std::endl;
    }
    
    top->final();
    delete top;
    return 0;
}
