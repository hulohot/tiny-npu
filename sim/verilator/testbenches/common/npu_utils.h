#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>

struct Instruction {
    uint8_t opcode;
    uint8_t flags;
    uint16_t dst;
    uint16_t src0;
    uint16_t src1;
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint16_t imm;
    
    // Pack into 128-bit array (little endian)
    void pack(uint8_t* buffer) const {
        // [7:0] opcode
        buffer[0] = opcode;
        // [15:8] flags
        buffer[1] = flags;
        // [31:16] dst
        *(uint16_t*)(buffer + 2) = dst;
        // [47:32] src0
        *(uint16_t*)(buffer + 4) = src0;
        // [63:48] src1
        *(uint16_t*)(buffer + 6) = src1;
        // [79:64] m
        *(uint16_t*)(buffer + 8) = m;
        // [95:80] n
        *(uint16_t*)(buffer + 10) = n;
        // [111:96] k
        *(uint16_t*)(buffer + 12) = k;
        // [127:112] imm
        *(uint16_t*)(buffer + 14) = imm;
    }
};

// Opcodes
enum Opcode {
    OP_NOP       = 0x00,
    OP_DMA_LOAD  = 0x01,
    OP_DMA_STORE = 0x02,
    OP_GEMM      = 0x03,
    OP_VEC       = 0x04,
    OP_SOFTMAX   = 0x05,
    OP_LAYERNORM = 0x06,
    OP_GELU      = 0x07,
    OP_VEC_ADD   = 0x08,
    OP_VEC_MUL   = 0x09,
    OP_VEC_COPY  = 0x0A,
    OP_BARRIER   = 0xFE,
    OP_END       = 0xFF
};

// Helper to write instructions to binary file
inline void write_microcode(const std::string& filename, const std::vector<Instruction>& instrs) {
    std::ofstream file(filename, std::ios::binary);
    for (const auto& instr : instrs) {
        uint8_t buffer[16];
        instr.pack(buffer);
        file.write((char*)buffer, 16);
    }
    file.close();
}
