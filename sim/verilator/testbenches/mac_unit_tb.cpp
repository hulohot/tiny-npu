// Testbench for MAC Unit
// Verilator-compatible C++ testbench

#include <iostream>
#include <cassert>
#include <cstdint>
#include <verilated.h>
#include "Vmac_unit.h"

// Helper function for signed 8-bit multiplication with 32-bit accumulation
int32_t golden_mac(int8_t a, int8_t w, int32_t partial) {
    return partial + (int32_t)a * (int32_t)w;
}

// Advance one clock cycle
void tick(Vmac_unit* mac) {
    mac->clk = !mac->clk;
    mac->eval();
    mac->clk = !mac->clk;
    mac->eval();
}

void test_mac_basic() {
    std::cout << "Test: MAC basic operation..." << std::endl;
    
    Vmac_unit* mac = new Vmac_unit;
    
    // Initialize
    mac->clk = 0;
    mac->rst_n = 0;
    mac->en = 0;
    mac->clr = 0;
    mac->load_weight = 0;
    mac->activation_in = 0;
    mac->weight_in = 0;
    mac->partial_sum_in = 0;
    
    // Reset (5 cycles)
    for (int i = 0; i < 5; i++) tick(mac);
    mac->rst_n = 1;
    tick(mac);
    
    // Load weight (takes effect next cycle, output 1 cycle after)
    mac->load_weight = 1;
    mac->weight_in = 5;
    tick(mac);
    mac->load_weight = 0;
    tick(mac);  // weight loaded into reg
    
    // Setup inputs
    mac->en = 1;
    mac->activation_in = 3;
    mac->partial_sum_in = 10;
    
    // Pipeline: mult -> accum -> output_reg
    tick(mac);  // mult happens
    tick(mac);  // accum happens, partial_sum_out updates
    tick(mac);  // output register updates
    
    int32_t expected = golden_mac(3, 5, 10);  // 10 + 3*5 = 25
    
    std::cout << "  activation=3, weight=5, partial=10" << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Got: " << (int32_t)mac->partial_sum_out << std::endl;
    
    assert(mac->partial_sum_out == expected);
    std::cout << "  PASSED" << std::endl;
    
    mac->final();
    delete mac;
}

void test_mac_clear() {
    std::cout << "Test: MAC clear operation..." << std::endl;
    
    Vmac_unit* mac = new Vmac_unit;
    
    // Initialize and reset
    mac->clk = 0;
    mac->rst_n = 0;
    for (int i = 0; i < 5; i++) tick(mac);
    mac->rst_n = 1;
    
    // Load weight
    mac->load_weight = 1;
    mac->weight_in = 10;
    tick(mac);
    mac->load_weight = 0;
    tick(mac);
    
    // Accumulate
    mac->en = 1;
    mac->activation_in = 2;
    mac->partial_sum_in = 0;
    tick(mac);
    tick(mac);
    tick(mac);
    // Result should be 20
    
    // Clear
    mac->clr = 1;
    mac->en = 0;
    mac->partial_sum_in = 100;  // Should be ignored
    tick(mac);
    mac->clr = 0;
    tick(mac);  // Output register updates
    
    std::cout << "  After clear, partial_sum should be 0" << std::endl;
    std::cout << "  Got: " << (int32_t)mac->partial_sum_out << std::endl;
    
    assert(mac->partial_sum_out == 0);
    std::cout << "  PASSED" << std::endl;
    
    mac->final();
    delete mac;
}

void test_mac_negative() {
    std::cout << "Test: MAC with negative values..." << std::endl;
    
    Vmac_unit* mac = new Vmac_unit;
    
    // Initialize and reset
    mac->clk = 0;
    mac->rst_n = 0;
    for (int i = 0; i < 5; i++) tick(mac);
    mac->rst_n = 1;
    
    // Load negative weight (-5)
    mac->load_weight = 1;
    mac->weight_in = (uint8_t)(-5);  // 0xFB = -5
    tick(mac);
    mac->load_weight = 0;
    tick(mac);
    
    // Multiply
    mac->en = 1;
    mac->activation_in = 3;
    mac->partial_sum_in = 10;
    tick(mac);
    tick(mac);
    tick(mac);
    
    int32_t expected = golden_mac(3, -5, 10);  // 10 + 3*(-5) = -5
    
    std::cout << "  activation=3, weight=-5, partial=10" << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Got: " << (int32_t)mac->partial_sum_out << std::endl;
    
    assert(mac->partial_sum_out == expected);
    std::cout << "  PASSED" << std::endl;
    
    mac->final();
    delete mac;
}

void test_mac_pipeline() {
    std::cout << "Test: MAC pipeline with multiple values..." << std::endl;
    
    Vmac_unit* mac = new Vmac_unit;
    
    // Initialize and reset
    mac->clk = 0;
    mac->rst_n = 0;
    for (int i = 0; i < 5; i++) tick(mac);
    mac->rst_n = 1;
    tick(mac);
    
    // Load weight
    mac->load_weight = 1;
    mac->weight_in = 2;
    tick(mac);
    mac->load_weight = 0;
    tick(mac);
    
    // Stream multiple activations through
    int8_t activations[] = {1, 2, 3, 4, 5};
    int32_t partial = 0;
    int32_t expected_results[5];
    
    // First input
    mac->en = 1;
    mac->activation_in = activations[0];
    mac->partial_sum_in = partial;
    tick(mac);
    partial = golden_mac(activations[0], 2, partial);
    expected_results[0] = partial;
    
    // Continue with remaining inputs
    for (int i = 1; i < 5; i++) {
        mac->activation_in = activations[i];
        mac->partial_sum_in = partial;
        tick(mac);
        partial = golden_mac(activations[i], 2, partial);
        expected_results[i] = partial;
    }
    
    // Flush pipeline (2 more cycles for output)
    mac->en = 0;
    tick(mac);
    tick(mac);
    
    std::cout << "  Final accumulator: " << (int32_t)mac->partial_sum_out << std::endl;
    std::cout << "  Expected: 30" << std::endl;
    
    // Expected: 0 + 2*(1+2+3+4+5) = 30
    assert(mac->partial_sum_out == 30);
    std::cout << "  PASSED" << std::endl;
    
    mac->final();
    delete mac;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "      MAC Unit Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    
    Verilated::commandArgs(argc, argv);
    
    try {
        test_mac_basic();
        test_mac_clear();
        test_mac_negative();
        test_mac_pipeline();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "    ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
