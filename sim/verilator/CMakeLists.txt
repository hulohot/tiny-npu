cmake_minimum_required(VERSION 3.14)
project(tiny-npu)

# Find Verilator
find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
if(NOT verilator_FOUND)
    message(FATAL_ERROR "Verilator not found. Install with: sudo apt-get install verilator")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../rtl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/testbenches)

# Testbenches
set(TESTBENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testbenches)
set(RTL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rtl)
set(GEMM_DIR ${RTL_DIR}/gemm)

# =============================================================================
# UNIT TESTS - Individual modules
# =============================================================================

# MAC Unit Test
add_executable(test_mac_unit
    ${TESTBENCH_DIR}/mac_unit_tb.cpp
)
verilate(test_mac_unit
    SOURCES ${GEMM_DIR}/mac_unit.sv
    TOP_MODULE mac_unit
    VERILATOR_ARGS --trace
)

# =============================================================================
# ENGINE TESTS - Full engines
# =============================================================================

# Systolic Array Test
add_executable(test_systolic_array
    ${TESTBENCH_DIR}/systolic_array_tb.cpp
)
verilate(test_systolic_array
    SOURCES ${GEMM_DIR}/systolic_array.sv ${GEMM_DIR}/mac_unit.sv
    TOP_MODULE systolic_array
    VERILATOR_ARGS --trace
)

# =============================================================================
# TOP-LEVEL TESTS (placeholders for now)
# =============================================================================

# RTL sources for full NPU
set(RTL_SOURCES
    ${RTL_DIR}/npu_top.sv
    ${GEMM_DIR}/mac_unit.sv
    ${GEMM_DIR}/systolic_array.sv
)

# NPU smoke test
add_executable(npu_sim
    ${TESTBENCH_DIR}/npu_tb.cpp
)
verilate(npu_sim
    SOURCES ${RTL_SOURCES}
    TOP_MODULE npu_top
    VERILATOR_ARGS --trace
)

# Engine tests placeholder
add_executable(engine_sim
    ${TESTBENCH_DIR}/engine_tb.cpp
)

# Integration test placeholder
add_executable(integration_sim
    ${TESTBENCH_DIR}/integration_tb.cpp
)

# GPT-2 block test placeholder
add_executable(gpt2_block_sim
    ${TESTBENCH_DIR}/gpt2_block_tb.cpp
)

# Demo inference placeholder
add_executable(demo_infer
    ${TESTBENCH_DIR}/demo_infer.cpp
)

# =============================================================================
# Testing
# =============================================================================

enable_testing()
add_test(NAME MAC_Unit COMMAND test_mac_unit)
add_test(NAME Systolic_Array COMMAND test_systolic_array)
add_test(NAME NPU_Smoke COMMAND npu_sim)

# Print status
message(STATUS "Tiny NPU build configuration:")
message(STATUS "  Verilator: ${VERILATOR_ROOT}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Tests:")
message(STATUS "  test_mac_unit       - MAC unit tests")
message(STATUS "  test_systolic_array - Systolic array tests")
message(STATUS "  npu_sim             - Top-level smoke test")
