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

# RTL sources
set(RTL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../rtl/npu_top.sv
)

# Testbenches
set(TESTBENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testbenches)

# Basic smoke test
add_executable(npu_sim
    ${TESTBENCH_DIR}/npu_tb.cpp
)
verilate(npu_sim
    SOURCES ${RTL_SOURCES}
    TOP_MODULE npu_top
    VERILATOR_ARGS --trace
)

# Engine tests
add_executable(engine_sim
    ${TESTBENCH_DIR}/engine_tb.cpp
)
verilate(engine_sim
    SOURCES ${RTL_SOURCES}
    TOP_MODULE npu_top
)

# Integration test
add_executable(integration_sim
    ${TESTBENCH_DIR}/integration_tb.cpp
)
verilate(integration_sim
    SOURCES ${RTL_SOURCES}
    TOP_MODULE npu_top
)

# GPT-2 block test
add_executable(gpt2_block_sim
    ${TESTBENCH_DIR}/gpt2_block_tb.cpp
)
verilate(gpt2_block_sim
    SOURCES ${RTL_SOURCES}
    TOP_MODULE npu_top
)

# Demo inference
add_executable(demo_infer
    ${TESTBENCH_DIR}/demo_infer.cpp
)
verilate(demo_infer
    SOURCES ${RTL_SOURCES}
    TOP_MODULE npu_top
)

# Print status
message(STATUS "Tiny NPU build configuration:")
message(STATUS "  Verilator: ${VERILATOR_ROOT}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
