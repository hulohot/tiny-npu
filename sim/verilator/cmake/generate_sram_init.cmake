# Helper script to generate SRAM initialization files
# Ensures directory exists before writing
# Parameters: OUTPUT_DIR, FILE_NAME (sram0_init.hex or sram1_init.hex), FILE_SIZE (number of hex values)

set(OUTPUT_FILE "${OUTPUT_DIR}/${FILE_NAME}")

# Ensure the output directory exists
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# Generate FILE_SIZE lines of "00" hex values
file(WRITE "${OUTPUT_FILE}" "")
foreach(i RANGE 1 ${FILE_SIZE})
    file(APPEND "${OUTPUT_FILE}" "00\n")
endforeach()

message(STATUS "Generated ${OUTPUT_FILE} with ${FILE_SIZE} bytes")
