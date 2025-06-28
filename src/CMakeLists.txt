cmake_minimum_required(VERSION 3.18)

project(LlamaCppAAR)

# --- Configuration for the llama.cpp sub-project ---
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "Disable -march=native for cross-compilation" FORCE)
# set(LLAMA_BLAS OFF CACHE BOOL "Disable BLAS")
# set(LLAMA_AVX2 OFF CACHE BOOL "Disable AVX2")

# --- End of Configuration ---
add_subdirectory(./llama.cpp)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
