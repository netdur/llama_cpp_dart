cmake_minimum_required(VERSION 3.22.1)

project(LlamaCppAAR)

# --- Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags (Crucial: -fno-finite-math-only fixes the CPU build error)
add_compile_options(-O3 -ffast-math -fno-finite-math-only)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan Backend" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)

# --- 1. Point to your downloaded Vulkan Headers ---
# You listed "Vulkan-Headers-main" in your src directory
set(VULKAN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers-main/include")

if(NOT EXISTS "${VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp")
    message(FATAL_ERROR "Vulkan headers not found at ${VULKAN_INCLUDE_DIR}. Please check that the 'include' folder exists inside Vulkan-Headers-main.")
endif()

# --- Add llama.cpp Subdirectory ---
add_subdirectory(llama.cpp)

# --- 2. Inject Vulkan Headers ---
if(TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE "${VULKAN_INCLUDE_DIR}")
endif()

# --- Android Specific Logic ---
if (ANDROID)
    message(STATUS "Configuring for Android...")

    find_library(log-lib log)
    find_library(vulkan-lib vulkan)

    add_library(mtmd SHARED
        llama.cpp/tools/mtmd/mtmd.cpp
        llama.cpp/tools/mtmd/mtmd-audio.cpp
        llama.cpp/tools/mtmd/clip.cpp
        llama.cpp/tools/mtmd/mtmd-helper.cpp
    )

    target_include_directories(mtmd PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd"
        # 3. Add root src path so C++ can find "miniaudio/miniaudio.h"
        "${CMAKE_CURRENT_SOURCE_DIR}" 
    )

    target_link_libraries(mtmd 
        PUBLIC 
        llama 
        ggml
        ${log-lib}
        ${vulkan-lib} 
    )

    find_package(Threads REQUIRED)
    target_link_libraries(mtmd PRIVATE Threads::Threads)
    
endif()