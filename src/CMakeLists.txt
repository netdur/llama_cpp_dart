cmake_minimum_required(VERSION 3.18)

project(LlamaCppAAR)

# --- Configuration for the llama.cpp sub-project ---
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "Disable -march=native for cross-compilation" FORCE)
# set(LLAMA_BLAS OFF CACHE BOOL "Disable BLAS")
# set(LLAMA_AVX2 OFF CACHE BOOL "Disable AVX2")

set(LLAMA_BUILD_TOOLS ON CACHE BOOL "Build tools" FORCE)

# --- End of Configuration ---
add_subdirectory(./llama.cpp)

if (ANDROID)
    message(STATUS "Android platform detected. Explicitly adding 'mtmd' shared library.")
    set (_ggml_vk_genshaders_cmd llama.cpp/build/bin/vulkan-shaders-gen)

    add_library(mtmd SHARED
        llama.cpp/tools/mtmd/mtmd.cpp
        llama.cpp/tools/mtmd/mtmd-audio.cpp
        llama.cpp/tools/mtmd/clip.cpp
        llama.cpp/tools/mtmd/mtmd-helper.cpp
    )

    target_include_directories(mtmd PUBLIC
        $<TARGET_PROPERTY:llama,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/examples/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor"
    )

    target_link_libraries(mtmd PUBLIC llama ggml)
    find_package(Threads REQUIRED)
    target_link_libraries(mtmd PRIVATE Threads::Threads)
endif()

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

if (ANDROID)
    install(
        FILES
            llama.cpp/tools/mtmd/mtmd.h
            llama.cpp/tools/mtmd/mtmd-helper.h
            llama.cpp/tools/mtmd/clip.h
        DESTINATION include
    )
endif()