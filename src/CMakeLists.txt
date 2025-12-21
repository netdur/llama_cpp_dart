cmake_minimum_required(VERSION 3.22.1)

project(llama_cpp_dart_wrapper)

# 1. Setup OpenCL paths BEFORE adding llama.cpp
set(OPENCL_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-Headers")

if (ANDROID)
    message(STATUS "Configuring for Android (OpenCL)...")
    message(STATUS "CMAKE_ANDROID_ARCH_ABI = ${CMAKE_ANDROID_ARCH_ABI}")

    # Define the path to the prebuilt library
    set(OPENCL_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/opencl-libs/android/${CMAKE_ANDROID_ARCH_ABI}/libOpenCL.so")

    # Create the imported target for your wrapper to use later
    add_library(opencl-lib SHARED IMPORTED)
    set_target_properties(opencl-lib PROPERTIES IMPORTED_LOCATION "${OPENCL_LIB_PATH}")
    
    # CRITICAL: Force llama.cpp (and ggml) to use these paths instead of searching system paths
    set(OpenCL_FOUND TRUE CACHE BOOL "OpenCL found" FORCE)
    set(OpenCL_INCLUDE_DIR "${OPENCL_HEADERS_DIR}" CACHE PATH "OpenCL include dir" FORCE)
    set(OpenCL_LIBRARY "${OPENCL_LIB_PATH}" CACHE FILEPATH "OpenCL library" FORCE)
endif()

# 2. Add llama.cpp (This builds llama, ggml, and applies the -DGGML_OPENCL flags)
add_subdirectory(llama.cpp)

# 3. Define your wrapper library (mtmd)
if (ANDROID)
    find_library(log-lib log)

    add_library(mtmd SHARED
        llama.cpp/tools/mtmd/mtmd.cpp
        llama.cpp/tools/mtmd/mtmd-audio.cpp
        llama.cpp/tools/mtmd/clip.cpp
        llama.cpp/tools/mtmd/mtmd-helper.cpp
    )

    # Include directories
    target_include_directories(mtmd PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${OPENCL_HEADERS_DIR}"
    )

    # Link libraries
    target_link_libraries(mtmd 
        PUBLIC 
            llama 
            ggml
            ${log-lib}
            opencl-lib 
    )

    find_package(Threads REQUIRED)
    target_link_libraries(mtmd PRIVATE Threads::Threads)
endif()