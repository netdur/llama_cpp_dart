cmake_minimum_required(VERSION 3.10)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    # Log the flags
    message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
endif()

#set(LLAMA_STATIC ON CACHE BOOL "llama: enable static link libraries" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

add_subdirectory(./llama.cpp)