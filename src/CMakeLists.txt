# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(llama_cpp_dart_library VERSION 0.0.1 LANGUAGES C)

add_library(llama_cpp_dart SHARED
  "llama_cpp_dart.c"
)

set_target_properties(llama_cpp_dart PROPERTIES
  PUBLIC_HEADER llama_cpp_dart.h
  OUTPUT_NAME "llama_cpp_dart"
)

target_compile_definitions(llama_cpp_dart PUBLIC DART_SHARED_LIB)
