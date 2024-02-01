# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "llama_cpp_dart")
project(${PROJECT_NAME} LANGUAGES CXX)

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR lib)
add_subdirectory(../src src)