cmake_minimum_required(VERSION 3.16)
project(EdgeLLMHardwareLab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Phase 2 extension point. No custom kernel or speedup is claimed in Phase 1.
add_library(edgellm_kernel_api INTERFACE)
target_include_directories(edgellm_kernel_api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
