cmake_minimum_required(VERSION 3.16)
project(EdgeLLMHardwareLab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(edgellm_kernel_api INTERFACE)
target_include_directories(edgellm_kernel_api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(edgellm_scalar STATIC src/kernels/scalar/gemv.cpp)
target_link_libraries(edgellm_scalar PUBLIC edgellm_kernel_api)
# A deliberately scalar FP32 baseline. Keep contraction and automatic SIMD off;
# the later NEON implementation will be a separate target with separate flags.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(edgellm_scalar PRIVATE -fno-fast-math -fno-tree-vectorize -ffp-contract=off)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(edgellm_scalar PRIVATE -fno-fast-math -fno-vectorize -fno-slp-vectorize -ffp-contract=off)
else()
  message(FATAL_ERROR "The scalar baseline requires GCC or Clang for explicit SIMD/FP controls")
endif()
set_property(TARGET edgellm_scalar PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_options(edgellm_scalar PRIVATE -Wall -Wextra -Wpedantic -Werror)

option(EDGELLM_ENABLE_NEON "Build the ARM64 NEON kernel when the target supports it" ON)
set(EDGELLM_HAS_NEON OFF)
if(EDGELLM_ENABLE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles("
    #include <arm_neon.h>
    #if !defined(__aarch64__) || !defined(__ARM_NEON)
    #error ARM64 NEON required
    #endif
    int main() { return static_cast<int>(vaddvq_f32(vdupq_n_f32(1))); }
  " EDGELLM_COMPILER_HAS_NEON)
  if(NOT EDGELLM_COMPILER_HAS_NEON)
    message(FATAL_ERROR "ARM64 target cannot compile NEON intrinsics")
  endif()
  set(EDGELLM_HAS_NEON ON)
  add_library(edgellm_neon STATIC src/kernels/neon/gemv.cpp)
  target_link_libraries(edgellm_neon PUBLIC edgellm_kernel_api)
  target_compile_definitions(edgellm_neon PUBLIC EDGELLM_HAS_NEON=1)
  target_compile_options(edgellm_neon PRIVATE -fno-fast-math -ffp-contract=off
    -Wall -Wextra -Wpedantic -Werror)
  set_property(TARGET edgellm_neon PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()
message(STATUS "ARM64 NEON kernel enabled: ${EDGELLM_HAS_NEON}")

add_executable(gemv-bench benchmarks/gemv_bench.cpp)
target_link_libraries(gemv-bench PRIVATE edgellm_scalar)
if(EDGELLM_HAS_NEON)
  target_link_libraries(gemv-bench PRIVATE edgellm_neon)
endif()
target_compile_options(gemv-bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_property(TARGET gemv-bench PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

include(CTest)
if(BUILD_TESTING)
  add_executable(test-scalar-gemv tests/kernels/test_scalar_gemv.cpp)
  target_link_libraries(test-scalar-gemv PRIVATE edgellm_scalar)
  target_compile_options(test-scalar-gemv PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME scalar-gemv-correctness COMMAND test-scalar-gemv)
  add_test(NAME scalar-gemv-benchmark-smoke
    COMMAND gemv-bench --rows 7 --cols 17 --repetitions 2 --min-ms 1)
  add_test(NAME scalar-gemv-reject-invalid COMMAND gemv-bench --rows -1)
  set_tests_properties(scalar-gemv-reject-invalid PROPERTIES WILL_FAIL TRUE)
  if(EDGELLM_HAS_NEON)
    add_executable(test-neon-gemv tests/kernels/test_neon_gemv.cpp)
    target_link_libraries(test-neon-gemv PRIVATE edgellm_neon edgellm_scalar)
    target_compile_options(test-neon-gemv PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME neon-gemv-correctness COMMAND test-neon-gemv)
    add_test(NAME neon-gemv-paired-smoke
      COMMAND gemv-bench --implementation paired --rows 7 --cols 17 --repetitions 2 --min-ms 1)
    add_test(NAME neon-gemv-reject-unbalanced
      COMMAND gemv-bench --implementation paired --repetitions 3)
    set_tests_properties(neon-gemv-reject-unbalanced PROPERTIES WILL_FAIL TRUE)
  else()
    add_test(NAME neon-unavailable-is-explicit COMMAND gemv-bench --implementation neon)
    set_tests_properties(neon-unavailable-is-explicit PROPERTIES WILL_FAIL TRUE)
  endif()
endif()
