#pragma once

#include <cstddef>

namespace edgellm {

// FP32 y[row] = sum(W[row * row_stride + col] * x[col]).
// W is row-major; row_stride is measured in float elements and must be >= cols.
// Input/output storage must be valid for the specified extents; y must not alias
// W or x. No alignment beyond alignof(float) is required. Finite, non-overflowing
// arithmetic is the numerical contract. No allocation or threading occurs here.
// rows == 0 is a no-op; cols == 0 writes zeros without reading W or x.
// Invalid pointer/stride/extent arguments throw std::invalid_argument before
// writing y. The caller remains responsible for actual allocation sizes.
void gemv_scalar(const float* weights, const float* x, float* y,
                 std::size_t rows, std::size_t cols, std::size_t row_stride);

#ifdef EDGELLM_HAS_NEON
// ARM64 NEON implementation with the same storage/argument contract. Vector
// reduction changes summation order; bitwise equality with scalar is not promised.
void gemv_neon(const float* weights, const float* x, float* y,
               std::size_t rows, std::size_t cols, std::size_t row_stride);
#endif

}  // namespace edgellm
