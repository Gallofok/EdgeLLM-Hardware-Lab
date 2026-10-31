#include "kernels/gemv.h"

#include "kernels/detail/gemv_validation.h"

#include <arm_neon.h>

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "This NEON implementation requires an ARM64 compiler with Advanced SIMD"
#endif

namespace edgellm {

void gemv_neon(const float* weights, const float* x, float* y,
               std::size_t rows, std::size_t cols, std::size_t row_stride) {
  if (rows == 0) return;
  detail::validate_gemv(weights, x, y, rows, cols, row_stride);
  for (std::size_t row = 0; row < rows; ++row) {
    if (cols == 0) {
      y[row] = 0;
      continue;
    }
    const float* w = weights + row * row_stride;
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    std::size_t col = 0;
    // Four independent accumulators reduce the dependency chain. Keep multiply
    // and add separate (contraction is disabled for this target).
    for (; cols - col >= 16; col += 16) {
      a = vaddq_f32(a, vmulq_f32(vld1q_f32(w + col), vld1q_f32(x + col)));
      b = vaddq_f32(b, vmulq_f32(vld1q_f32(w + col + 4), vld1q_f32(x + col + 4)));
      c = vaddq_f32(c, vmulq_f32(vld1q_f32(w + col + 8), vld1q_f32(x + col + 8)));
      d = vaddq_f32(d, vmulq_f32(vld1q_f32(w + col + 12), vld1q_f32(x + col + 12)));
    }
    for (; cols - col >= 4; col += 4)
      a = vaddq_f32(a, vmulq_f32(vld1q_f32(w + col), vld1q_f32(x + col)));
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; col < cols; ++col) sum += w[col] * x[col];
    y[row] = sum;
  }
}

}  // namespace edgellm
