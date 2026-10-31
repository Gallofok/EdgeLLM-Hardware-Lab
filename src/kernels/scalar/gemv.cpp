#include "kernels/gemv.h"

#include "kernels/detail/gemv_validation.h"

namespace edgellm {

void gemv_scalar(const float* weights, const float* x, float* y,
                 std::size_t rows, std::size_t cols, std::size_t row_stride) {
  if (rows == 0) return;
  detail::validate_gemv(weights, x, y, rows, cols, row_stride);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    if (cols != 0) {
      const float* w = weights + row * row_stride;
      for (std::size_t col = 0; col < cols; ++col) sum += w[col] * x[col];
    }
    y[row] = sum;
  }
}

}  // namespace edgellm
