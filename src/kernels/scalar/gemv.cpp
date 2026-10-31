#include "kernels/gemv.h"

#include <limits>
#include <stdexcept>

namespace edgellm {

void gemv_scalar(const float* weights, const float* x, float* y,
                 std::size_t rows, std::size_t cols, std::size_t row_stride) {
  if (rows == 0) return;
  constexpr auto max_elements =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
  if (y == nullptr || rows > max_elements || cols > max_elements || row_stride < cols) {
    throw std::invalid_argument("Invalid GEMV output, dimensions or row stride");
  }
  if (cols != 0 && (weights == nullptr || x == nullptr ||
                   (rows > 1 && row_stride > (max_elements - cols) / (rows - 1)))) {
    throw std::invalid_argument("Invalid GEMV input or overflowing matrix extent");
  }
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
