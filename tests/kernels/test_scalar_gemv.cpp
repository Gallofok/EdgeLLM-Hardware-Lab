#include "kernels/gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Expected std::invalid_argument");
}

void oracle_case(std::size_t rows, std::size_t cols, std::size_t padding) {
  const auto stride = cols + padding;
  // Offset storage by one float to exercise addresses without SIMD alignment.
  std::vector<float> w(rows * stride + 1, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> x(cols + 1), y(rows + 2, -12345.0F);
  std::uint32_t state = 42;
  const auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<int>((state >> 16) % 2049) - 1024) / 256.0F;
  };
  for (std::size_t col = 0; col < cols; ++col) x[col + 1] = next();
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col) w[1 + row * stride + col] = next();
  edgellm::gemv_scalar(w.data() + 1, x.data() + 1, y.data() + 1, rows, cols, stride);
  const long double u = std::numeric_limits<float>::epsilon();
  const long double gamma = 2 * cols * u / (1 - 2 * cols * u);
  for (std::size_t row = 0; row < rows; ++row) {
    long double expected = 0, magnitude = 0;
    for (std::size_t col = 0; col < cols; ++col) {
      const auto term = static_cast<long double>(w[1 + row * stride + col]) * x[col + 1];
      expected += term;
      magnitude += std::abs(term);
    }
    require(std::isfinite(y[row + 1]), "Non-finite result (possibly read padding)");
    require(std::abs(y[row + 1] - expected) <= 1e-6L + gamma * magnitude,
            "Result differs from independent long-double dot-product oracle");
  }
  require(y.front() == -12345.0F && y.back() == -12345.0F, "Output overrun");
}

}  // namespace

int main() {
  try {
    const float w[] = {1, -2, 3, 4, 5, -6};
    const float x[] = {2, -1, 3};
    float y[] = {999, 999};
    edgellm::gemv_scalar(w, x, y, 2, 3, 3);
    require(y[0] == 13 && y[1] == -15, "Hand-calculated rectangular GEMV failed");
    const float cancelling[] = {1000000, 1, -1000000, -2};
    const float ones[] = {1, 1, 1, 1};
    edgellm::gemv_scalar(cancelling, ones, y, 1, 4, 4);
    require(y[0] == -1, "Cancellation case failed");
    for (const auto cols : {1U, 3U, 4U, 7U, 16U, 17U, 31U, 33U, 127U, 512U, 2048U}) {
      oracle_case(1, cols, 0);
      oracle_case(7, cols, 3);
    }
    edgellm::gemv_scalar(nullptr, nullptr, nullptr, 0, 5, 5);
    edgellm::gemv_scalar(nullptr, nullptr, y, 2, 0, 0);
    require(y[0] == 0 && y[1] == 0, "Zero-column output must be overwritten");
    y[0] = 7;
    rejects([&]() { edgellm::gemv_scalar(w, x, y, 1, 3, 2); });
    rejects([&]() { edgellm::gemv_scalar(nullptr, x, y, 1, 3, 3); });
    rejects([&]() { edgellm::gemv_scalar(w, nullptr, y, 1, 3, 3); });
    rejects([&]() { edgellm::gemv_scalar(w, x, nullptr, 1, 3, 3); });
    rejects([&]() {
      edgellm::gemv_scalar(w, x, y, 2, 3, std::numeric_limits<std::size_t>::max());
    });
    rejects([&]() {
      edgellm::gemv_scalar(w, x, y, std::numeric_limits<std::size_t>::max(), 0, 0);
    });
    require(y[0] == 7, "Invalid arguments modified the output");
    std::cout << "Scalar GEMV correctness: passed (independent oracle, tails, stride, bounds)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
