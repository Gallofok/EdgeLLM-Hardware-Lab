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
  try { function(); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "Invalid NEON arguments were not rejected");
}

void oracle_case(std::size_t cols, std::size_t padding, std::size_t offset) {
  constexpr std::size_t rows = 5;
  const auto stride = cols + padding;
  std::vector<float> w(offset + (rows - 1) * stride + cols,
                       std::numeric_limits<float>::quiet_NaN());
  std::vector<float> x(offset + cols), scalar(rows + 2, -12345), neon(rows + 2, -12345);
  std::uint32_t state = 17;
  const auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    // Non-binary fractions and varying magnitudes exercise actual rounding.
    return std::ldexp(static_cast<float>(static_cast<int>((state >> 8) % 2001) - 1000)
                          / 1000.0F, static_cast<int>(state % 9) - 4);
  };
  for (std::size_t col = 0; col < cols; ++col) x[offset + col] = next();
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col) w[offset + row * stride + col] = next();
  edgellm::gemv_scalar(w.data() + offset, x.data() + offset, scalar.data() + 1,
                       rows, cols, stride);
  edgellm::gemv_neon(w.data() + offset, x.data() + offset, neon.data() + 1,
                     rows, cols, stride);
  const long double u = std::numeric_limits<float>::epsilon();
  const long double gamma = 2 * cols * u / (1 - 2 * cols * u);
  for (std::size_t row = 0; row < rows; ++row) {
    long double expected = 0, magnitude = 0;
    for (std::size_t col = 0; col < cols; ++col) {
      const auto term = static_cast<long double>(w[offset + row * stride + col]) * x[offset + col];
      expected += term;
      magnitude += std::abs(term);
    }
    for (const auto value : {scalar[row + 1], neon[row + 1]}) {
      require(std::isfinite(value), "Non-finite result or padding read");
      require(std::abs(value - expected) <= 1e-6L + gamma * magnitude,
              "Kernel exceeds independent long-double forward-error bound");
    }
  }
  require(neon.front() == -12345 && neon.back() == -12345, "NEON output overrun");
}

}  // namespace

int main() {
  try {
    const float w[] = {1, -2, 3, 4, 5, -6}, x[] = {2, -1, 3};
    float y[] = {999, 999};
    edgellm::gemv_neon(w, x, y, 2, 3, 3);
    require(y[0] == 13 && y[1] == -15, "Hand-calculated NEON product failed");
    const float cancelling[] = {1000000, 1, -1000000, -2}, ones[] = {1, 1, 1, 1};
    edgellm::gemv_neon(cancelling, ones, y, 1, 4, 4);
    require(y[0] == -1, "Exact cancellation case failed");
    for (std::size_t cols = 1; cols <= 65; ++cols)
      for (const auto offset : {0U, 1U}) oracle_case(cols, 0, offset);
    for (const auto cols : {127U, 129U, 511U, 512U, 513U, 2048U, 2049U, 4095U})
      oracle_case(cols, 3, 1);
    edgellm::gemv_neon(nullptr, nullptr, nullptr, 0, 3, 3);
    edgellm::gemv_neon(nullptr, nullptr, y, 2, 0, 0);
    require(y[0] == 0 && y[1] == 0, "Zero-column NEON output failed");
    y[0] = 7;
    rejects([&]() { edgellm::gemv_neon(w, x, y, 1, 3, 2); });
    rejects([&]() { edgellm::gemv_neon(nullptr, x, y, 1, 3, 3); });
    rejects([&]() { edgellm::gemv_neon(w, nullptr, y, 1, 3, 3); });
    rejects([&]() { edgellm::gemv_neon(w, x, nullptr, 1, 3, 3); });
    rejects([&]() {
      edgellm::gemv_neon(w, x, y, 2, 3, std::numeric_limits<std::size_t>::max());
    });
    rejects([&]() {
      edgellm::gemv_neon(w, x, y, std::numeric_limits<std::size_t>::max(), 0, 0);
    });
    require(y[0] == 7, "Invalid NEON arguments modified output");
    std::cout << "NEON and scalar passed independent oracle, tails, strides and bounds\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
