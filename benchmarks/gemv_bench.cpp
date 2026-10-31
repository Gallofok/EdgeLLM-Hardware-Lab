#include "kernels/gemv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::size_t positive(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Arguments must be positive decimal integers");
  const auto n = std::stoull(value);
  if (n == 0 || n > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Argument out of range");
  return static_cast<std::size_t>(n);
}

double checksum(const std::vector<float>& y) {
  double sum = 0;
  for (const auto value : y) sum += value;
  return sum;
}

void verify(const std::vector<float>& w, const std::vector<float>& x,
            const std::vector<float>& y, std::size_t rows, std::size_t cols) {
  const long double u = std::numeric_limits<float>::epsilon();
  const long double gamma = 2 * cols * u / (1 - 2 * cols * u);
  for (std::size_t row = 0; row < rows; ++row) {
    long double expected = 0, magnitude = 0;
    for (std::size_t col = 0; col < cols; ++col) {
      const auto term = static_cast<long double>(w[row * cols + col]) * x[col];
      expected += term;
      magnitude += std::abs(term);
    }
    if (!std::isfinite(y[row]) || std::abs(y[row] - expected) > 1e-6L + gamma * magnitude)
      throw std::runtime_error("Independent long-double correctness check failed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t rows = 512, cols = 512, repetitions = 5, min_ms = 100;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "gemv-bench [--rows N] [--cols N] [--repetitions N] [--min-ms N]\n"
                     "Single-thread FP32 scalar GEMV; JSON to stdout. Defaults: 512,512,5,100.\n"
                     "Caps: 4096 per dimension, 100 repetitions, 1000 min-ms.\n";
        return 0;
      }
      if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
      const auto value = positive(argv[++i]);
      if (key == "--rows") rows = value;
      else if (key == "--cols") cols = value;
      else if (key == "--repetitions") repetitions = value;
      else if (key == "--min-ms") min_ms = value;
      else throw std::invalid_argument("Unknown argument: " + key);
    }
    if (rows > 4096 || cols > 4096 || repetitions > 100 || min_ms > 1000)
      throw std::invalid_argument("Resource limit exceeded; see --help");
    std::vector<float> w(rows * cols), x(cols), y(rows);
    std::uint32_t state = 42;
    const auto next = [&state]() {
      state = state * 1664525U + 1013904223U;
      return static_cast<float>(static_cast<int>((state >> 16) % 2049) - 1024) / 256.0F;
    };
    for (auto& value : w) value = next();
    for (auto& value : x) value = next();
    edgellm::gemv_scalar(w.data(), x.data(), y.data(), rows, cols, cols);
    verify(w, x, y, rows, cols);
    const auto batch = [&](std::size_t iterations) {
      const auto start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i)
        edgellm::gemv_scalar(w.data(), x.data(), y.data(), rows, cols, cols);
      return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    };
    std::size_t iterations = 1;
    while (batch(iterations) < static_cast<std::int64_t>(min_ms * 1000000)) {
      if (iterations >= (1U << 24)) throw std::runtime_error("Calibration did not converge");
      iterations *= 2;
    }
    std::vector<std::int64_t> samples;
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
      const auto elapsed = batch(iterations);
      if (elapsed <= 0) throw std::runtime_error("Insufficient clock resolution");
      samples.push_back(elapsed);
      verify(w, x, y, rows, cols);
    }
    // Only publish JSON once all correctness checks have passed. Keep exact
    // integer batch times so summaries can be independently reconstructed.
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"benchmark\":\"fp32_gemv_microbenchmark\","
                 "\"implementation\":\"scalar\",\"threads\":1,\"seed\":42,"
                 "\"data_kind\":\"measured_local\",\"correctness_passed\":true,"
                 "\"timing_scope\":\"repeated buffers, no cache flush; allocation and oracle excluded\","
              << "\"rows\":" << rows << ",\"cols\":" << cols << ",\"row_stride\":" << cols
              << ",\"matrix_bytes\":" << w.size() * sizeof(float)
              << ",\"repetitions\":" << repetitions << ",\"iterations_per_sample\":" << iterations
              << ",\"min_sample_ms\":" << min_ms << ",\"output_checksum\":" << checksum(y)
              << ",\"samples_batch_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << samples[i];
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gemv-bench: " << error.what() << '\n';
    return 1;
  }
}
