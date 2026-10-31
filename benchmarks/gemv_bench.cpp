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

struct Accuracy { double max_abs_error = 0, max_bound_fraction = 0; };

Accuracy verify(const std::vector<float>& w, const std::vector<float>& x,
            const std::vector<float>& y, std::size_t rows, std::size_t cols) {
  Accuracy accuracy;
  const long double u = std::numeric_limits<float>::epsilon();
  const long double gamma = 2 * cols * u / (1 - 2 * cols * u);
  for (std::size_t row = 0; row < rows; ++row) {
    long double expected = 0, magnitude = 0;
    for (std::size_t col = 0; col < cols; ++col) {
      const auto term = static_cast<long double>(w[row * cols + col]) * x[col];
      expected += term;
      magnitude += std::abs(term);
    }
    const auto error = std::abs(y[row] - expected);
    const auto bound = 1e-6L + gamma * magnitude;
    accuracy.max_abs_error = std::max(accuracy.max_abs_error, static_cast<double>(error));
    accuracy.max_bound_fraction = std::max(accuracy.max_bound_fraction, static_cast<double>(error / bound));
    if (!std::isfinite(y[row]) || error > bound)
      throw std::runtime_error("Independent long-double correctness check failed");
  }
  return accuracy;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t rows = 512, cols = 512, repetitions = 5, min_ms = 100;
    std::string implementation = "scalar";
    bool explicit_repetitions = false;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--features") {
        std::cout << "{\"implementations\":[\"scalar\""
#ifdef EDGELLM_HAS_NEON
                  << ",\"neon\""
#endif
                  << "]}\n";
        return 0;
      }
      if (key == "--implementation") {
        if (++i == argc) throw std::invalid_argument("Missing implementation");
        implementation = argv[i];
        continue;
      }
      if (key == "--help") {
        std::cout << "gemv-bench [--implementation scalar|neon|paired] [--rows N] [--cols N]\n"
                     "           [--repetitions N] [--min-ms N] [--features]\n"
                     "Single-thread FP32 GEMV; JSON to stdout. Defaults: 512,512,5,100 (paired:6 reps).\n"
                     "Caps: 4096 per dimension, 100 repetitions, 1000 min-ms.\n";
        return 0;
      }
      if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
      const auto value = positive(argv[++i]);
      if (key == "--rows") rows = value;
      else if (key == "--cols") cols = value;
      else if (key == "--repetitions") { repetitions = value; explicit_repetitions = true; }
      else if (key == "--min-ms") min_ms = value;
      else throw std::invalid_argument("Unknown argument: " + key);
    }
    if (implementation != "scalar" && implementation != "neon" && implementation != "paired")
      throw std::invalid_argument("Unknown implementation");
#ifndef EDGELLM_HAS_NEON
    if (implementation != "scalar") throw std::invalid_argument("ARM64 NEON is unavailable in this build");
#endif
    if (implementation == "paired" && !explicit_repetitions) repetitions = 6;
    if (implementation == "paired" && repetitions % 2 != 0)
      throw std::invalid_argument("Paired repetitions must be even to balance execution order");
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
    using Kernel = decltype(&edgellm::gemv_scalar);
    Kernel selected = edgellm::gemv_scalar;
#ifdef EDGELLM_HAS_NEON
    if (implementation == "neon") selected = edgellm::gemv_neon;
#endif
    const auto batch = [&](Kernel kernel, std::size_t iterations, std::vector<float>& result) {
      const auto start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i)
        kernel(w.data(), x.data(), result.data(), rows, cols, cols);
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - start).count();
      if (elapsed <= 0) throw std::runtime_error("Insufficient clock resolution");
      return elapsed;
    };
    const auto calibrate = [&](Kernel kernel, std::vector<float>& result) {
      kernel(w.data(), x.data(), result.data(), rows, cols, cols);
      verify(w, x, result, rows, cols);
      std::size_t count = 1;
      while (batch(kernel, count, result) < static_cast<std::int64_t>(min_ms * 1000000)) {
        if (count >= (1U << 24)) throw std::runtime_error("Calibration did not converge");
        count *= 2;
      }
      return count;
    };
#ifdef EDGELLM_HAS_NEON
    if (implementation == "paired") {
      std::vector<float> other(rows);
      const auto scalar_iterations = calibrate(edgellm::gemv_scalar, y);
      const auto neon_iterations = calibrate(edgellm::gemv_neon, other);
      std::vector<std::int64_t> scalar_ns(repetitions), neon_ns(repetitions);
      for (std::size_t i = 0; i < repetitions; ++i) {
        if (i % 2 == 0) {
          scalar_ns[i] = batch(edgellm::gemv_scalar, scalar_iterations, y);
          neon_ns[i] = batch(edgellm::gemv_neon, neon_iterations, other);
        } else {
          neon_ns[i] = batch(edgellm::gemv_neon, neon_iterations, other);
          scalar_ns[i] = batch(edgellm::gemv_scalar, scalar_iterations, y);
        }
        verify(w, x, y, rows, cols);
        verify(w, x, other, rows, cols);
      }
      const auto scalar_error = verify(w, x, y, rows, cols);
      const auto neon_error = verify(w, x, other, rows, cols);
      std::cout << std::setprecision(17)
                << "{\"schema_version\":2,\"benchmark\":\"fp32_gemv_paired\","
                   "\"data_kind\":\"measured_local\",\"threads\":1,\"seed\":42,"
                   "\"correctness_passed\":true,"
                   "\"timing_scope\":\"alternating batches, repeated buffers; oracle excluded\","
                << "\"rows\":" << rows << ",\"cols\":" << cols
                << ",\"row_stride\":" << cols << ",\"matrix_bytes\":" << w.size() * sizeof(float)
                << ",\"repetitions\":" << repetitions << ",\"min_sample_ms\":" << min_ms
                << ",\"scalar_iterations_per_sample\":" << scalar_iterations
                << ",\"neon_iterations_per_sample\":" << neon_iterations
                << ",\"scalar_checksum\":" << checksum(y)
                << ",\"neon_checksum\":" << checksum(other)
                << ",\"scalar_max_abs_error\":" << scalar_error.max_abs_error
                << ",\"neon_max_abs_error\":" << neon_error.max_abs_error
                << ",\"scalar_max_bound_fraction\":" << scalar_error.max_bound_fraction
                << ",\"neon_max_bound_fraction\":" << neon_error.max_bound_fraction
                << ",\"pairs\":[";
      for (std::size_t i = 0; i < repetitions; ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"order\":[" << (i % 2 == 0 ? "\"scalar\",\"neon\"" : "\"neon\",\"scalar\"")
                  << "],\"scalar_batch_ns\":" << scalar_ns[i]
                  << ",\"neon_batch_ns\":" << neon_ns[i] << '}';
      }
      std::cout << "]}\n";
      return 0;
    }
#endif
    const auto iterations = calibrate(selected, y);
    std::vector<std::int64_t> samples;
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
      samples.push_back(batch(selected, iterations, y));
      verify(w, x, y, rows, cols);
    }
    // Only publish JSON once all correctness checks have passed. Keep exact
    // integer batch times so summaries can be independently reconstructed.
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"benchmark\":\"fp32_gemv_microbenchmark\","
                 "\"implementation\":\"" << implementation << "\",\"threads\":1,\"seed\":42,"
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
