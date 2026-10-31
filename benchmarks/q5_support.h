#pragma once

#include "kernels/quantized/q5.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace q5bench {
using namespace edgellm::quantized;
using Dot = void (*)(int, float*, std::size_t, const void*, std::size_t,
                    const void*, std::size_t, int);

class Upstream {
  void* handle_ = nullptr;
 public:
  Dot dot = nullptr;
  explicit Upstream(const std::string& library) {
    const std::uint16_t one = 1;
    if (*reinterpret_cast<const unsigned char*>(&one) != 1)
      throw std::runtime_error("Pinned GGML ABI adapter requires little-endian storage");
    handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error("Cannot load upstream GGML library (no fallback)");
    auto init = reinterpret_cast<void (*)()>(dlsym(handle_, "ggml_cpu_init"));
    dot = reinterpret_cast<Dot>(dlsym(handle_, "ggml_vec_dot_q5_0_q8_0"));
    if (!dot || !init) {
      dlclose(handle_); handle_ = nullptr;
      throw std::runtime_error("Required upstream GGML symbols unavailable");
    }
    init();  // Includes FP16 table initialization, excluded from all timings.
  }
  Upstream(const Upstream&) = delete;
  Upstream& operator=(const Upstream&) = delete;
  ~Upstream() { if (handle_) dlclose(handle_); }
};

inline void reference(int n, float* result, std::size_t, const void* w,
                      std::size_t, const void* x, std::size_t, int nrc) {
  if (nrc != 1 || n < 0) throw std::invalid_argument("Unsupported dot ABI shape");
  *result = dot_q5_q8_reference(static_cast<const Q5Block*>(w),
                               static_cast<const Q8Block*>(x), static_cast<std::size_t>(n));
}

// Independent mathematical binary16 decoder. Does not call the kernel decoder.
inline double oracle_half(std::uint16_t bits) {
  const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
  if (exponent == 31) throw std::invalid_argument("Oracle requires finite scales");
  const double value = exponent ? std::ldexp(1024.0 + fraction, exponent - 25)
                                : std::ldexp(static_cast<double>(fraction), -24);
  return bits & 0x8000 ? -value : value;
}
struct Expected { double value = 0, magnitude = 0; };
inline Expected oracle(const Q5Block* w, const Q8Block* x, std::size_t n) {
  Expected result;
  for (std::size_t b = 0; b < n / 32; ++b) {
    // Decode each operand independently; do not use an integer block dot.
    const auto ws = oracle_half(w[b].half_scale), xs = oracle_half(x[b].half_scale);
    for (std::size_t i = 0; i < 32; ++i) {
      const int low = i < 16 ? w[b].low[i] % 16 : w[b].low[i - 16] / 16;
      const bool fifth = (w[b].high[i / 8] & (1U << (i % 8))) != 0;
      const double weight = ws * (low + (fifth ? 16 : 0) - 16);
      const double activation = xs * x[b].values[i];
      const auto product = weight * activation;
      result.value += product; result.magnitude += std::abs(product);
    }
  }
  return result;
}
struct Accuracy { double max_abs_error = 0, max_bound_fraction = 0; };
inline void verify(float value, const Expected& expected, std::size_t n, Accuracy& accuracy) {
  const auto terms = 8.0 * (n / 32 + 1);
  const double epsilon = std::numeric_limits<float>::epsilon();
  const auto gamma = terms * epsilon / (1 - terms * epsilon);
  const auto bound = 1e-6 + gamma * expected.magnitude;
  const auto error = std::abs(value - expected.value);
  if (!std::isfinite(value) || error > bound)
    throw std::runtime_error("Independent decoded FP64 oracle check failed");
  accuracy.max_abs_error = std::max(accuracy.max_abs_error, error);
  accuracy.max_bound_fraction = std::max(accuracy.max_bound_fraction, error / bound);
}
inline void fill(std::vector<Q5Block>& w, std::vector<Q8Block>& x, std::uint32_t seed, bool adversarial = true) {
  const auto next = [&seed]() { seed = seed * 1664525U + 1013904223U; return seed; };
  // Finite positive/negative, non-power-of-two, subnormal and zero scales.
  constexpr std::uint16_t scales[] = {0x3001, 0xb255, 0x3555, 0x2803, 0x0001, 0, 0x8000};
  for (auto& block : w) {
    block.half_scale = scales[(next() >> 16) % (adversarial ? 7 : 4)];
    for (auto& value : block.high) value = static_cast<std::uint8_t>(next() >> 16);
    for (auto& value : block.low) value = static_cast<std::uint8_t>(next() >> 16);
  }
  for (auto& block : x) {
    block.half_scale = scales[(next() >> 16) % (adversarial ? 7 : 4)];
    for (auto& value : block.values) value = static_cast<std::int8_t>(static_cast<int>((next() >> 16) % 256) - 128);
  }
}
}  // namespace q5bench
