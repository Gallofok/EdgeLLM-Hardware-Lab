#include "q5_support.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
using Clock = std::chrono::steady_clock;
std::size_t positive(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Expected a positive decimal integer");
  const auto n = std::stoull(value);
  if (n == 0 || n > 2147483647) throw std::invalid_argument("Integer out of range");
  return static_cast<std::size_t>(n);
}
}
int main(int argc, char** argv) {
  try {
    using namespace q5bench;
    std::size_t rows = 192, cols = 576, vectors = 1, repetitions = 6, min_ms = 100, seed = 42;
    std::string library;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "q5-bench --upstream LIB [--rows N] [--cols N] [--vectors N]\n"
                     "  [--repetitions N] [--min-ms N] [--seed N]\n"
                     "Packed Q5_0 x Q8_0; single-thread row dots, reference/upstream paired batches.\n"
                     "Defaults:192,576,1,6,100,42. Caps:4096,4096,128,100,1000.\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("Missing argument value");
      if (key == "--upstream") { library = argv[i]; continue; }
      const auto n = positive(argv[i]);
      if (key == "--rows") rows = n;
      else if (key == "--cols") cols = n;
      else if (key == "--vectors") vectors = n;
      else if (key == "--repetitions") repetitions = n;
      else if (key == "--min-ms") min_ms = n;
      else if (key == "--seed") seed = n;
      else throw std::invalid_argument("Unknown argument: " + key);
    }
    if (rows > 4096 || cols > 4096 || cols % 32 || vectors > 128 ||
        repetitions > 100 || repetitions % 2 || min_ms > 1000)
      throw std::invalid_argument("Invalid size, block width or unbalanced repetition count");
    if (library.empty()) throw std::invalid_argument("Explicit upstream library required; no fallback");
    Upstream upstream(library);
    const auto blocks = cols / 32;
    std::vector<Q5Block> w(rows * blocks);
    std::vector<Q8Block> x(vectors * blocks);
    fill(w, x, static_cast<std::uint32_t>(seed), false);
    std::vector<Expected> expected(rows * vectors);
    for (std::size_t v = 0; v < vectors; ++v)
      for (std::size_t r = 0; r < rows; ++r)
        expected[v * rows + r] = oracle(w.data() + r * blocks, x.data() + v * blocks, cols);
    std::vector<float> ref_y(rows * vectors), upstream_y(rows * vectors);
    const auto check = [&](const std::vector<float>& y, Accuracy& accuracy) {
      for (std::size_t i = 0; i < y.size(); ++i) verify(y[i], expected[i], cols, accuracy);
    };
    const auto call = [&](Dot dot, std::vector<float>& y) {
      for (std::size_t v = 0; v < vectors; ++v)
        for (std::size_t r = 0; r < rows; ++r)
          dot(static_cast<int>(cols), &y[v * rows + r], 0, w.data() + r * blocks, 0,
              x.data() + v * blocks, 0, 1);
    };
    const auto batch = [&](Dot dot, std::size_t iterations, std::vector<float>& y) {
      const auto start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i) call(dot, y);
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
      if (ns <= 0) throw std::runtime_error("Insufficient clock resolution");
      return ns;
    };
    Accuracy ref_accuracy, upstream_accuracy;
    const auto calibrate = [&](Dot dot, std::vector<float>& y, Accuracy& accuracy) {
      call(dot, y); check(y, accuracy);
      std::size_t iterations = 1;
      while (batch(dot, iterations, y) < static_cast<std::int64_t>(min_ms * 1000000)) {
        if (iterations >= (1U << 24)) throw std::runtime_error("Calibration failed to converge");
        iterations *= 2;
      }
      check(y, accuracy);
      return iterations;
    };
    const auto ref_iterations = calibrate(reference, ref_y, ref_accuracy);
    const auto upstream_iterations = calibrate(upstream.dot, upstream_y, upstream_accuracy);
    std::vector<std::int64_t> ref_ns(repetitions), upstream_ns(repetitions);
    for (std::size_t i = 0; i < repetitions; ++i) {
      if (i % 2 == 0) {
        ref_ns[i] = batch(reference, ref_iterations, ref_y);
        upstream_ns[i] = batch(upstream.dot, upstream_iterations, upstream_y);
      } else {
        upstream_ns[i] = batch(upstream.dot, upstream_iterations, upstream_y);
        ref_ns[i] = batch(reference, ref_iterations, ref_y);
      }
      check(ref_y, ref_accuracy); check(upstream_y, upstream_accuracy);
    }
    std::cout << std::setprecision(17)
      << "{\"schema_version\":1,\"benchmark\":\"q5_0_q8_0_reference_upstream\","
         "\"data_kind\":\"measured_local\",\"correctness_passed\":true,\"threads\":1,"
         "\"upstream_symbol\":\"ggml_vec_dot_q5_0_q8_0\","
         "\"abi_engine_commit\":\"a7a98e0fffed794396b3fbad4dcdbbc184963645\","
         "\"timing_scope\":\"reused synthetic packed buffers; serial per-row dots; initialization, allocation and decoded FP64 oracle excluded\","
      << "\"rows\":" << rows << ",\"cols\":" << cols << ",\"vectors\":" << vectors
      << ",\"seed\":" << seed << ",\"repetitions\":" << repetitions << ",\"min_sample_ms\":" << min_ms
      << ",\"q5_bytes\":" << w.size() * sizeof(Q5Block) << ",\"q8_bytes\":" << x.size() * sizeof(Q8Block)
      << ",\"reference_iterations\":" << ref_iterations << ",\"upstream_iterations\":" << upstream_iterations
      << ",\"reference_checksum\":" << std::accumulate(ref_y.begin(), ref_y.end(), 0.0)
      << ",\"upstream_checksum\":" << std::accumulate(upstream_y.begin(), upstream_y.end(), 0.0)
      << ",\"reference_max_abs_error\":" << ref_accuracy.max_abs_error
      << ",\"upstream_max_abs_error\":" << upstream_accuracy.max_abs_error
      << ",\"reference_max_bound_fraction\":" << ref_accuracy.max_bound_fraction
      << ",\"upstream_max_bound_fraction\":" << upstream_accuracy.max_bound_fraction
      << ",\"pairs\":[";
    for (std::size_t i = 0; i < repetitions; ++i) {
      if (i) std::cout << ',';
      std::cout << "{\"order\":[" << (i % 2 == 0 ? "\"reference\",\"upstream\"" : "\"upstream\",\"reference\"")
                << "],\"reference_batch_ns\":" << ref_ns[i] << ",\"upstream_batch_ns\":" << upstream_ns[i] << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << "q5-bench: " << e.what() << '\n'; return 1; }
}
