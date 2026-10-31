#include "kernels/quantized/q5_neon.h"
#include "q5_support.h"
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
std::size_t positive(const std::string &s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Positive integer required");
  const auto n = std::stoull(s);
  if (!n || n > 2147483647)
    throw std::invalid_argument("Integer out of range");
  return static_cast<std::size_t>(n);
}
} // namespace
int main(int argc, char **argv) {
  try {
    using namespace q5bench;
    using Clock = std::chrono::steady_clock;
    std::size_t rows = 192, cols = 576, vectors = 1, repetitions = 6,
                min_ms = 100, seed = 42;
    std::string library;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "q5-neon-bench --upstream LIB [--rows N --cols N "
                     "--vectors N --repetitions N --min-ms N --seed N]\nThree "
                     "variants: original upstream / row1 ablation / row4 "
                     "candidate. Repetitions must be a multiple of6.\n";
        return 0;
      }
      if (++i == argc)
        throw std::invalid_argument("Missing value");
      if (key == "--upstream") {
        library = argv[i];
        continue;
      }
      const auto n = positive(argv[i]);
      if (key == "--rows")
        rows = n;
      else if (key == "--cols")
        cols = n;
      else if (key == "--vectors")
        vectors = n;
      else if (key == "--repetitions")
        repetitions = n;
      else if (key == "--min-ms")
        min_ms = n;
      else if (key == "--seed")
        seed = n;
      else
        throw std::invalid_argument("Unknown argument");
    }
    if (rows > 4096 || cols > 4096 || cols % 32 || vectors > 128 ||
        repetitions > 96 || repetitions % 6 || min_ms > 1000)
      throw std::invalid_argument(
          "Invalid size or unbalanced three-variant order");
    if (library.empty())
      throw std::invalid_argument("Original upstream library required");
    Upstream upstream(library);
    const std::size_t blocks = cols / 32;
    std::vector<Q5Block> w(rows * blocks);
    std::vector<Q8Block> x(vectors * blocks);
    fill(w, x, static_cast<std::uint32_t>(seed), false);
    std::vector<Expected> expected(rows * vectors);
    for (std::size_t v = 0; v < vectors; ++v)
      for (std::size_t r = 0; r < rows; ++r)
        expected[v * rows + r] =
            oracle(w.data() + r * blocks, x.data() + v * blocks, cols);
    constexpr const char *names[3] = {"upstream", "row1", "row4"};
    constexpr int orders[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1},
                                  {2, 1, 0}, {1, 0, 2}, {0, 2, 1}};
    std::array<std::vector<float>, 3> y;
    std::array<Accuracy, 3> accuracy;
    for (auto &values : y)
      values.resize(rows * vectors);
    const auto check = [&](int kind) {
      for (std::size_t i = 0; i < expected.size(); ++i)
        verify(y[kind][i], expected[i], cols, accuracy[kind]);
    };
    const auto call = [&](int kind) {
      if (kind == 0) {
        for (std::size_t v = 0; v < vectors; ++v)
          for (std::size_t r = 0; r < rows; ++r)
            upstream.dot(static_cast<int>(cols), &y[0][v * rows + r], 0,
                         w.data() + r * blocks, 0, x.data() + v * blocks, 0, 1);
      } else if (kind == 1)
        matvec_q5_q8_neon_row1(w.data(), x.data(), y[1].data(), rows, cols,
                               vectors);
      else
        matvec_q5_q8_neon_row4(w.data(), x.data(), y[2].data(), rows, cols,
                               vectors);
    };
    const auto batch = [&](int kind, std::size_t iterations) {
      const auto start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i)
        call(kind);
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          Clock::now() - start)
                          .count();
      if (ns <= 0)
        throw std::runtime_error("Clock resolution");
      return ns;
    };
    std::array<std::size_t, 3> iterations{};
    std::array<int, 3> calibration{};
    for (int i = 0; i < 3; ++i) {
      const int kind = static_cast<int>((seed + i) % 3);
      calibration[i] = kind;
      call(kind);
      check(kind);
      iterations[kind] = 1;
      while (batch(kind, iterations[kind]) <
             static_cast<std::int64_t>(min_ms * 1000000)) {
        if (iterations[kind] >= (1U << 24))
          throw std::runtime_error("Calibration did not converge");
        iterations[kind] *= 2;
      }
      check(kind);
    }
    std::vector<std::array<std::int64_t, 3>> samples(repetitions);
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
      for (const int kind : orders[rep % 6])
        samples[rep][kind] = batch(kind, iterations[kind]);
      for (int k = 0; k < 3; ++k)
        check(k);
    }
    std::cout
        << std::setprecision(17)
        << "{\"schema_version\":1,\"benchmark\":\"q5_neon_candidate\",\"data_"
           "kind\":\"measured_local\",\"correctness_passed\":true,\"threads\":"
           "1,\"upstream_symbol\":\"ggml_vec_dot_q5_0_q8_0\",\"abi_engine_"
           "commit\":\"a7a98e0fffed794396b3fbad4dcdbbc184963645\","
        << "\"rows\":" << rows << ",\"cols\":" << cols
        << ",\"vectors\":" << vectors << ",\"seed\":" << seed
        << ",\"repetitions\":" << repetitions << ",\"min_sample_ms\":" << min_ms
        << ",\"q5_bytes\":" << w.size() * sizeof(Q5Block)
        << ",\"q8_bytes\":" << x.size() * sizeof(Q8Block)
        << ",\"timing_scope\":\"reused packed buffers; original row-dot loop "
           "vs matrix candidates; initialization, oracle, allocation and "
           "quantization excluded\",\"calibration_order\":[";
    for (int i = 0; i < 3; ++i) {
      if (i)
        std::cout << ',';
      std::cout << '"' << names[calibration[i]] << '"';
    }
    std::cout << "],\"variants\":{";
    for (int k = 0; k < 3; ++k) {
      if (k)
        std::cout << ',';
      std::cout << '"' << names[k] << "\":{\"iterations\":" << iterations[k]
                << ",\"checksum\":"
                << std::accumulate(y[k].begin(), y[k].end(), 0.0)
                << ",\"max_abs_error\":" << accuracy[k].max_abs_error
                << ",\"max_bound_fraction\":" << accuracy[k].max_bound_fraction
                << '}';
    }
    std::cout << "},\"triplets\":[";
    for (std::size_t i = 0; i < repetitions; ++i) {
      if (i)
        std::cout << ',';
      std::cout << "{\"order\":[";
      for (int j = 0; j < 3; ++j) {
        if (j)
          std::cout << ',';
        std::cout << '"' << names[orders[i % 6][j]] << '"';
      }
      std::cout << "],\"batch_ns\":{";
      for (int k = 0; k < 3; ++k) {
        if (k)
          std::cout << ',';
        std::cout << '"' << names[k] << "\":" << samples[i][k];
      }
      std::cout << "}}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "q5-neon-bench: " << e.what() << '\n';
    return 1;
  }
}
