#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "integration/q5_ggml_bridge.h"
#include "q5_support.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <link.h>
#include <numeric>

namespace {
using namespace q5bench;
using Clock = std::chrono::steady_clock;
constexpr const char *names[] = {"original", "compatible", "candidate"};
constexpr int orders[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1},
                              {2, 1, 0}, {1, 0, 2}, {0, 2, 1}};

// Separate glibc link namespaces prevent the identically named GGML
// dependency sets from interposing. No kernel is recompiled or copied here.
class Engine {
  void *handle_ = nullptr;

public:
  Lmid_t namespace_id{};
  explicit Engine(const std::string &path) {
    handle_ = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      throw std::runtime_error("Isolated GGML load failed (no fallback)");
    link_map *map = nullptr;
    if (dlinfo(handle_, RTLD_DI_LINKMAP, &map) ||
        dlinfo(handle_, RTLD_DI_LMID, &namespace_id))
      throw std::runtime_error("Cannot verify isolated loader namespace");
    const auto directory = std::filesystem::canonical(path).parent_path();
    while (map->l_prev)
      map = map->l_prev;
    for (; map; map = map->l_next) {
      const std::filesystem::path loaded(map->l_name);
      if (loaded.filename().string().rfind("libggml", 0) == 0 &&
          std::filesystem::canonical(loaded).parent_path() != directory)
        throw std::runtime_error(
            "GGML dependency escaped its expected directory");
    }
    symbol<void (*)()>("ggml_cpu_init")();
  }
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  ~Engine() {
    if (handle_)
      dlclose(handle_);
  }
  bool has(const char *name) { return dlsym(handle_, name) != nullptr; }
  template <class T> T symbol(const char *name) {
    auto pointer = reinterpret_cast<T>(dlsym(handle_, name));
    if (!pointer)
      throw std::runtime_error("Required GGML symbol unavailable");
    return pointer;
  }
};

std::size_t positive(const std::string &s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Positive integer required");
  const auto n = std::stoull(s);
  if (!n || n > 2147483647)
    throw std::invalid_argument("Integer out of range");
  return n;
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::size_t rows = 192, vectors = 1, repetitions = 6, min_ms = 100,
                seed = 42, chunk = 16;
    constexpr std::size_t cols = 576, blocks = cols / 32;
    std::string original_path, integrated_path, candidate_path;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout
            << "q5-two-block-bench --original LIB --integrated LIB --candidate "
               "LIB [--rows N --vectors N --repetitions N --min-ms N --seed N "
               "--chunk N]\nFixed width576. Original dot / measured compatible "
               "bridge / two-block candidate bridge. Balanced six-order timing, "
               "auditing off.\n";
        return 0;
      }
      if (++i == argc)
        throw std::invalid_argument("Missing value");
      if (key == "--original")
        original_path = argv[i];
      else if (key == "--integrated")
        integrated_path = argv[i];
      else if (key == "--candidate")
        candidate_path = argv[i];
      else {
        const auto n = positive(argv[i]);
        if (key == "--rows")
          rows = n;
        else if (key == "--vectors")
          vectors = n;
        else if (key == "--repetitions")
          repetitions = n;
        else if (key == "--min-ms")
          min_ms = n;
        else if (key == "--seed")
          seed = n;
        else if (key == "--chunk")
          chunk = n;
        else
          throw std::invalid_argument("Unknown argument");
      }
    }
    if (rows > 4096 || vectors > 8 || repetitions > 96 || repetitions % 6 ||
        min_ms > 1000 || chunk > 16 || original_path.empty() ||
        integrated_path.empty() || candidate_path.empty())
      throw std::invalid_argument(
          "Invalid size, libraries or unbalanced order");
    Engine original(original_path), integrated(integrated_path),
        candidate(candidate_path);
    if (original.namespace_id == integrated.namespace_id ||
        original.namespace_id == candidate.namespace_id ||
        integrated.namespace_id == candidate.namespace_id)
      throw std::runtime_error("Engine namespaces are not isolated");
    if (integrated.has("edgellm_q5_loop_variant") ||
        std::strcmp(candidate.symbol<const char *(*)()>(
                        "edgellm_q5_loop_variant")(),
                    "two-block"))
      throw std::runtime_error("Measured library identity is wrong");
    const auto dot = original.symbol<Dot>("ggml_vec_dot_q5_0_q8_0");
    using Configure = void (*)(int, int);
    using Snapshot = edgellm_q5_audit (*)();
    using Bridge = decltype(&edgellm_q5_try_row4);
    constexpr const char *dot_name = "ggml_vec_dot_q5_0_q8_0";
    const Dot control_dots[] = {integrated.symbol<Dot>(dot_name),
                                candidate.symbol<Dot>(dot_name)};
    const Configure configures[] = {
        integrated.symbol<Configure>("edgellm_q5_configure"),
        candidate.symbol<Configure>("edgellm_q5_configure")};
    const Snapshot snapshots[] = {
        integrated.symbol<Snapshot>("edgellm_q5_snapshot"),
        candidate.symbol<Snapshot>("edgellm_q5_snapshot")};
    const Bridge bridges[] = {
        integrated.symbol<Bridge>("edgellm_q5_try_row4"),
        candidate.symbol<Bridge>("edgellm_q5_try_row4")};
    std::vector<Q5Block> w(rows * blocks);
    std::vector<Q8Block> x(vectors * blocks);
    fill(w, x, static_cast<std::uint32_t>(seed), false);
    std::vector<Expected> expected(rows * vectors);
    std::array<std::vector<float>, 3> y;
    std::array<Accuracy, 3> accuracy{};
    for (auto &output : y)
      output.resize(rows * vectors);
    for (std::size_t v = 0; v < vectors; ++v)
      for (std::size_t r = 0; r < rows; ++r)
        expected[v * rows + r] =
            oracle(w.data() + r * blocks, x.data() + v * blocks, cols);
    const auto call = [&](int kind) {
      for (std::size_t v = 0; v < vectors; ++v)
        for (std::size_t r = 0; r < rows; r += chunk) {
          alignas(16) float temporary[16];
          const auto count = std::min(chunk, rows - r);
          if (kind == 0) {
            for (std::size_t i = 0; i < count; ++i)
              dot(cols, temporary + i, 0, w.data() + (r + i) * blocks, 0,
                  x.data() + v * blocks, 0, 1);
          } else if (!bridges[kind - 1](cols, count, w.data() + r * blocks,
                                        blocks * sizeof(Q5Block),
                                        x.data() + v * blocks, temporary, 0)) {
            throw std::runtime_error("Candidate unexpectedly fell back");
          }
          std::memcpy(y[kind].data() + v * rows + r, temporary,
                      count * sizeof(float));
        }
    };
    const auto check = [&](int kind) {
      for (std::size_t i = 0; i < expected.size(); ++i)
        verify(y[kind][i], expected[i], cols, accuracy[kind]);
      if (kind &&
          std::memcmp(y[0].data(), y[kind].data(), y[0].size() * sizeof(float)))
        throw std::runtime_error("Output differs from original bytes");
    };
    call(0);
    check(0);
    // Check the integrated library's ordinary dot against the actual original.
    for (std::size_t v = 0; v < vectors; ++v)
      for (std::size_t r = 0; r < rows; ++r) {
        for (const auto control_dot : control_dots) {
          float value;
          control_dot(cols, &value, 0, w.data() + r * blocks, 0,
                      x.data() + v * blocks, 0, 1);
          if (std::memcmp(&value, &y[0][v * rows + r], sizeof(float)))
            throw std::runtime_error("Original/integrated dot controls differ");
        }
      }
    // Independent audited proof; none of this enters timing.
    for (int kind : {1, 2}) {
      configures[kind - 1](1, 1);
      call(kind);
      check(kind);
      const auto proof = snapshots[kind - 1]();
      if (proof.handled != vectors * ((rows + chunk - 1) / chunk) ||
          proof.rows != rows * vectors || proof.thread_mask != 1 ||
          proof.disabled || proof.width || proof.stride || proof.invalid)
        throw std::runtime_error("Audited dispatch/row proof failed");
    }
    configures[0](1, 0);
    configures[1](1, 0);
    const auto batch = [&](int kind, std::size_t iterations) {
      const auto start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i)
        call(kind);
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          Clock::now() - start)
                          .count();
      for (const auto snapshot : snapshots) {
        const auto audit = snapshot();
        if (audit.handled || audit.rows || audit.disabled || audit.width ||
            audit.stride || audit.invalid || audit.thread_mask)
          throw std::runtime_error(
              "Audit counter changed during formal timing");
      }
      if (ns <= 0)
        throw std::runtime_error("Clock resolution");
      return ns;
    };
    std::array<std::size_t, 3> iterations{};
    for (int i = 0; i < 3; ++i) {
      const int kind = (seed + i) % 3;
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
      for (int kind : orders[rep % 6])
        samples[rep][kind] = batch(kind, iterations[kind]);
      for (int kind = 0; kind < 3; ++kind)
        check(kind);
    }
    std::cout
        << std::setprecision(17)
        << "{\"schema_version\":1,\"benchmark\":\"q5_two_block\","
           "\"data_kind\":\"measured_local\",\"correctness_passed\":true,"
           "\"isolated_namespaces\":true,\"library_identity_passed\":true,"
           "\"bridges_bitwise_equal\":true,"
           "\"integrated_dot_equal\":true,\"audit_during_timing\":false,"
           "\"dispatch_proofs_passed\":true,\"threads\":1,\"cols\":576,"
        << "\"rows\":" << rows << ",\"vectors\":" << vectors
        << ",\"chunk\":" << chunk << ",\"seed\":" << seed
        << ",\"repetitions\":" << repetitions << ",\"min_sample_ms\":" << min_ms
        << ",\"timing_scope\":\"same reused synthetic packed inputs; original "
           "row-dot loop, measured compatible bridge and candidate bridge; "
           "common chunk output copy; initialization, oracle, allocation, "
           "configuration and quantization excluded\","
           "\"variants\":{";
    for (int kind = 0; kind < 3; ++kind) {
      if (kind)
        std::cout << ',';
      std::cout << '"' << names[kind]
                << "\":{\"iterations\":" << iterations[kind] << ",\"checksum\":"
                << std::accumulate(y[kind].begin(), y[kind].end(), 0.0)
                << ",\"max_abs_error\":" << accuracy[kind].max_abs_error
                << ",\"max_bound_fraction\":"
                << accuracy[kind].max_bound_fraction << '}';
    }
    std::cout << "},\"triplets\":[";
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
      if (rep)
        std::cout << ',';
      std::cout << "{\"order\":[";
      for (int j = 0; j < 3; ++j) {
        if (j)
          std::cout << ',';
        std::cout << '"' << names[orders[rep % 6][j]] << '"';
      }
      std::cout << "],\"batch_ns\":{";
      for (int kind = 0; kind < 3; ++kind) {
        if (kind)
          std::cout << ',';
        std::cout << '"' << names[kind] << "\":" << samples[rep][kind];
      }
      std::cout << "}}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "q5-two-block-bench: " << error.what() << '\n';
    return 1;
  }
}
