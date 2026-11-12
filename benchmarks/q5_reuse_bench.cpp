#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// Diagnostic: does the two-block loop's margin over the original dot depend on
// how often a 16-row weight block is reused and on the weight footprint?
// Traversal order matches the pinned engine: block by block, each block applied
// to `reuse` activation vectors before moving on. Not an LLM measurement.
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
constexpr const char *names[] = {"original", "candidate"};
constexpr std::size_t chunk = 16;

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
    std::size_t rows = 192, reuse = 1, repetitions = 6, min_ms = 100, seed = 42,
                cols = 576;
    std::string original_path, candidate_path, variant = "two-block";
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "q5-reuse-bench --original LIB --candidate LIB [--variant "
                     "NAME --rows N --reuse N --repetitions N --min-ms N "
                     "--seed N --cols N]\nWidth 576 by default, 16-row blocks, engine "
                     "traversal order. Original dot versus the named "
                     "candidate bridge, auditing off.\n";
        return 0;
      }
      if (++i == argc)
        throw std::invalid_argument("Missing value");
      if (key == "--original")
        original_path = argv[i];
      else if (key == "--candidate")
        candidate_path = argv[i];
      else if (key == "--variant")
        variant = argv[i];
      else {
        const auto n = positive(argv[i]);
        if (key == "--rows")
          rows = n;
        else if (key == "--reuse")
          reuse = n;
        else if (key == "--repetitions")
          repetitions = n;
        else if (key == "--min-ms")
          min_ms = n;
        else if (key == "--seed")
          seed = n;
        else if (key == "--cols")
          cols = n;
        else
          throw std::invalid_argument("Unknown argument");
      }
    }
    // An even block count keeps the upstream two-accumulator order well defined.
    if (cols % 64 || cols > 4096)
      throw std::invalid_argument("Width must be a multiple of 64 up to 4096");
    const std::size_t blocks = cols / 32;
    if (rows % chunk || rows > 262144 || reuse > 16 || repetitions > 96 ||
        repetitions % 2 || min_ms > 1000 || original_path.empty() ||
        candidate_path.empty())
      throw std::invalid_argument("Invalid size, libraries or unbalanced order");
    Engine original(original_path), candidate(candidate_path);
    if (original.namespace_id == candidate.namespace_id)
      throw std::runtime_error("Engine namespaces are not isolated");
    if (std::strcmp(candidate.symbol<const char *(*)()>(
                        "edgellm_q5_loop_variant")(),
                    variant.c_str()) ||
        (variant != "two-block" && variant != "two-row-prefetch" &&
         variant != "two-row-plain"))
      throw std::runtime_error("Measured library identity is wrong");
    const auto dot = original.symbol<Dot>("ggml_vec_dot_q5_0_q8_0");
    const auto configure =
        candidate.symbol<void (*)(int, int)>("edgellm_q5_configure");
    const auto snapshot =
        candidate.symbol<edgellm_q5_audit (*)()>("edgellm_q5_snapshot");
    const auto bridge = candidate.symbol<decltype(&edgellm_q5_try_row4)>(
        "edgellm_q5_try_row4");
    std::vector<Q5Block> w(rows * blocks);
    std::vector<Q8Block> x(reuse * blocks);
    fill(w, x, static_cast<std::uint32_t>(seed), false);
    std::array<std::vector<float>, 2> y;
    for (auto &output : y)
      output.resize(rows * reuse);
    // Engine order: one 16-row block serves every vector before the next block.
    const auto call = [&](int kind) {
      for (std::size_t r = 0; r < rows; r += chunk)
        for (std::size_t v = 0; v < reuse; ++v) {
          alignas(16) float temporary[16];
          if (kind == 0) {
            for (std::size_t i = 0; i < chunk; ++i)
              dot(cols, temporary + i, 0, w.data() + (r + i) * blocks, 0,
                  x.data() + v * blocks, 0, 1);
          } else if (!bridge(cols, chunk, w.data() + r * blocks,
                             blocks * sizeof(Q5Block), x.data() + v * blocks,
                             temporary, 0)) {
            throw std::runtime_error("Candidate unexpectedly fell back");
          }
          std::memcpy(y[kind].data() + v * rows + r, temporary,
                      chunk * sizeof(float));
        }
    };
    Accuracy accuracy;
    const auto check = [&](int kind) {
      if (kind &&
          std::memcmp(y[0].data(), y[1].data(), y[0].size() * sizeof(float)))
        throw std::runtime_error("Output differs from original bytes");
    };
    call(0);
    // The independent oracle covers a fixed prefix; the candidate must then
    // match the original byte for byte over the whole output.
    for (std::size_t v = 0; v < reuse; ++v)
      for (std::size_t r = 0; r < std::min<std::size_t>(rows, 64); ++r)
        verify(y[0][v * rows + r],
               oracle(w.data() + r * blocks, x.data() + v * blocks, cols), cols,
               accuracy);
    configure(1, 1);
    call(1);
    check(1);
    const auto proof = snapshot();
    if (proof.handled != reuse * (rows / chunk) || proof.rows != rows * reuse ||
        proof.thread_mask != 1 || proof.disabled || proof.width ||
        proof.stride || proof.invalid)
      throw std::runtime_error("Audited dispatch/row proof failed");
    configure(1, 0);
    const auto batch = [&](int kind, std::size_t iterations) {
      const auto start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i)
        call(kind);
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          Clock::now() - start)
                          .count();
      const auto audit = snapshot();
      if (audit.handled || audit.rows || audit.disabled || audit.width ||
          audit.stride || audit.invalid || audit.thread_mask)
        throw std::runtime_error("Audit counter changed during formal timing");
      if (ns <= 0)
        throw std::runtime_error("Clock resolution");
      return ns;
    };
    std::array<std::size_t, 2> iterations{};
    for (int i = 0; i < 2; ++i) {
      const int kind = (seed + i) % 2;
      iterations[kind] = 1;
      while (batch(kind, iterations[kind]) <
             static_cast<std::int64_t>(min_ms * 1000000)) {
        if (iterations[kind] >= (1U << 24))
          throw std::runtime_error("Calibration did not converge");
        iterations[kind] *= 2;
      }
      check(kind);
    }
    std::vector<std::array<std::int64_t, 2>> samples(repetitions);
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
      const int first = rep % 2;
      samples[rep][first] = batch(first, iterations[first]);
      samples[rep][1 - first] = batch(1 - first, iterations[1 - first]);
      check(1);
    }
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"benchmark\":\"q5_reuse\","
                 "\"data_kind\":\"measured_local\",\"correctness_passed\":true,"
                 "\"isolated_namespaces\":true,\"library_identity_passed\":true,"
                 "\"bridge_bitwise_equal\":true,\"audit_during_timing\":false,"
                 "\"dispatch_proof_passed\":true,\"threads\":1,\"cols\":"
              << cols << ",\"chunk\":16,\"variant\":\"" << variant
              << "\",\"rows\":" << rows << ",\"reuse\":" << reuse
              << ",\"weight_bytes\":" << w.size() * sizeof(Q5Block)
              << ",\"seed\":" << seed << ",\"repetitions\":" << repetitions
              << ",\"min_sample_ms\":" << min_ms
              << ",\"max_abs_error\":" << accuracy.max_abs_error
              << ",\"max_bound_fraction\":" << accuracy.max_bound_fraction
              << ",\"variants\":{";
    for (int kind = 0; kind < 2; ++kind) {
      if (kind)
        std::cout << ',';
      std::cout << '"' << names[kind]
                << "\":{\"iterations\":" << iterations[kind] << ",\"checksum\":"
                << std::accumulate(y[kind].begin(), y[kind].end(), 0.0) << '}';
    }
    std::cout << "},\"pairs\":[";
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
      if (rep)
        std::cout << ',';
      std::cout << "{\"first\":\"" << names[rep % 2] << "\",\"batch_ns\":{"
                << "\"original\":" << samples[rep][0]
                << ",\"candidate\":" << samples[rep][1] << "}}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "q5-reuse-bench: " << error.what() << '\n';
    return 1;
  }
}
