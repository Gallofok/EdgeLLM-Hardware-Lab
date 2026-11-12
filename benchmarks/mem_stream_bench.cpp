// Diagnostic: aggregate sequential read rate of N threads over private buffers
// far larger than L2. A reference for how fast this board can pull lines from
// memory; not an LLM measurement and not a general memory benchmark.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
std::size_t positive(const std::string &s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Positive integer required");
  const auto n = std::stoull(s);
  if (!n || n > 1000000)
    throw std::invalid_argument("Integer out of range");
  return n;
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::size_t threads = 1, megabytes = 64, milliseconds = 1000;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "mem-stream-bench [--threads N --mb N --ms N]\nEach "
                     "thread sums its own buffer front to back until the "
                     "deadline. Reports aggregate bytes read.\n";
        return 0;
      }
      if (++i == argc)
        throw std::invalid_argument("Missing value");
      const auto n = positive(argv[i]);
      if (key == "--threads")
        threads = n;
      else if (key == "--mb")
        megabytes = n;
      else if (key == "--ms")
        milliseconds = n;
      else
        throw std::invalid_argument("Unknown argument");
    }
    if (threads > 16 || megabytes > 256 || milliseconds > 60000)
      throw std::invalid_argument("Invalid size");
    const std::size_t words = megabytes * 1024 * 1024 / sizeof(std::uint64_t);
    std::vector<std::vector<std::uint64_t>> buffers(threads);
    for (std::size_t t = 0; t < threads; ++t) {
      buffers[t].resize(words);
      for (std::size_t i = 0; i < words; ++i) // Touch every page before timing.
        buffers[t][i] = i * 0x9e3779b97f4a7c15ULL + t;
    }
    std::atomic<int> ready{0};
    std::atomic<bool> go{false}, stop{false};
    std::vector<std::uint64_t> passes(threads), sums(threads);
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire))
          std::this_thread::yield();
        std::uint64_t sum = 0, done = 0;
        // Whole passes only, so bytes read are exact.
        while (!stop.load(std::memory_order_relaxed)) {
          const std::uint64_t *p = buffers[t].data();
          for (std::size_t i = 0; i < words; ++i)
            sum += p[i];
          ++done;
        }
        sums[t] = sum;
        passes[t] = done;
      });
    while (ready.load() != static_cast<int>(threads))
      std::this_thread::yield();
    const auto start = Clock::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    stop.store(true);
    for (auto &thread : pool)
      thread.join();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - start)
                        .count();
    std::uint64_t total = 0, checksum = 0;
    for (std::size_t t = 0; t < threads; ++t) {
      total += passes[t];
      checksum ^= sums[t];
    }
    const std::uint64_t bytes = total * words * sizeof(std::uint64_t);
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"benchmark\":\"mem_stream\","
                 "\"data_kind\":\"measured_local\",\"threads\":"
              << threads << ",\"megabytes_per_thread\":" << megabytes
              << ",\"requested_ms\":" << milliseconds << ",\"elapsed_ns\":" << ns
              << ",\"passes\":" << total << ",\"bytes\":" << bytes
              << ",\"bytes_per_second\":" << bytes * 1e9 / ns
              << ",\"checksum\":" << checksum << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "mem-stream-bench: " << error.what() << '\n';
    return 1;
  }
}
