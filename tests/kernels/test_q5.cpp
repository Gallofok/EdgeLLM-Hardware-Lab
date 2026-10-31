#include "../../benchmarks/q5_support.h"

#include <iostream>
#include <memory>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
  bool caught = false;
  try { f(); } catch (const std::invalid_argument&) { caught = true; }
  require(caught, "Invalid input was accepted");
}
}
int main(int argc, char** argv) {
  try {
    q5bench::Dot upstream = nullptr;
    std::unique_ptr<q5bench::Upstream> loaded;
    if (argc == 3 && std::string(argv[1]) == "--upstream") {
      loaded = std::make_unique<q5bench::Upstream>(argv[2]); upstream = loaded->dot;
    } else if (argc != 1) throw std::invalid_argument("Usage: test-q5 [--upstream LIB]");
    using namespace q5bench;
    // Exhaust every finite binary16 bit pattern against the independent formula.
    for (std::uint32_t bits = 0; bits < 65536; ++bits) {
      if (((bits >> 10) & 31) != 31)
        require(half_to_float(static_cast<std::uint16_t>(bits)) == oracle_half(static_cast<std::uint16_t>(bits)), "Half conversion mismatch");
    }
    require(std::signbit(half_to_float(0x8000)), "Negative zero lost");
    require(std::isinf(half_to_float(0x7c00)) && std::isnan(half_to_float(0x7e01)), "Nonfinite half decode");
    // A single active Q8 lane isolates each Q5 nibble and fifth bit.
    for (int lane = 0; lane < 32; ++lane) for (int q = -16; q < 16; ++q) {
      Q5Block w{}; Q8Block x{}; w.half_scale = x.half_scale = 0x3c00;
      const auto encoded = static_cast<unsigned>(q + 16);
      w.low[lane % 16] = static_cast<std::uint8_t>((encoded & 15U) << (lane < 16 ? 0 : 4));
      w.high[lane / 8] = static_cast<std::uint8_t>((encoded >> 4) << (lane % 8));
      float decoded[32]; decode_q5(w, decoded);
      require(decoded[lane] == q, "Q5 signed bit/nibble decode failed");
      for (const int activation : {-128, -127, -1, 0, 1, 127}) {
        x.values[lane] = static_cast<std::int8_t>(activation);
        require(dot_q5_q8_reference(&w, &x, 32) == q * activation, "Exact integer fixture failed");
        decode_q8(x, decoded); require(decoded[lane] == activation, "Q8 decode failed");
        if (upstream) {
          float value; upstream(32, &value, 0, &w, 0, &x, 0, 1);
          require(value == q * activation, "Upstream single-lane golden fixture failed");
        }
      }
    }
    for (const std::size_t blocks : {1U, 2U, 3U, 17U, 18U, 19U, 48U}) {
      std::vector<Q5Block> w(blocks + 1); std::vector<Q8Block> x(blocks + 1);
      for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        fill(w, x, seed);
        // One-block offset requires only 2-byte alignment (not SIMD alignment).
        const auto expected = oracle(w.data() + 1, x.data() + 1, blocks * 32);
        Accuracy accuracy;
        verify(dot_q5_q8_reference(w.data() + 1, x.data() + 1, blocks * 32), expected, blocks * 32, accuracy);
        if (upstream) { float value; upstream(static_cast<int>(blocks * 32), &value, 0, w.data() + 1, 0, x.data() + 1, 0, 1); verify(value, expected, blocks * 32, accuracy); }
      }
    }
    require(dot_q5_q8_reference(nullptr, nullptr, 0) == 0, "Zero width read input");
    Q5Block w{}; Q8Block x{};
    rejects([&] { dot_q5_q8_reference(&w, &x, 31); });
    rejects([&] { dot_q5_q8_reference(nullptr, &x, 32); });
    rejects([&] { dot_q5_q8_reference(&w, nullptr, 32); });
    rejects([&] { dot_q5_q8_reference(&w, &x, std::numeric_limits<std::size_t>::max() - 31); });
    rejects([&] { decode_q5(w, nullptr); }); rejects([&] { decode_q8(x, nullptr); });
    std::cout << "Q5/Q8 correctness passed: exhaustive half/bit fixtures, odd/even block tails, decoded FP64 oracle; upstream=" << (upstream ? "checked" : "not loaded") << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << "test-q5: " << e.what() << '\n'; return 1; }
}
