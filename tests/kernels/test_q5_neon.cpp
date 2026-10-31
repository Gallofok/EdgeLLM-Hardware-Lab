#include "../../benchmarks/q5_support.h"
#include "kernels/quantized/q5_neon.h"
#include <iostream>
#include <memory>

namespace {
using namespace q5bench;
using Matrix = decltype(&matvec_q5_q8_neon_row1);
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, "Invalid candidate input accepted");
}
void matrix_case(Matrix kernel, Dot upstream, std::size_t rows,
                 std::size_t blocks, std::size_t vectors) {
  std::vector<Q5Block> w(rows * blocks + 1);
  std::vector<Q8Block> x(vectors * blocks + 1);
  fill(w, x, static_cast<std::uint32_t>(rows * 131 + blocks * 17 + vectors));
  std::vector<float> y(rows * vectors + 2, -98765);
  kernel(w.data() + 1, x.data() + 1, y.data() + 1, rows, blocks * 32, vectors);
  Accuracy accuracy;
  for (std::size_t v = 0; v < vectors; ++v)
    for (std::size_t r = 0; r < rows; ++r) {
      const auto *weights = w.data() + 1 + r * blocks;
      const auto *activation = x.data() + 1 + v * blocks;
      const auto expected = oracle(weights, activation, blocks * 32);
      verify(y[1 + v * rows + r], expected, blocks * 32, accuracy);
      if (upstream) {
        float value;
        upstream(static_cast<int>(blocks * 32), &value, 0, weights, 0,
                 activation, 0, 1);
        verify(value, expected, blocks * 32, accuracy);
      }
    }
  require(y.front() == -98765 && y.back() == -98765,
          "Candidate output guard overwritten");
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::unique_ptr<Upstream> upstream;
    if (argc == 3 && std::string(argv[1]) == "--upstream")
      upstream = std::make_unique<Upstream>(argv[2]);
    else if (argc != 1)
      throw std::invalid_argument("Usage: test-q5-neon [--upstream LIB]");
    for (Matrix kernel : {matvec_q5_q8_neon_row1, matvec_q5_q8_neon_row4}) {
      for (int lane = 0; lane < 32; ++lane)
        for (int encoded = 0; encoded < 32; ++encoded) {
          Q5Block w[5]{};
          Q8Block x{};
          x.half_scale = 0x3c00;
          for (int r = 0; r < 5; ++r) {
            const auto value = static_cast<unsigned>((encoded + r * 7) % 32);
            w[r].half_scale = 0x3c00;
            w[r].low[lane % 16] =
                static_cast<std::uint8_t>((value & 15) << (lane < 16 ? 0 : 4));
            w[r].high[lane / 8] =
                static_cast<std::uint8_t>((value >> 4) << (lane % 8));
          }
          for (const int activation : {-128, -127, -1, 0, 1, 127}) {
            x.values[lane] = static_cast<std::int8_t>(activation);
            float y[5];
            kernel(w, &x, y, 5, 32, 1);
            for (int r = 0; r < 5; ++r)
              require(y[r] == (((encoded + r * 7) % 32) - 16) * activation,
                      "Candidate fifth-bit/lane/row golden check failed");
          }
        }
      // Exhaust candidate hardware scale conversion, in both operand positions.
      for (std::uint32_t bits = 0; bits < 65536; ++bits) {
        if (((bits >> 10) & 31) == 31)
          continue;
        Q5Block w{};
        Q8Block x{};
        for (auto &b : w.low)
          b = 0x11;
        for (auto &b : w.high)
          b = 255; // All integer weights ==1.
        for (auto &b : x.values)
          b = 1;
        const auto expected = static_cast<float>(
            32 * oracle_half(static_cast<std::uint16_t>(bits)));
        float y;
        w.half_scale = static_cast<std::uint16_t>(bits);
        x.half_scale = 0x3c00;
        kernel(&w, &x, &y, 1, 32, 1);
        require(y == expected, "Candidate weight scale conversion failed");
        x.half_scale = static_cast<std::uint16_t>(bits);
        w.half_scale = 0x3c00;
        kernel(&w, &x, &y, 1, 32, 1);
        require(y == expected, "Candidate activation scale conversion failed");
      }
      for (const std::size_t rows : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U})
        for (const std::size_t blocks : {1U, 2U, 3U, 17U, 18U, 19U})
          for (const std::size_t vectors : {1U, 3U, 8U})
            matrix_case(kernel, upstream ? upstream->dot : nullptr, rows,
                        blocks, vectors);
      // Max-magnitude int16 folding: four products/lane remain bounded by8192.
      Q5Block w{};
      Q8Block x{};
      w.half_scale = x.half_scale = 0x3c00;
      for (auto &v : x.values)
        v = -128; // Q5 defaults to -16.
      float y[2] = {123, 456};
      kernel(&w, &x, y, 1, 32, 1);
      require(y[0] == 65536 && y[1] == 456,
              "Candidate integer folding overflow");
      kernel(nullptr, nullptr, nullptr, 0, 33, 1);
      kernel(nullptr, nullptr, nullptr, 1, 33, 0);
      kernel(nullptr, nullptr, y, 2, 0, 1);
      require(y[0] == 0 && y[1] == 0, "Zero width output");
      y[0] = 123;
      rejects([&] { kernel(&w, &x, y, 1, 33, 1); });
      rejects([&] { kernel(nullptr, &x, y, 1, 32, 1); });
      rejects([&] { kernel(&w, nullptr, y, 1, 32, 1); });
      rejects([&] { kernel(&w, &x, nullptr, 1, 32, 1); });
      rejects([&] {
        kernel(&w, &x, y, std::numeric_limits<std::size_t>::max(), 32, 2);
      });
      rejects([&] {
        kernel(&w, &x, y, 2, std::numeric_limits<std::size_t>::max() - 31, 1);
      });
      require(y[0] == 123, "Invalid candidate call wrote output");
    }
    std::cout << "Q5 NEON row1/row4: golden, half, tails, oracle and bounds "
                 "passed; candidate+upstream="
              << (upstream ? "checked" : "not loaded") << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "test-q5-neon: " << e.what() << '\n';
    return 1;
  }
}
