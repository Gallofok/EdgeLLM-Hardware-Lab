#include "kernels/quantized/q5.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace edgellm::quantized {

float half_to_float(std::uint16_t bits) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
  std::uint32_t exponent = (bits >> 10) & 31U;
  std::uint32_t fraction = bits & 1023U;
  std::uint32_t result;
  if (exponent == 0) {
    if (fraction == 0) result = sign;
    else {
      int shift = 0;
      while ((fraction & 1024U) == 0) { fraction <<= 1; ++shift; }
      result = sign | (static_cast<std::uint32_t>(113 - shift) << 23) |
               ((fraction & 1023U) << 13);
    }
  } else {
    result = sign | ((exponent == 31 ? 255U : exponent + 112U) << 23) | (fraction << 13);
  }
  float value;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

namespace {
int integer_q5(const Q5Block& block, std::size_t i) {
  const auto nibble = (block.low[i % 16] >> (i < 16 ? 0 : 4)) & 15U;
  const auto high = (block.high[i / 8] >> (i % 8)) & 1U;
  return static_cast<int>(nibble | (high << 4)) - 16;
}
}

void decode_q5(const Q5Block& block, float* output) {
  if (!output) throw std::invalid_argument("Null Q5 decode output");
  const float scale = half_to_float(block.half_scale);
  for (std::size_t i = 0; i < 32; ++i) output[i] = scale * integer_q5(block, i);
}
void decode_q8(const Q8Block& block, float* output) {
  if (!output) throw std::invalid_argument("Null Q8 decode output");
  const float scale = half_to_float(block.half_scale);
  for (std::size_t i = 0; i < 32; ++i) output[i] = scale * block.values[i];
}
float dot_q5_q8_reference(const Q5Block* w, const Q8Block* x, std::size_t n) {
  constexpr auto extent = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (n % 32 || n / 32 > extent / sizeof(Q5Block) || n / 32 > extent / sizeof(Q8Block) ||
      (n && (!w || !x))) throw std::invalid_argument("Invalid Q5/Q8 dot inputs or width");
  float sum = 0;
  for (std::size_t b = 0; b < n / 32; ++b) {
    std::int32_t integer_sum = 0;
    for (std::size_t i = 0; i < 32; ++i)
      integer_sum += integer_q5(w[b], i) * static_cast<int>(x[b].values[i]);
    sum += (half_to_float(w[b].half_scale) * half_to_float(x[b].half_scale)) * integer_sum;
  }
  return sum;
}
}  // namespace edgellm::quantized
