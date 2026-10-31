#include "kernels/quantized/q5_neon.h"
#include <algorithm>
#include <arm_neon.h>
#include <limits>
#include <stdexcept>

namespace edgellm::quantized {
namespace {
void validate(const Q5Block *w, const Q8Block *x, float *y, std::size_t rows,
              std::size_t cols, std::size_t vectors) {
  if (rows == 0 || vectors == 0)
    return;
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!y || cols % 32 || rows > limit / sizeof(float) / vectors)
    throw std::invalid_argument("Invalid Q5 NEON output or shape");
  if (cols && (!w || !x || cols / 32 > limit / sizeof(Q5Block) / rows ||
               cols / 32 > limit / sizeof(Q8Block) / vectors))
    throw std::invalid_argument("Invalid Q5 NEON inputs or extent");
}

// Expand each high-bit byte with lane masks rather than a scalar lookup table.
// Subtract16 exactly when the fifth bit is clear, producing [-16,15].
inline int32x4_t integer_dot(const Q5Block &w, int8x16_t xlo, int8x16_t xhi) {
  const std::uint8_t lane_bits[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  const auto bits = vld1_u8(lane_bits);
  const auto low = vld1q_u8(w.low);
  const auto lo_present = vcombine_u8(vtst_u8(vdup_n_u8(w.high[0]), bits),
                                      vtst_u8(vdup_n_u8(w.high[1]), bits));
  const auto hi_present = vcombine_u8(vtst_u8(vdup_n_u8(w.high[2]), bits),
                                      vtst_u8(vdup_n_u8(w.high[3]), bits));
  const auto qlo =
      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(low, vdupq_n_u8(15))),
               vreinterpretq_s8_u8(vbicq_u8(vdupq_n_u8(16), lo_present)));
  const auto qhi =
      vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(low, 4)),
               vreinterpretq_s8_u8(vbicq_u8(vdupq_n_u8(16), hi_present)));
  auto a = vmull_s8(vget_low_s8(qlo), vget_low_s8(xlo));
  auto b = vmull_high_s8(qlo, xlo);
  a = vmlal_s8(a, vget_low_s8(qhi), vget_low_s8(xhi));
  b = vmlal_high_s8(b, qhi, xhi);
  // Four signed products/lane: magnitude <=4*16*128=8192, fits int16.
  return vpaddlq_s16(vaddq_s16(a, b));
}

template <int Rows>
void tile(const Q5Block *w, const Q8Block *x, float *y, std::size_t blocks) {
  float32x4_t acc[Rows];
  for (int r = 0; r < Rows; ++r)
    acc[r] = vdupq_n_f32(0);
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto xlo = vld1q_s8(x[b].values), xhi = vld1q_s8(x[b].values + 16);
    std::uint16_t scales[4] = {};
    for (int r = 0; r < Rows; ++r)
      scales[r] = w[r * blocks + b].half_scale;
    const auto ws = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(scales)));
    const auto xs =
        vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(x[b].half_scale)));
    const auto combined = vmulq_f32(ws, xs);
    // Compile-time lanes enable scalar-lane vector FMA; all row tiles share X.
    acc[0] = vfmaq_laneq_f32(acc[0], vcvtq_f32_s32(integer_dot(w[b], xlo, xhi)),
                             combined, 0);
    if constexpr (Rows > 1)
      acc[1] = vfmaq_laneq_f32(
          acc[1], vcvtq_f32_s32(integer_dot(w[blocks + b], xlo, xhi)), combined,
          1);
    if constexpr (Rows > 2)
      acc[2] = vfmaq_laneq_f32(
          acc[2], vcvtq_f32_s32(integer_dot(w[2 * blocks + b], xlo, xhi)),
          combined, 2);
    if constexpr (Rows > 3)
      acc[3] = vfmaq_laneq_f32(
          acc[3], vcvtq_f32_s32(integer_dot(w[3 * blocks + b], xlo, xhi)),
          combined, 3);
  }
  for (int r = 0; r < Rows; ++r)
    y[r] = vaddvq_f32(acc[r]);
}

template <bool Blocked>
void matvec(const Q5Block *w, const Q8Block *x, float *y, std::size_t rows,
            std::size_t cols, std::size_t vectors) {
  validate(w, x, y, rows, cols, vectors);
  if (rows == 0 || vectors == 0)
    return;
  if (cols == 0) {
    std::fill(y, y + rows * vectors, 0.0F);
    return;
  }
  const auto blocks = cols / 32;
  for (std::size_t v = 0; v < vectors; ++v) {
    std::size_t r = 0;
    if constexpr (Blocked) {
      for (; rows - r >= 4; r += 4)
        tile<4>(w + r * blocks, x + v * blocks, y + v * rows + r, blocks);
      switch (rows - r) {
      case 3:
        tile<3>(w + r * blocks, x + v * blocks, y + v * rows + r, blocks);
        break;
      case 2:
        tile<2>(w + r * blocks, x + v * blocks, y + v * rows + r, blocks);
        break;
      case 1:
        tile<1>(w + r * blocks, x + v * blocks, y + v * rows + r, blocks);
        break;
      }
    } else {
      for (; r < rows; ++r)
        tile<1>(w + r * blocks, x + v * blocks, y + v * rows + r, blocks);
    }
  }
}
} // namespace
void matvec_q5_q8_neon_row1(const Q5Block *w, const Q8Block *x, float *y,
                            std::size_t rows, std::size_t cols,
                            std::size_t vectors) {
  matvec<false>(w, x, y, rows, cols, vectors);
}
void matvec_q5_q8_neon_row4(const Q5Block *w, const Q8Block *x, float *y,
                            std::size_t rows, std::size_t cols,
                            std::size_t vectors) {
  matvec<true>(w, x, y, rows, cols, vectors);
}
} // namespace edgellm::quantized
