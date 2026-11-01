#include "q5_ggml_bridge.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "kernels/quantized/q5_neon.h"
#include <arm_neon.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static_assert(sizeof(block_q5_0) == sizeof(edgellm::quantized::Q5Block));
static_assert(sizeof(block_q8_0) == sizeof(edgellm::quantized::Q8Block));
static_assert(offsetof(block_q5_0, d) ==
              offsetof(edgellm::quantized::Q5Block, half_scale));
static_assert(offsetof(block_q5_0, qh) ==
              offsetof(edgellm::quantized::Q5Block, high));
static_assert(offsetof(block_q5_0, qs) ==
              offsetof(edgellm::quantized::Q5Block, low));
static_assert(offsetof(block_q8_0, qs) ==
              offsetof(edgellm::quantized::Q8Block, values));

namespace {
// Fixed width576 integration profile. Preserve the pinned Cortex-A53
// integer lane grouping, even/odd accumulators and non-fused FP32 operations.
// No performance measurement exists for this new compatibility profile.
using namespace edgellm::quantized;
// Two call sites per row must not turn the baseline's inlined path into calls.
[[gnu::always_inline]] inline int32x4_t
compatible_integer(const Q5Block &w, int8x16_t xlo, int8x16_t xhi) {
  const uint8_t masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  const auto bits = vld1_u8(masks);
  const auto low = vld1q_u8(w.low);
  const auto present0 = vcombine_u8(vtst_u8(vdup_n_u8(w.high[0]), bits),
                                    vtst_u8(vdup_n_u8(w.high[1]), bits));
  const auto present1 = vcombine_u8(vtst_u8(vdup_n_u8(w.high[2]), bits),
                                    vtst_u8(vdup_n_u8(w.high[3]), bits));
  const auto qlo =
      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(low, vdupq_n_u8(15))),
               vreinterpretq_s8_u8(vbicq_u8(vdupq_n_u8(16), present0)));
  const auto qhi =
      vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(low, 4)),
               vreinterpretq_s8_u8(vbicq_u8(vdupq_n_u8(16), present1)));
  auto a = vmull_s8(vget_low_s8(qlo), vget_low_s8(xlo));
  auto b = vmull_high_s8(qlo, xlo);
  a = vmlal_s8(a, vget_low_s8(qhi), vget_low_s8(xhi));
  b = vmlal_high_s8(b, qhi, xhi);
  return vpaddlq_s16(vaddq_s16(a, b));
}
// Candidate loop: the same per-accumulator update sequence, without selecting
// an accumulator on every block. No performance measurement exists yet.
template <int Rows>
[[gnu::always_inline]] inline void
accumulate_block(const Q5Block *w, const Q8Block *x, int b,
                 float32x4_t (&accumulators)[Rows]) {
  const auto xl = vld1q_s8(x[b].values), xh = vld1q_s8(x[b].values + 16);
  uint16_t halves[4] = {};
  for (int r = 0; r < Rows; ++r)
    halves[r] = w[r * 18 + b].half_scale;
  const auto ws = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(halves)));
  const auto xs =
      vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(x[b].half_scale)));
  const auto scales = vmulq_f32(ws, xs);
  float scale[4];
  vst1q_f32(scale, scales);
  for (int r = 0; r < Rows; ++r) {
    const auto product = vmulq_n_f32(
        vcvtq_f32_s32(compatible_integer(w[r * 18 + b], xl, xh)), scale[r]);
    accumulators[r] = vaddq_f32(accumulators[r], product);
  }
}
template <int Rows>
void compatible_tile(const Q5Block *w, const Q8Block *x, float *y) {
  float32x4_t even[Rows], odd[Rows];
  for (int r = 0; r < Rows; ++r)
    even[r] = odd[r] = vdupq_n_f32(0);
  for (int b = 0; b < 18; b += 2) {
    accumulate_block<Rows>(w, x, b, even);
    accumulate_block<Rows>(w, x, b + 1, odd);
  }
  for (int r = 0; r < Rows; ++r)
    y[r] = vaddvq_f32(even[r]) + vaddvq_f32(odd[r]);
}
void compatible_rows(const Q5Block *w, const Q8Block *x, float *y,
                     size_t rows) {
  size_t r = 0;
  for (; r + 4 <= rows; r += 4)
    compatible_tile<4>(w + r * 18, x, y + r);
  switch (rows - r) {
  case 3:
    compatible_tile<3>(w + r * 18, x, y + r);
    break;
  case 2:
    compatible_tile<2>(w + r * 18, x, y + r);
    break;
  case 1:
    compatible_tile<1>(w + r * 18, x, y + r);
    break;
  }
}

bool environment_opt_in() {
  const char *value = std::getenv("EDGELLM_Q5_ROW4");
  return value && std::strcmp(value, "1") == 0;
}
std::atomic<bool> enabled{environment_opt_in()}, audit{false},
    prototype_fma{false};
std::atomic<uint64_t> handled{0}, rows_done{0}, disabled{0}, width{0},
    stride{0}, invalid{0}, thread_mask{0};
void count(std::atomic<uint64_t> &counter) {
  if (audit.load(std::memory_order_relaxed))
    counter.fetch_add(1, std::memory_order_relaxed);
}
} // namespace
extern "C" {
void edgellm_q5_configure(int on, int collect) {
  enabled.store(on != 0);
  prototype_fma.store(on == 2);
  audit.store(collect != 0);
  handled = 0;
  rows_done = 0;
  disabled = 0;
  width = 0;
  stride = 0;
  invalid = 0;
  thread_mask = 0;
}
edgellm_q5_audit edgellm_q5_snapshot() {
  return {handled.load(), rows_done.load(), disabled.load(),   width.load(),
          stride.load(),  invalid.load(),   thread_mask.load()};
}
int edgellm_q5_try_row4(int64_t cols, int64_t rows, const void *w,
                        size_t row_stride, const void *x, float *y, int ith) {
  using namespace edgellm::quantized;
  if (!enabled.load(std::memory_order_relaxed)) {
    count(disabled);
    return 0;
  }
  // First integration deliberately limited to the model's observed Q5 width.
  if (cols != 576) {
    count(width);
    return 0;
  }
  if (row_stride != 18 * sizeof(Q5Block)) {
    count(stride);
    return 0;
  }
  if (rows < 1 || rows > 16 || !w || !x || !y || ith < 0 || ith >= 64 ||
      reinterpret_cast<uintptr_t>(w) % alignof(Q5Block) ||
      reinterpret_cast<uintptr_t>(x) % alignof(Q8Block) ||
      reinterpret_cast<uintptr_t>(y) % alignof(float)) {
    count(invalid);
    return 0;
  }
  try {
    if (prototype_fma.load(std::memory_order_relaxed)) {
      matvec_q5_q8_neon_row4(static_cast<const Q5Block *>(w),
                             static_cast<const Q8Block *>(x), y, rows, cols, 1);
    } else {
      compatible_rows(static_cast<const Q5Block *>(w),
                      static_cast<const Q8Block *>(x), y, rows);
    }
  } catch (...) {
    return 0;
  } // Never unwind across the GGML C ABI.
  if (audit.load(std::memory_order_relaxed)) {
    handled.fetch_add(1, std::memory_order_relaxed);
    rows_done.fetch_add(rows, std::memory_order_relaxed);
    thread_mask.fetch_or(uint64_t{1} << ith, std::memory_order_relaxed);
  }
  return 1;
}
}

// Lets a harness prove which measured library it loaded.
extern "C" EDGELLM_EXPORT const char *edgellm_q5_loop_variant() {
  return "two-block";
}
