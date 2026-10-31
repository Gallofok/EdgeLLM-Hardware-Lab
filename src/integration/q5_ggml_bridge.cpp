#include "q5_ggml_bridge.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "kernels/quantized/q5_neon.h"
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
bool environment_opt_in() {
  const char *value = std::getenv("EDGELLM_Q5_ROW4");
  return value && std::strcmp(value, "1") == 0;
}
std::atomic<bool> enabled{environment_opt_in()}, audit{false};
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
    matvec_q5_q8_neon_row4(static_cast<const Q5Block *>(w),
                           static_cast<const Q8Block *>(x), y, rows, cols, 1);
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
