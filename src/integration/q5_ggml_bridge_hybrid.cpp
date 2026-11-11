// Hybrid dispatch candidate. The loop is the measured two-block bridge, compiled
// from its unchanged source; this file adds only the dispatch decision.
// No performance measurement exists for this candidate.
#include "q5_ggml_bridge_two_block.cpp"

#include "q5_ggml_bridge_hybrid.h"

namespace {
std::atomic<uint64_t> single_vector_calls{0}, single_vector_rows{0};
} // namespace
extern "C" {
void edgellm_q5_configure_hybrid(int on, int collect) {
  edgellm_q5_configure(on, collect);
  single_vector_calls = 0;
  single_vector_rows = 0;
}
edgellm_q5_hybrid_audit edgellm_q5_hybrid_snapshot() {
  return {single_vector_calls.load(), single_vector_rows.load()};
}
int edgellm_q5_try_row4_vectors(int64_t cols, int64_t rows, const void *w,
                                size_t row_stride, const void *x, float *y,
                                int ith, int64_t block_vectors) {
  // A disabled bridge must keep reporting through its own counter.
  if (block_vectors < 2 && enabled.load(std::memory_order_relaxed)) {
    if (audit.load(std::memory_order_relaxed)) {
      single_vector_calls.fetch_add(1, std::memory_order_relaxed);
      single_vector_rows.fetch_add(rows > 0 ? uint64_t(rows) : 0,
                                   std::memory_order_relaxed);
    }
    return 0;
  }
  return edgellm_q5_try_row4(cols, rows, w, row_stride, x, y, ith);
}
const char *edgellm_q5_dispatch_policy() { return "multi-vector"; }
}
