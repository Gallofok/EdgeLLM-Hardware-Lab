#pragma once
#include "q5_ggml_bridge.h"
#ifdef __cplusplus
extern "C" {
#endif
// Hybrid dispatch candidate. A weight block that serves one activation vector
// stays on the original dot; only multi-vector blocks reach the two-block loop.
struct edgellm_q5_hybrid_audit {
  uint64_t single_vector_calls, single_vector_rows;
};
// Same contract as edgellm_q5_configure; also resets the hybrid counters.
EDGELLM_EXPORT void edgellm_q5_configure_hybrid(int enabled, int audit);
EDGELLM_EXPORT struct edgellm_q5_hybrid_audit edgellm_q5_hybrid_snapshot(void);
// block_vectors is the number of activation vectors the engine applies to this
// weight block before moving on. Declines before writing output.
EDGELLM_EXPORT int edgellm_q5_try_row4_vectors(int64_t cols, int64_t rows,
                                               const void *weights,
                                               size_t row_stride,
                                               const void *activation,
                                               float *output, int thread_index,
                                               int64_t block_vectors);
EDGELLM_EXPORT const char *edgellm_q5_dispatch_policy(void);
#ifdef __cplusplus
}
#endif
