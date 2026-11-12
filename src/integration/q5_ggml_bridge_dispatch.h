#pragma once
#include "q5_ggml_bridge.h"
#ifdef __cplusplus
extern "C" {
#endif
// Dispatch candidate. A weight block reused by several activation vectors goes
// to the plain two-row loop. A block read once goes to the hinted loop while
// the graph runs on fewer threads than the machine has processors, and stays on
// the original dot otherwise.
struct edgellm_q5_dispatch_audit {
  uint64_t hinted_calls, plain_calls, declined_calls, declined_rows, processors;
};
// Same contract as edgellm_q5_configure; also resets the dispatch counters.
EDGELLM_EXPORT void edgellm_q5_configure_dispatch(int enabled, int audit);
EDGELLM_EXPORT struct edgellm_q5_dispatch_audit
edgellm_q5_dispatch_snapshot(void);
// thread_count is the number of threads computing the node; block_vectors is
// the number of activation vectors applied to this weight block. Declines
// before writing output.
EDGELLM_EXPORT int edgellm_q5_try_row4_dispatch(int64_t cols, int64_t rows,
                                                const void *weights,
                                                size_t row_stride,
                                                const void *activation,
                                                float *output, int thread_index,
                                                int thread_count,
                                                int64_t block_vectors);
#ifdef __cplusplus
}
#endif
