#pragma once
#include "q5_ggml_bridge.h"
#ifdef __cplusplus
extern "C" {
#endif
// Thread-adaptive candidate. The two-row loop issues prefetch hints only when
// the graph runs on fewer threads than the machine has online processors.
struct edgellm_q5_adaptive_audit {
  uint64_t hinted_calls, plain_calls, processors;
};
// Same contract as edgellm_q5_configure; also resets the adaptive counters.
EDGELLM_EXPORT void edgellm_q5_configure_adaptive(int enabled, int audit);
EDGELLM_EXPORT struct edgellm_q5_adaptive_audit
edgellm_q5_adaptive_snapshot(void);
// thread_count is the number of threads computing the current graph node.
EDGELLM_EXPORT int edgellm_q5_try_row4_threads(int64_t cols, int64_t rows,
                                               const void *weights,
                                               size_t row_stride,
                                               const void *activation,
                                               float *output, int thread_index,
                                               int thread_count);
#ifdef __cplusplus
}
#endif
