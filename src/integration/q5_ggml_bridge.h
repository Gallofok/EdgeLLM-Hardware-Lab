#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(__GNUC__)
#define EDGELLM_EXPORT __attribute__((visibility("default")))
#else
#define EDGELLM_EXPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Configure/reset only when no GGML graph is in flight. Disabled by default;
// EDGELLM_Q5_ROW4=1 opts in for ordinary executables at process startup.
// enabled=1 preserves upstream accumulation; enabled=2 is ONLY the rejected
// FMA prototype diagnostic. Ordinary startup opt-in always selects profile1.
struct edgellm_q5_audit {
  uint64_t handled, rows, disabled, width, stride, invalid, thread_mask;
};
EDGELLM_EXPORT void edgellm_q5_configure(int enabled, int audit);
EDGELLM_EXPORT struct edgellm_q5_audit edgellm_q5_snapshot(void);
// Called only by the pinned Q5_0/Q8_0, nrc=1 MUL_MAT fallback. Reject before
// writing output; unsupported layouts retain the original dot implementation.
EDGELLM_EXPORT int edgellm_q5_try_row4(int64_t cols, int64_t rows,
                                       const void *weights, size_t row_stride,
                                       const void *activation, float *output,
                                       int thread_index);
#ifdef __cplusplus
}
#endif
