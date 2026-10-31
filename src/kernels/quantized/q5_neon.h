#pragma once
#include "kernels/quantized/q5.h"

#ifdef EDGELLM_HAS_Q5_NEON
namespace edgellm::quantized {
// Packed W[rows][cols/32], X[vectors][cols/32], Y[vectors][rows].
// Original GGML Q5_0/Q8_0 blocks; no repacking, allocation or threading.
// rows==0 or vectors==0 is a no-op. cols==0 writes zeros without reading W/X.
// Otherwise cols must be a multiple of32; valid arrays and disjoint output are
// caller obligations. Struct alignment suffices; finite, non-overflowing input
// arithmetic/default IEEE FP environment required. Invalid shape/null/overflow
// throws before writes. Explicit NEON FMA and changed reduction order are
// allowed. row1 is an ablation: same decoding/integer reduction, no cross-row
// reuse.
void matvec_q5_q8_neon_row1(const Q5Block *w, const Q8Block *x, float *y,
                            std::size_t rows, std::size_t cols,
                            std::size_t vectors);
void matvec_q5_q8_neon_row4(const Q5Block *w, const Q8Block *x, float *y,
                            std::size_t rows, std::size_t cols,
                            std::size_t vectors);
} // namespace edgellm::quantized
#endif
