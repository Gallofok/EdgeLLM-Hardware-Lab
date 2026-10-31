# Kernel extension boundary

Phase 1 uses the unmodified pinned GGML CPU backend. Phase 2 now provides
`gemv.h` and a single-thread FP32 reference in `scalar/gemv.cpp`, with explicit
scalar compiler settings, independent numerical tests and `gemv-bench`.
`neon/gemv.cpp` provides an ARM64 vector implementation with four independent
accumulator chains and scalar tails. The [paired protocol](../../docs/neon_gemv.md)
defines numerical checks and alternating-order timing. See
[the kernel protocol](../../docs/scalar_gemv.md) for the API, compiler flags,
measurement semantics and limitations. These targets do not modify llama.cpp.
