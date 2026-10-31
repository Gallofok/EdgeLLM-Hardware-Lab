# Kernel extension boundary

Phase 1 uses the unmodified pinned GGML CPU backend. `scalar/` and `neon/`
are reserved for Phase 2 GEMV correctness and performance work. The root CMake
interface target is an extension point; it does not currently modify llama.cpp.
