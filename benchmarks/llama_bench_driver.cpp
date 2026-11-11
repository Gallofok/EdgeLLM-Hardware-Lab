// Preserve the official benchmark's token stream, warmup and timing regions.
// Only configure the integration before main; audit output is outside timing.
#define main edgellm_official_bench_main
#include EDGELLM_LLAMA_BENCH_SOURCE
#undef main
#ifdef EDGELLM_Q5_HYBRID
#include "integration/q5_ggml_bridge_hybrid.h"
#else
#include "integration/q5_ggml_bridge.h"
#endif
#include <cstdlib>
#include <cstring>
#include <iostream>
int main(int argc, char **argv) {
  const auto yes = [](const char *name) {
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
  };
  const bool enabled = yes("EDGELLM_Q5_ROW4");
  const bool audit = yes("EDGELLM_Q5_AUDIT");
#ifdef EDGELLM_Q5_HYBRID
  edgellm_q5_configure_hybrid(enabled, audit);
#else
  edgellm_q5_configure(enabled, audit);
#endif
  const int result = edgellm_official_bench_main(argc, argv);
  if (audit) {
    const auto a = edgellm_q5_snapshot();
    std::cerr << "EDGELLM_AUDIT {\"enabled\":" << (enabled ? "true" : "false")
              << ",\"audit\":true,\"handled\":" << a.handled
              << ",\"rows\":" << a.rows << ",\"disabled\":" << a.disabled
              << ",\"thread_mask\":" << a.thread_mask;
#ifdef EDGELLM_Q5_HYBRID
    const auto h = edgellm_q5_hybrid_snapshot();
    std::cerr << ",\"single_vector_calls\":" << h.single_vector_calls
              << ",\"single_vector_rows\":" << h.single_vector_rows;
#endif
    std::cerr << "}\n";
  }
  return result;
}
