// Preserve the official benchmark's token stream, warmup and timing regions.
// Only configure the integration before main; audit output is outside timing.
#define main edgellm_official_bench_main
#include EDGELLM_LLAMA_BENCH_SOURCE
#undef main
#include "integration/q5_ggml_bridge.h"
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
  edgellm_q5_configure(enabled, audit);
  const int result = edgellm_official_bench_main(argc, argv);
  if (audit) {
    const auto a = edgellm_q5_snapshot();
    std::cerr << "EDGELLM_AUDIT {\"enabled\":" << (enabled ? "true" : "false")
              << ",\"audit\":true,\"handled\":" << a.handled
              << ",\"rows\":" << a.rows << ",\"disabled\":" << a.disabled
              << ",\"thread_mask\":" << a.thread_mask << "}\n";
  }
  return result;
}
