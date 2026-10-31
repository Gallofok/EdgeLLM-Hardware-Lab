#include "integration/q5_ggml_bridge.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
struct Step {
  std::vector<float> logits;
  llama_token greedy;
};
void run(llama_model *model, const char *prompt, int prompt_id, int threads) {
  auto *vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  std::vector<llama_token> tokens(128);
  const int n =
      llama_tokenize(vocab, prompt, int(std::char_traits<char>::length(prompt)),
                     tokens.data(), 128, true, false);
  require(n > 0 && n < 96, "Public prompt tokenization failed");
  tokens.resize(n);
  std::vector<Step> baseline;
  // Predeclared feasibility gate; this is not a perplexity/quality benchmark.
  constexpr double atol = 1e-3, rtol = 1e-4;
  for (int mode = 0; mode < 2; ++mode) {
    auto p = llama_context_default_params();
    p.n_ctx = 128;
    p.n_batch = 128;
    p.n_ubatch = 128;
    p.n_threads = threads;
    p.n_threads_batch = threads;
    p.offload_kqv = false;
    p.op_offload = false;
    p.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model, p), llama_free);
    require(bool(ctx), "Model context initialization failed");
    std::vector<llama_token> input = tokens;
    for (int step = 0; step < 9; ++step) {
#ifdef EDGELLM_INTEGRATED
      edgellm_q5_configure(mode, 1);
#endif
      require(llama_decode(ctx.get(), llama_batch_get_one(input.data(),
                                                          int(input.size()))) ==
                  0,
              "Model decode failed");
      const float *logits = llama_get_logits_ith(ctx.get(), -1);
      require(logits, "No logits returned");
      const auto greedy =
          llama_token(std::max_element(logits, logits + n_vocab) - logits);
      double max_abs = 0, max_fraction = 0, squared = 0;
      for (int i = 0; i < n_vocab; ++i) {
        require(std::isfinite(logits[i]), "Nonfinite model logit");
        if (mode) {
          const double error =
              std::abs(double(logits[i]) - baseline[step].logits[i]);
          max_abs = std::max(max_abs, error);
          squared += error * error;
          max_fraction = std::max(
              max_fraction,
              error / (atol + rtol * std::abs(baseline[step].logits[i])));
        }
      }
      if (!mode)
        baseline.push_back(
            {std::vector<float>(logits, logits + n_vocab), greedy});
      require(max_fraction <= 1, "Model logits exceeded predeclared tolerance");
      require(greedy == baseline[step].greedy,
              "Greedy token diverged from baseline");
      edgellm_q5_audit a{};
#ifdef EDGELLM_INTEGRATED
      a = edgellm_q5_snapshot();
      require(!mode || a.handled > 0, "Model graph did not execute row4 path");
      require(mode || a.handled == 0, "Disabled model path executed row4");
#endif
      std::cout << "{\"schema\":1,\"fixture\":false,\"prompt\":" << prompt_id
                << ",\"threads\":" << threads << ",\"mode\":" << mode
                << ",\"step\":" << step << ",\"input_tokens\":" << input.size()
                << ",\"vocab\":" << n_vocab << ",\"greedy_token\":" << greedy
                << ",\"max_abs_error\":" << max_abs
                << ",\"rms_error\":" << std::sqrt(squared / n_vocab)
                << ",\"max_bound_fraction\":" << max_fraction
                << ",\"atol\":" << atol << ",\"rtol\":" << rtol
                << ",\"handled\":" << a.handled
                << ",\"handled_rows\":" << a.rows
                << ",\"disabled\":" << a.disabled
                << ",\"thread_mask\":" << a.thread_mask << ",\"passed\":true}\n"
                << std::flush;
      // Teacher-force the baseline greedy prefix so every comparison has the
      // same history, even if a future candidate fails the argmax gate.
      input = {baseline[step].greedy};
    }
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 2, "Usage: test-model-row4 MODEL.gguf");
    std::cout << std::setprecision(17);
    llama_backend_init();
    auto p = llama_model_default_params();
    p.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], p), llama_model_free);
    require(bool(model), "Model load failed");
    int id = 0;
    for (const char *prompt :
         {"The capital of France is",
          "Explain why the sky appears blue in one sentence."}) {
      for (int threads : {1, 4})
        run(model.get(), prompt, id, threads);
      ++id;
    }
    model.reset();
    llama_backend_free();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
