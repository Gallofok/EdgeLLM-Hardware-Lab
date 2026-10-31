#include "ggml-cpu.h"
#include "ggml.h"
#include "integration/q5_ggml_bridge.h"
#include "q5_support.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
using namespace edgellm::quantized;
void require(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
struct Case {
  int cols, rows, vectors, w2, w3, x2, x3;
  bool f32, xs, wp, ys;
};
void run(const Case &c, int threads, int id) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
      ggml_init({32 * 1024 * 1024, nullptr, false}), ggml_free);
  require(bool(ctx), "Cannot allocate GGML test context");
  auto *ws = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q5_0, c.cols + 32 * c.wp,
                                c.rows, c.w2, c.w3);
  auto *w = ggml_view_4d(ctx.get(), ws, c.cols, c.rows, c.w2, c.w3, ws->nb[1],
                         ws->nb[2], ws->nb[3], 0);
  const auto xtype = c.f32 ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
  auto *xs =
      ggml_new_tensor_4d(ctx.get(), xtype, c.cols,
                         c.xs ? 2 * c.vectors + 1 : c.vectors, c.x2, c.x3);
  auto *x = ggml_view_4d(ctx.get(), xs, c.cols, c.vectors, c.x2, c.x3,
                         xs->nb[1] * (c.xs ? 2 : 1), xs->nb[2], xs->nb[3],
                         c.xs ? xs->nb[1] : 0);
  auto *y = ggml_mul_mat(ctx.get(), w, x);
  // Redirect output into a larger allocation to exercise column/batch strides.
  auto *ys = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                                c.rows + (c.ys ? 3 : 0), c.vectors, c.x2, c.x3);
  y->data = ys->data;
  for (int d = 1; d < 4; ++d)
    y->nb[d] = ys->nb[d];
  std::vector<Q5Block> weights(c.rows * c.cols / 32 * c.w2 * c.w3);
  std::vector<Q8Block> activations(c.vectors * c.cols / 32 * c.x2 * c.x3);
  q5bench::fill(weights, activations, 91 + id, false);
  const int blocks = c.cols / 32;
  for (int b3 = 0; b3 < c.w3; ++b3)
    for (int b2 = 0; b2 < c.w2; ++b2)
      for (int r = 0; r < c.rows; ++r)
        std::memcpy(static_cast<char *>(w->data) + b3 * w->nb[3] +
                        b2 * w->nb[2] + r * w->nb[1],
                    &weights[((b3 * c.w2 + b2) * c.rows + r) * blocks],
                    blocks * sizeof(Q5Block));
  std::vector<float> f(c.cols);
  for (int b3 = 0; b3 < c.x3; ++b3)
    for (int b2 = 0; b2 < c.x2; ++b2)
      for (int v = 0; v < c.vectors; ++v) {
        auto *packed =
            &activations[((b3 * c.x2 + b2) * c.vectors + v) * blocks];
        char *dest = static_cast<char *>(x->data) + b3 * x->nb[3] +
                     b2 * x->nb[2] + v * x->nb[1];
        if (c.f32) {
          for (int i = 0; i < c.cols; ++i)
            f[i] =
                float(((i * 17 + v * 23 + b2 * 7 + b3 * 31) % 257) - 128) / 127;
          std::memcpy(dest, f.data(), f.size() * sizeof(float));
          ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)
              ->from_float(f.data(), packed, c.cols);
        } else
          std::memcpy(dest, packed, blocks * sizeof(Q8Block));
      }
  auto *graph = ggml_new_graph(ctx.get());
  ggml_build_forward_expand(graph, y);
  q5bench::Accuracy accuracy;
  std::vector<float> baseline;
  for (int mode = 0; mode < 2; ++mode) {
#ifdef EDGELLM_INTEGRATED
    edgellm_q5_configure(mode, 1);
#endif
    std::fill_n(static_cast<float *>(ys->data), ggml_nelements(ys), -123456.0f);
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, threads) ==
                GGML_STATUS_SUCCESS,
            "GGML graph execution failed");
    std::size_t index = 0;
    for (int b3 = 0; b3 < c.x3; ++b3)
      for (int b2 = 0; b2 < c.x2; ++b2)
        for (int v = 0; v < c.vectors; ++v) {
          const auto *row = reinterpret_cast<const float *>(
              static_cast<const char *>(y->data) + b3 * y->nb[3] +
              b2 * y->nb[2] + v * y->nb[1]);
          for (int r = 0; r < c.rows; ++r, ++index) {
            const auto *ww =
                &weights[(((b3 / (c.x3 / c.w3)) * c.w2 + b2 / (c.x2 / c.w2)) *
                              c.rows +
                          r) *
                         blocks];
            const auto *xx =
                &activations[((b3 * c.x2 + b2) * c.vectors + v) * blocks];
            const auto expected = q5bench::oracle(ww, xx, c.cols);
            q5bench::verify(row[r], expected, c.cols, accuracy);
            if (!mode)
              baseline.push_back(row[r]);
              // Both outputs independently satisfy the oracle; disabled mode
              // must preserve original values bit-for-bit between repeated
              // executions.
#ifndef EDGELLM_INTEGRATED
            else
              require(row[r] == baseline[index],
                      "Default-off path changed output");
#endif
          }
          if (c.ys)
            for (int r = c.rows; r < c.rows + 3; ++r)
              require(row[r] == -123456.0f, "Output padding overwritten");
        }
    edgellm_q5_audit a{};
#ifdef EDGELLM_INTEGRATED
    a = edgellm_q5_snapshot();
    if (!mode)
      require(a.handled == 0, "Disabled integration executed");
    // Eligible cases are chosen to reach the GGML fallback even with the
    // original llamafile option enabled. Audit proves actual execution.
    if (mode && c.cols == 576 && !c.wp) {
      require(a.handled > 0, "Eligible graph bypassed the candidate");
      require(a.rows == uint64_t(c.rows * c.vectors * c.x2 * c.x3),
              "Missing or overlapping row partition");
    }
    if (mode && (c.cols != 576 || c.wp))
      require(a.handled == 0, "Unsupported layout reached candidate");
#endif
    std::cout << "{\"schema\":1,\"fixture\":false,\"case\":" << id
              << ",\"threads\":" << threads << ",\"mode\":" << mode
              << ",\"cols\":" << c.cols << ",\"rows\":" << c.rows
              << ",\"vectors\":" << c.vectors << ",\"f32_workspace\":" << c.f32
              << ",\"strided_activation\":" << c.xs
              << ",\"padded_weights\":" << c.wp << ",\"padded_output\":" << c.ys
              << ",\"w2\":" << c.w2 << ",\"w3\":" << c.w3 << ",\"x2\":" << c.x2
              << ",\"x3\":" << c.x3 << ",\"handled\":" << a.handled
              << ",\"handled_rows\":" << a.rows
              << ",\"disabled\":" << a.disabled
              << ",\"width_fallback\":" << a.width
              << ",\"stride_fallback\":" << a.stride
              << ",\"thread_mask\":" << a.thread_mask
              << ",\"max_abs_error\":" << accuracy.max_abs_error
              << ",\"max_bound_fraction\":" << accuracy.max_bound_fraction
              << ",\"passed\":true}\n";
  }
}
} // namespace
int main() {
  try {
    ggml_cpu_init();
    std::cout << std::setprecision(17);
    std::vector<Case> cases;
    for (int rows : {1, 3, 4, 5, 17, 65, 192})
      cases.push_back({576, rows, 1, 1, 1, 1, 1, false, false, false, false});
    cases.push_back({576, 17, 3, 1, 1, 1, 1, true, false, false, true});
    cases.push_back({576, 17, 3, 1, 1, 1, 1, false, true, false, true});
    cases.push_back({576, 17, 3, 2, 1, 4, 2, true, true, false, true});
    cases.push_back({576, 17, 3, 1, 2, 2, 4, false, true, false, true});
    cases.push_back({576, 17, 3, 1, 1, 1, 1, false, true, true, true});
    for (int cols : {32, 96, 608})
      cases.push_back({cols, 17, 3, 1, 1, 1, 1, false, true, false, true});
    int id = 0;
    for (const auto &c : cases) {
      for (int threads : {1, 2, 4})
        run(c, threads, id);
      ++id;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
