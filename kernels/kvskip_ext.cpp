// Torch bindings for the kvskip decode kernels.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>

#include "kvskip.h"

namespace {

void check_kv(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, " must be a CUDA bf16 tensor");
  TORCH_CHECK(t.dim() == 4, name, " must be [B, Hkv, T, D]");
  TORCH_CHECK(t.stride(3) == 1, name, " must have a contiguous last dim");
  TORCH_CHECK(t.stride(2) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(0) % 8 == 0,
              name, " strides must be multiples of 8 elements (16-byte rows)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

}  // namespace

// Returns [out bf16 [B,Hq,D], lse fp32 [B,Hq], blk_max, blk_sum ([B,Hq,nb] or empty), out_f32 (or empty),
//          P fp16 [B,Hq,nb*64] (write_p) or empty].
std::vector<at::Tensor> decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale,
                               const c10::optional<at::Tensor>& block_idx,
                               const c10::optional<at::Tensor>& block_cnt, bool stats, bool load_v,
                               int64_t num_splits, int64_t stages, bool want_f32, bool write_p) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3, "q must be CUDA bf16 [B, Hq, D]");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0, "q rows must be 16-byte aligned");
  check_kv(k, "k");
  check_kv(v, "v");
  const c10::cuda::CUDAGuard guard(q.device());
  const int B = q.size(0), Hq = q.size(1), D = q.size(2);
  const int Hkv = k.size(1), T = k.size(2);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && k.size(3) == D && v.size(3) == D && v.size(1) == Hkv &&
                  v.size(2) == T,
              "shape mismatch between q, k, v");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(num_splits >= 1, "num_splits >= 1");
  TORCH_CHECK(load_v || stats, "load_v=false only makes sense with stats=true");
  TORCH_CHECK(!write_p || stats, "write_p needs stats=true");
  const bool masked = block_idx.has_value();
  const int nb = (T + kvskip::TILE - 1) / kvskip::TILE;

  auto f32 = q.options().dtype(at::kFloat);
  at::Tensor out = load_v ? at::empty({B, Hq, D}, q.options()) : at::Tensor();
  at::Tensor out32 = (load_v && want_f32) ? at::empty({B, Hq, D}, f32) : at::Tensor();
  at::Tensor lse = at::empty({B, Hq}, f32);
  at::Tensor bmax, bsum;
  if (stats) {
    bmax = at::full({B, Hq, nb}, -INFINITY, f32);
    bsum = at::zeros({B, Hq, nb}, f32);
  }
  at::Tensor pbuf = write_p ? at::empty({B, Hq, static_cast<int64_t>(nb) * kvskip::TILE},
                                        q.options().dtype(at::kHalf))
                             : at::Tensor();
  at::Tensor part_acc, part_ml;
  if (num_splits > 1) {
    part_acc = at::empty({B, Hq, num_splits, D}, f32);
    part_ml = at::empty({B, Hq, num_splits, 2}, f32);
  }

  kvskip::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.q_sb = q.stride(0); p.q_sh = q.stride(1);
  p.k_sb = k.stride(0); p.k_sh = k.stride(1); p.k_st = k.stride(2);
  p.v_sb = v.stride(0); p.v_sh = v.stride(1); p.v_st = v.stride(2);
  p.B = B; p.Hq = Hq; p.Hkv = Hkv; p.T = T;
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  if (masked) {
    TORCH_CHECK(block_cnt.has_value(), "block_cnt is required with block_idx");
    const auto& bi = *block_idx;
    const auto& bc = *block_cnt;
    TORCH_CHECK(bi.is_cuda() && bi.scalar_type() == at::kInt && bi.dim() == 3 && bi.is_contiguous() &&
                    bi.size(0) == B && bi.size(1) == Hkv,
                "block_idx must be contiguous int32 [B, Hkv, max_blocks]");
    TORCH_CHECK(bc.is_cuda() && bc.scalar_type() == at::kInt && bc.is_contiguous() && bc.numel() == B * Hkv,
                "block_cnt must be contiguous int32 [B, Hkv]");
    p.block_idx = bi.data_ptr<int>();
    p.block_cnt = bc.data_ptr<int>();
    p.max_blocks = bi.size(2);
  }
  p.num_splits = num_splits;
  p.part_acc = num_splits > 1 ? part_acc.data_ptr<float>() : nullptr;
  p.part_ml = num_splits > 1 ? part_ml.data_ptr<float>() : nullptr;
  p.out = load_v ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
  p.out_f32 = out32.defined() ? out32.data_ptr<float>() : nullptr;
  p.lse = lse.data_ptr<float>();
  p.blk_max = stats ? bmax.data_ptr<float>() : nullptr;
  p.blk_sum = stats ? bsum.data_ptr<float>() : nullptr;
  p.nb = nb;
  p.t_pad = static_cast<int64_t>(nb) * kvskip::TILE;
  p.p_out = write_p ? reinterpret_cast<__half*>(pbuf.data_ptr()) : nullptr;

  kvskip::launch_decode(p, D, Hq / Hkv, stages, masked, stats, load_v, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, bmax, bsum, out32, pbuf};
}

// Two-pass V-skip, pass 2. P fp16 [B, Hq, nb*64] (pass 1), v [B, Hkv, T, D],
// blk_scale fp32 [B, Hkv, nb, G] = exp(blk_max - lse_kept) (0 = dropped),
// block_idx / block_cnt: kept blocks per (b, kv head). Returns [out bf16, out_f32 or empty].
std::vector<at::Tensor> pv(const at::Tensor& P, const at::Tensor& v, const at::Tensor& blk_scale,
                           const at::Tensor& block_idx, const at::Tensor& block_cnt, int64_t num_splits,
                           bool want_f32) {
  check_kv(v, "v");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == at::kHalf && P.dim() == 3 && P.is_contiguous(), "P: fp16 [B, Hq, t_pad]");
  TORCH_CHECK(blk_scale.is_cuda() && blk_scale.scalar_type() == at::kFloat && blk_scale.dim() == 4 &&
                  blk_scale.is_contiguous(), "blk_scale: contiguous fp32 [B, Hkv, nb, G]");
  const c10::cuda::CUDAGuard guard(v.device());
  const int B = v.size(0), Hkv = v.size(1), T = v.size(2), D = v.size(3);
  const int Hq = P.size(1), G = Hq / Hkv;
  const int nb = (T + kvskip::TILE - 1) / kvskip::TILE;
  TORCH_CHECK(P.size(0) == B && P.size(2) == static_cast<int64_t>(nb) * kvskip::TILE, "P shape");
  TORCH_CHECK(blk_scale.size(0) == B && blk_scale.size(1) == Hkv && blk_scale.size(2) == nb && blk_scale.size(3) == G,
              "blk_scale shape");
  TORCH_CHECK(block_idx.scalar_type() == at::kInt && block_idx.is_contiguous() && block_idx.size(0) == B &&
                  block_idx.size(1) == Hkv, "block_idx");
  TORCH_CHECK(block_cnt.scalar_type() == at::kInt && block_cnt.is_contiguous() && block_cnt.numel() == B * Hkv,
              "block_cnt");
  auto f32 = v.options().dtype(at::kFloat);
  at::Tensor out = at::empty({B, Hq, D}, v.options());
  at::Tensor out32 = want_f32 ? at::empty({B, Hq, D}, f32) : at::Tensor();
  at::Tensor part_acc = num_splits > 1 ? at::empty({B, Hq, num_splits, D}, f32) : at::Tensor();
  kvskip::Params p{};
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.v_sb = v.stride(0); p.v_sh = v.stride(1); p.v_st = v.stride(2);
  p.B = B; p.Hq = Hq; p.Hkv = Hkv; p.T = T;
  p.block_idx = block_idx.data_ptr<int>();
  p.block_cnt = block_cnt.data_ptr<int>();
  p.max_blocks = block_idx.size(2);
  p.num_splits = num_splits;
  p.part_acc = num_splits > 1 ? part_acc.data_ptr<float>() : nullptr;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_f32 = want_f32 ? out32.data_ptr<float>() : nullptr;
  p.nb = nb;
  p.t_pad = static_cast<int64_t>(nb) * kvskip::TILE;
  p.p_in = reinterpret_cast<const __half*>(P.data_ptr());
  p.blk_scale = blk_scale.data_ptr<float>();
  kvskip::launch_pv(p, D, G, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, out32};
}

// Single-pass V-skip. Returns [out bf16, lse, out_f32 (or empty), vcount int32 [B, Hkv] (or empty)].
std::vector<at::Tensor> vskip1(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale,
                               double tau, int64_t local, int64_t num_splits, bool want_f32, bool count,
                               const c10::optional<at::Tensor>& lse_prev) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3, "q must be CUDA bf16 [B, Hq, D]");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0, "q rows must be 16-byte aligned");
  check_kv(k, "k");
  check_kv(v, "v");
  TORCH_CHECK(tau > 0.0 && local >= 0 && num_splits >= 1, "tau > 0, local >= 0, num_splits >= 1");
  const c10::cuda::CUDAGuard guard(q.device());
  const int B = q.size(0), Hq = q.size(1), D = q.size(2), Hkv = k.size(1), T = k.size(2);
  auto f32 = q.options().dtype(at::kFloat);
  at::Tensor out = at::empty({B, Hq, D}, q.options());
  at::Tensor out32 = want_f32 ? at::empty({B, Hq, D}, f32) : at::Tensor();
  at::Tensor lse = at::empty({B, Hq}, f32);
  at::Tensor vcount = count ? at::zeros({B, Hkv}, q.options().dtype(at::kInt)) : at::Tensor();
  at::Tensor part_acc, part_ml;
  if (num_splits > 1) {
    part_acc = at::empty({B, Hq, num_splits, D}, f32);
    part_ml = at::empty({B, Hq, num_splits, 2}, f32);
  }
  kvskip::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.q_sb = q.stride(0); p.q_sh = q.stride(1);
  p.k_sb = k.stride(0); p.k_sh = k.stride(1); p.k_st = k.stride(2);
  p.v_sb = v.stride(0); p.v_sh = v.stride(1); p.v_st = v.stride(2);
  p.B = B; p.Hq = Hq; p.Hkv = Hkv; p.T = T;
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.num_splits = num_splits;
  p.part_acc = num_splits > 1 ? part_acc.data_ptr<float>() : nullptr;
  p.part_ml = num_splits > 1 ? part_ml.data_ptr<float>() : nullptr;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_f32 = want_f32 ? out32.data_ptr<float>() : nullptr;
  p.lse = lse.data_ptr<float>();
  p.log2_tau = static_cast<float>(std::log2(tau));
  p.local = local;
  p.vcount = count ? vcount.data_ptr<int>() : nullptr;
  if (lse_prev.has_value()) {
    TORCH_CHECK(lse_prev->is_cuda() && lse_prev->scalar_type() == at::kFloat && lse_prev->is_contiguous() &&
                    lse_prev->numel() == static_cast<int64_t>(B) * Hq, "lse_prev: contiguous fp32 [B, Hq]");
    p.lse_prev = lse_prev->data_ptr<float>();
  }
  kvskip::launch_vskip1(p, D, Hq / Hkv, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, out32, vcount};
}

void read_bw(const at::Tensor& x, at::Tensor& out, int64_t blocks) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.nbytes() % 16 == 0, "x: contiguous CUDA, 16-byte multiple");
  kvskip::launch_read(x.data_ptr(), x.nbytes(), reinterpret_cast<uint32_t*>(out.data_ptr()), blocks,
                      at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read_bw", &read_bw, "streaming read of x (bandwidth ceiling)");
  m.def("vskip1", &vskip1, "kvskip single-pass V-skip (sink-anchored bound)", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("scale"), py::arg("tau"), py::arg("local") = 2, py::arg("num_splits") = 1,
        py::arg("want_f32") = false, py::arg("count") = false, py::arg("lse_prev") = py::none());
  m.def("decode", &decode, "kvskip split-K decode attention",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("block_idx") = py::none(),
        py::arg("block_cnt") = py::none(), py::arg("stats") = false, py::arg("load_v") = true,
        py::arg("num_splits") = 1, py::arg("stages") = 2, py::arg("want_f32") = false,
        py::arg("write_p") = false);
  m.def("pv", &pv, "kvskip two-pass V-skip, pass 2", py::arg("P"), py::arg("v"), py::arg("blk_scale"),
        py::arg("block_idx"), py::arg("block_cnt"), py::arg("num_splits") = 1, py::arg("want_f32") = false);
}
