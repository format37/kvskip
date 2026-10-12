// kvskip decode kernels: split-K flash-decoding, GQA-packed, CUDA-core FMA, cp.async pipeline.
//
// Grid (num_splits, Hkv, B), 128 threads. One CTA streams a contiguous range of
// 64-token tiles of one KV head (or a range of a kept-block list in masked mode)
// and serves all G query heads of that KV head, so K and V are read once.
// Per tile:
//   QK      8 lanes per token, 4 tokens per warp-iteration; lane j holds dims
//           [8j, 8j+8) (+64 for D=128) of the G pre-scaled queries in registers;
//           the 8 partial dots are reduced with 3 xor-shuffles.
//   softmax warp g owns query head g: tile max, online rescale, P to smem;
//           in stats mode also the per-block max and sum exp(s - block max).
//   PV      warp w owns 16 tokens, lane owns D/32 dims for all G heads.
// The 4 warps' partial accumulators are summed in the epilogue. With
// num_splits > 1 each CTA writes (acc, m, l) and combine_kernel reduces.
#include "kvskip.h"

#include <cmath>
#include <stdexcept>

namespace kvskip {

namespace {

constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int n = pred ? 16 : 0;  // src-size 0 -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float lo_bf16(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }

__device__ __forceinline__ void bf16x8(const uint4& r, float* f) {
  f[0] = lo_bf16(r.x); f[1] = hi_bf16(r.x);
  f[2] = lo_bf16(r.y); f[3] = hi_bf16(r.y);
  f[4] = lo_bf16(r.z); f[5] = hi_bf16(r.z);
  f[6] = lo_bf16(r.w); f[7] = hi_bf16(r.w);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int D, bool LOAD_V, int STAGES>
constexpr int smem_bytes() {
  return STAGES * TILE * D * 2 * (LOAD_V ? 2 : 1) + (2 * 4 * TILE + 3 * 4) * 4;
}

template <int D, int G, int STAGES, bool MASKED, bool STATS, bool LOAD_V>
__global__ void __launch_bounds__(NTHREADS, 1) decode_kernel(const Params p) {
  constexpr int NW = NTHREADS / 32;
  constexpr int CH = D / 64;   // 16-byte K chunks per lane in QK
  constexpr int DPL = D / 32;  // V dims per lane in PV
  constexpr int CPR = D / 8;   // 16-byte chunks per K/V row
  static_assert(G == NW, "softmax maps one warp to one query head");
  static_assert(D == 64 || D == 128, "head dim");
  static_assert(STAGES >= 2, "pipeline");

  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sV = sK + STAGES * TILE * D;
  float* sS = reinterpret_cast<float*>(smem + STAGES * TILE * D * 2 * (LOAD_V ? 2 : 1));  // [G][TILE]
  float* sP = sS + G * TILE;                                                              // [TILE][G]
  float* sA = sP + TILE * G;                                                              // [G]
  float* sML = sA + G;                                                                    // [G][2]

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = lane >> 3, j = lane & 7;

  int n_items;
  const int* idx = nullptr;
  if constexpr (MASKED) {
    n_items = p.block_cnt[b * p.Hkv + h];
    idx = p.block_idx + static_cast<int64_t>(b * p.Hkv + h) * p.max_blocks;
  } else {
    n_items = (p.T + TILE - 1) / TILE;
  }
  const int per = (n_items + p.num_splits - 1) / p.num_splits;
  const int i0 = min(n_items, split * per);
  const int n = min(n_items, i0 + per) - i0;

  const __nv_bfloat16* Kb = p.k + b * p.k_sb + h * p.k_sh;
  const __nv_bfloat16* Vb = p.v + b * p.v_sb + h * p.v_sh;

  // Queries of the G heads, pre-scaled to log2 units, in the QK lane layout.
  float qr[G][CH * 8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const __nv_bfloat16* qp = p.q + b * p.q_sb + (h * G + g) * p.q_sh;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      float f[8];
      bf16x8(*reinterpret_cast<const uint4*>(qp + c * 64 + 8 * j), f);
#pragma unroll
      for (int e = 0; e < 8; ++e) qr[g][c * 8 + e] = f[e] * p.scale_log2;
    }
  }

  auto block_of = [&](int i) { return MASKED ? idx[i0 + i] : i0 + i; };
  auto load_tile = [&](int stage, int i) {
    const int t0 = block_of(i) * TILE;
#pragma unroll
    for (int u = 0; u < TILE * CPR / NTHREADS; ++u) {
      const int c = tid + u * NTHREADS;
      const int r = c / CPR, cc = c % CPR, t = t0 + r;
      const bool ok = t < p.T;
      const int64_t tt = ok ? t : 0;
      cp_async16(sK + (stage * TILE + r) * D + cc * 8, Kb + tt * p.k_st + cc * 8, ok);
      if constexpr (LOAD_V) cp_async16(sV + (stage * TILE + r) * D + cc * 8, Vb + tt * p.v_st + cc * 8, ok);
    }
  };

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < n) load_tile(s, s);
    cp_async_commit();
  }

  float m_run = -INFINITY, l_run = 0.f;  // of query head `warp` (softmax phase)
  float acc[G][DPL];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < DPL; ++e) acc[g][e] = 0.f;

  for (int i = 0; i < n; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();  // tile i visible to all; everybody is done with tile i-1
    if (i + STAGES - 1 < n) load_tile((i + STAGES - 1) % STAGES, i + STAGES - 1);
    cp_async_commit();

    const int stage = i % STAGES;
    const int blk = block_of(i);
    const int t0 = blk * TILE;
    const __nv_bfloat16* tK = sK + stage * TILE * D;

    // ---- QK
#pragma unroll
    for (int it = 0; it < TILE / (NW * 4); ++it) {
      const int r = warp * (TILE / NW) + it * 4 + grp;
      float sacc[G];
#pragma unroll
      for (int g = 0; g < G; ++g) sacc[g] = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        float f[8];
        bf16x8(*reinterpret_cast<const uint4*>(tK + r * D + c * 64 + 8 * j), f);
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int e = 0; e < 8; ++e) sacc[g] = fmaf(qr[g][c * 8 + e], f[e], sacc[g]);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1)
#pragma unroll
        for (int g = 0; g < G; ++g) sacc[g] += __shfl_xor_sync(0xffffffffu, sacc[g], off);
      if (j < G) {
        float val = sacc[0];
#pragma unroll
        for (int g = 1; g < G; ++g)
          if (j == g) val = sacc[g];
        sS[j * TILE + r] = (t0 + r < p.T) ? val : -INFINITY;
      }
    }
    __syncthreads();

    // ---- online softmax, warp g <-> query head g
    {
      const int g = warp;
      const float s0 = sS[g * TILE + lane], s1 = sS[g * TILE + lane + 32];
      const float mt = warp_max(fmaxf(s0, s1));
      const float m_new = fmaxf(m_run, mt);
      const float mu = (m_new == -INFINITY) ? 0.f : m_new;
      const float p0 = ex2(s0 - mu), p1 = ex2(s1 - mu);
      const float lt = warp_sum(p0 + p1);
      const float alpha = ex2(m_run - mu);
      l_run = l_run * alpha + lt;
      m_run = m_new;
      sP[lane * G + g] = p0;
      sP[(lane + 32) * G + g] = p1;
      if (lane == 0) sA[g] = alpha;
      if constexpr (STATS) {
        const float mtu = (mt == -INFINITY) ? 0.f : mt;
        const float e0 = ex2(s0 - mtu), e1 = ex2(s1 - mtu);
        const float st = warp_sum(e0 + e1);
        const int64_t hq = static_cast<int64_t>(b) * p.Hq + h * G + g;
        if (lane == 0) {
          p.blk_max[hq * p.nb + blk] = mt * LN2;
          p.blk_sum[hq * p.nb + blk] = st;
        }
        if (p.p_out) {
          __half* po = p.p_out + hq * p.t_pad + t0;
          po[lane] = __float2half_rn(e0);
          po[lane + 32] = __float2half_rn(e1);
        }
      }
    }
    __syncthreads();

    // ---- PV
    if constexpr (LOAD_V) {
      const __nv_bfloat16* tV = sV + stage * TILE * D;
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float a = sA[g];
#pragma unroll
        for (int e = 0; e < DPL; ++e) acc[g][e] *= a;
      }
#pragma unroll
      for (int ii = 0; ii < TILE / NW; ++ii) {
        const int r = warp * (TILE / NW) + ii;
        float vf[DPL];
        if constexpr (DPL == 4) {
          const uint2 raw = *reinterpret_cast<const uint2*>(tV + r * D + 4 * lane);
          vf[0] = lo_bf16(raw.x); vf[1] = hi_bf16(raw.x);
          vf[2] = lo_bf16(raw.y); vf[3] = hi_bf16(raw.y);
        } else {
          const uint32_t raw = *reinterpret_cast<const uint32_t*>(tV + r * D + 2 * lane);
          vf[0] = lo_bf16(raw); vf[1] = hi_bf16(raw);
        }
        const float4 pr = *reinterpret_cast<const float4*>(sP + r * G);
        const float pg[4] = {pr.x, pr.y, pr.z, pr.w};
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int e = 0; e < DPL; ++e) acc[g][e] = fmaf(pg[g], vf[e], acc[g][e]);
      }
    }
  }
  cp_async_wait<0>();

  // ---- epilogue
  if (lane == 0) {
    sML[2 * warp] = m_run;
    sML[2 * warp + 1] = l_run;
  }
  __syncthreads();
  if constexpr (LOAD_V) {
    float* red = reinterpret_cast<float*>(smem);  // [NW][G][D], aliases the K stages
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int e = 0; e < DPL; ++e) red[(warp * G + g) * D + DPL * lane + e] = acc[g][e];
    __syncthreads();
    for (int o = tid; o < G * D; o += NTHREADS) {
      const int g = o / D, d = o % D;
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < NW; ++w) sum += red[(w * G + g) * D + d];
      const int64_t hq = static_cast<int64_t>(b) * p.Hq + h * G + g;
      if (p.num_splits == 1) {
        const float l = sML[2 * g + 1];
        const float out = l > 0.f ? sum / l : 0.f;
        p.out[hq * D + d] = __float2bfloat16(out);
        if (p.out_f32) p.out_f32[hq * D + d] = out;
      } else {
        p.part_acc[(hq * p.num_splits + split) * D + d] = sum;
      }
    }
  }
  if (tid < G) {
    const int64_t hq = static_cast<int64_t>(b) * p.Hq + h * G + tid;
    const float m = sML[2 * tid], l = sML[2 * tid + 1];
    if (p.num_splits == 1) {
      if (p.lse) p.lse[hq] = l > 0.f ? (m + log2f(l)) * LN2 : -INFINITY;
    } else {
      p.part_ml[(hq * p.num_splits + split) * 2] = m;
      p.part_ml[(hq * p.num_splits + split) * 2 + 1] = l;
    }
  }
}

// One CTA per (b, query head), one thread per dim.
template <int D>
__global__ void __launch_bounds__(D) combine_kernel(const Params p) {
  const int64_t hq = blockIdx.x;
  const int d = threadIdx.x;
  const int S = p.num_splits;
  const float* ml = p.part_ml + hq * S * 2;
  float m = -INFINITY;
  for (int s = 0; s < S; ++s) m = fmaxf(m, ml[2 * s]);
  const float mu = (m == -INFINITY) ? 0.f : m;
  float l = 0.f, acc = 0.f;
  for (int s = 0; s < S; ++s) {
    const float w = ex2(ml[2 * s] - mu);
    l += ml[2 * s + 1] * w;
    acc += p.part_acc[(hq * S + s) * D + d] * w;
  }
  const float out = l > 0.f ? acc / l : 0.f;
  p.out[hq * D + d] = __float2bfloat16(out);
  if (p.out_f32) p.out_f32[hq * D + d] = out;
  if (d == 0 && p.lse) p.lse[hq] = l > 0.f ? (m + log2f(l)) * LN2 : -INFINITY;
}

// Two-pass V-skip, pass 2: out = sum over kept blocks of blk_scale * P . V.
// P (fp16, relative to the block max) and blk_scale come from pass 1 + the planner,
// so the weights are already normalised: no online softmax, splits just add up.
template <int D, int G, int STAGES>
__global__ void __launch_bounds__(NTHREADS, 1) pv_kernel(const Params p) {
  constexpr int NW = NTHREADS / 32;
  constexpr int DPL = D / 32;
  constexpr int CPR = D / 8;
  static_assert(G == 4, "");
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(smem);                       // [STAGES][TILE][D]
  __half* sP = reinterpret_cast<__half*>(smem + STAGES * TILE * D * 2);              // [STAGES][G][TILE]
  float* sF = reinterpret_cast<float*>(smem + STAGES * TILE * D * 2 + STAGES * G * TILE * 2);  // [STAGES][G]

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_items = p.block_cnt[b * p.Hkv + h];
  const int* idx = p.block_idx + static_cast<int64_t>(b * p.Hkv + h) * p.max_blocks;
  const int per = (n_items + p.num_splits - 1) / p.num_splits;
  const int i0 = min(n_items, split * per);
  const int n = min(n_items, i0 + per) - i0;
  const __nv_bfloat16* Vb = p.v + b * p.v_sb + h * p.v_sh;
  const int64_t hq0 = static_cast<int64_t>(b) * p.Hq + h * G;

  auto load_tile = [&](int stage, int i) {
    const int blk = idx[i0 + i];
    const int t0 = blk * TILE;
#pragma unroll
    for (int u = 0; u < TILE * CPR / NTHREADS; ++u) {
      const int c = tid + u * NTHREADS;
      const int r = c / CPR, cc = c % CPR, t = t0 + r;
      const bool ok = t < p.T;
      cp_async16(sV + (stage * TILE + r) * D + cc * 8, Vb + static_cast<int64_t>(ok ? t : 0) * p.v_st + cc * 8, ok);
    }
    if (tid < G * TILE / 8) {  // 32 chunks of 8 halves
      const int g = tid / (TILE / 8), part = tid % (TILE / 8);
      cp_async16(sP + (stage * G + g) * TILE + part * 8, p.p_in + (hq0 + g) * p.t_pad + t0 + part * 8, true);
    }
    if (tid == 0)
      cp_async16(sF + stage * G, p.blk_scale + ((static_cast<int64_t>(b) * p.Hkv + h) * p.nb + blk) * G, true);
  };

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < n) load_tile(s, s);
    cp_async_commit();
  }
  float acc[G][DPL];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < DPL; ++e) acc[g][e] = 0.f;

  for (int i = 0; i < n; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (i + STAGES - 1 < n) load_tile((i + STAGES - 1) % STAGES, i + STAGES - 1);
    cp_async_commit();
    const int stage = i % STAGES;
    const __nv_bfloat16* tV = sV + stage * TILE * D;
    const __half* tP = sP + stage * G * TILE;
    float f[G];
#pragma unroll
    for (int g = 0; g < G; ++g) f[g] = sF[stage * G + g];
#pragma unroll
    for (int ii = 0; ii < TILE / NW; ++ii) {
      const int r = warp * (TILE / NW) + ii;
      float vf[DPL];
      if constexpr (DPL == 4) {
        const uint2 raw = *reinterpret_cast<const uint2*>(tV + r * D + 4 * lane);
        vf[0] = lo_bf16(raw.x); vf[1] = hi_bf16(raw.x);
        vf[2] = lo_bf16(raw.y); vf[3] = hi_bf16(raw.y);
      } else {
        const uint32_t raw = *reinterpret_cast<const uint32_t*>(tV + r * D + 2 * lane);
        vf[0] = lo_bf16(raw); vf[1] = hi_bf16(raw);
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float pg = __half2float(tP[g * TILE + r]) * f[g];
#pragma unroll
        for (int e = 0; e < DPL; ++e) acc[g][e] = fmaf(pg, vf[e], acc[g][e]);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);  // [NW][G][D]
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < DPL; ++e) red[(warp * G + g) * D + DPL * lane + e] = acc[g][e];
  __syncthreads();
  for (int o = tid; o < G * D; o += NTHREADS) {
    const int g = o / D, d = o % D;
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) sum += red[(w * G + g) * D + d];
    const int64_t hq = hq0 + g;
    if (p.num_splits == 1) {
      p.out[hq * D + d] = __float2bfloat16(sum);
      if (p.out_f32) p.out_f32[hq * D + d] = sum;
    } else {
      p.part_acc[(hq * p.num_splits + split) * D + d] = sum;
    }
  }
}

template <int D>
__global__ void __launch_bounds__(D) combine_sum_kernel(const Params p) {
  const int64_t hq = blockIdx.x;
  const int d = threadIdx.x;
  float acc = 0.f;
  for (int s = 0; s < p.num_splits; ++s) acc += p.part_acc[(hq * p.num_splits + s) * D + d];
  p.out[hq * D + d] = __float2bfloat16(acc);
  if (p.out_f32) p.out_f32[hq * D + d] = acc;
}

template <int D, int G, int STAGES>
void launch_pv_t(const Params& p, cudaStream_t stream) {
  auto kernel = pv_kernel<D, G, STAGES>;
  constexpr int smem = STAGES * TILE * D * 2 + STAGES * G * TILE * 2 + STAGES * G * 4;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  dim3 grid(p.num_splits, p.Hkv, p.B);
  kernel<<<grid, NTHREADS, smem, stream>>>(p);
  if (p.num_splits > 1) combine_sum_kernel<D><<<p.B * p.Hq, D, 0, stream>>>(p);
}

template <int D, int G, int STAGES, bool MASKED, bool STATS, bool LOAD_V>
void launch(const Params& p, cudaStream_t stream) {
  auto kernel = decode_kernel<D, G, STAGES, MASKED, STATS, LOAD_V>;
  constexpr int smem = smem_bytes<D, LOAD_V, STAGES>();
  static bool configured = false;  // one attribute call per instantiation
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  dim3 grid(p.num_splits, p.Hkv, p.B);
  kernel<<<grid, NTHREADS, smem, stream>>>(p);
  if (LOAD_V && p.num_splits > 1) combine_kernel<D><<<p.B * p.Hq, D, 0, stream>>>(p);
}

template <int D, int G, int STAGES>
void dispatch_flags(const Params& p, bool masked, bool stats, bool load_v, cudaStream_t s) {
#define KVSKIP_CASE(M, ST, LV) \
  if (masked == M && stats == ST && load_v == LV) return launch<D, G, STAGES, M, ST, LV>(p, s);
  KVSKIP_CASE(false, false, true)
  KVSKIP_CASE(false, true, true)
  KVSKIP_CASE(false, true, false)
  KVSKIP_CASE(true, false, true)
  KVSKIP_CASE(true, true, true)
  KVSKIP_CASE(true, true, false)
#undef KVSKIP_CASE
  throw std::runtime_error("kvskip: unsupported flag combination (load_v=false needs stats=true)");
}


// Single-pass V-skip with deferred V loads. Per tile: QK -> block stats -> decide
// (union over the G heads) whether V is needed -> issue V's cp.async; the PV of
// tile i runs one iteration later, overlapping V(i)'s load with QK(i+1).
// cp.async group order: prologue K(0), [V(-1)=empty]; iteration i commits K(i+1)
// then V(i) (possibly empty). Top of iteration i: wait_group<1> -> K(i) landed;
// before PV(i-1): wait_group<2> -> V(i-1) landed.
template <int D, int G>
__global__ void __launch_bounds__(NTHREADS, 1) vskip1_kernel(const Params p) {
  constexpr int NW = NTHREADS / 32;
  constexpr int CH = D / 64;
  constexpr int DPL = D / 32;
  constexpr int CPR = D / 8;
  static_assert(G == NW, "softmax maps one warp to one query head");
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);   // [2][TILE][D]
  __nv_bfloat16* sV = sK + 2 * TILE * D;                         // [2][TILE][D]
  float* sS = reinterpret_cast<float*>(sV + 2 * TILE * D);       // [G][TILE]
  float* sP = sS + G * TILE;                                     // [2][TILE][G]
  float* sA = sP + 2 * TILE * G;                                 // [2][G]
  float* sML = sA + 2 * G;                                       // [G][2]
  int* sNeedW = reinterpret_cast<int*>(sML + 2 * G);             // [G] per-head vote
  int* sNeed = sNeedW + G;                                       // [2] tile needs V

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = lane >> 3, j = lane & 7;
  const int nb = (p.T + TILE - 1) / TILE;
  const int per = (nb + p.num_splits - 1) / p.num_splits;
  const int i0 = min(nb, split * per);
  const int n = min(nb, i0 + per) - i0;
  const __nv_bfloat16* Kb = p.k + b * p.k_sb + h * p.k_sh;
  const __nv_bfloat16* Vb = p.v + b * p.v_sb + h * p.v_sh;

  float qr[G][CH * 8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const __nv_bfloat16* qp = p.q + b * p.q_sb + (h * G + g) * p.q_sh;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      float f[8];
      bf16x8(*reinterpret_cast<const uint4*>(qp + c * 64 + 8 * j), f);
#pragma unroll
      for (int e = 0; e < 8; ++e) qr[g][c * 8 + e] = f[e] * p.scale_log2;
    }
  }

  auto load_rows = [&](__nv_bfloat16* dst, const __nv_bfloat16* base, int64_t st, int blk) {
    const int t0 = blk * TILE;
#pragma unroll
    for (int u = 0; u < TILE * CPR / NTHREADS; ++u) {
      const int c = tid + u * NTHREADS;
      const int r = c / CPR, cc = c % CPR, t = t0 + r;
      const bool ok = t < p.T;
      cp_async16(dst + r * D + cc * 8, base + static_cast<int64_t>(ok ? t : 0) * st + cc * 8, ok);
    }
  };
  auto qk = [&](const __nv_bfloat16* tK, int t0) {
#pragma unroll
    for (int it = 0; it < TILE / (NW * 4); ++it) {
      const int r = warp * (TILE / NW) + it * 4 + grp;
      float sacc[G];
#pragma unroll
      for (int g = 0; g < G; ++g) sacc[g] = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        float f[8];
        bf16x8(*reinterpret_cast<const uint4*>(tK + r * D + c * 64 + 8 * j), f);
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int e = 0; e < 8; ++e) sacc[g] = fmaf(qr[g][c * 8 + e], f[e], sacc[g]);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1)
#pragma unroll
        for (int g = 0; g < G; ++g) sacc[g] += __shfl_xor_sync(0xffffffffu, sacc[g], off);
      if (j < G) {
        float val = sacc[0];
#pragma unroll
        for (int g = 1; g < G; ++g)
          if (j == g) val = sacc[g];
        sS[j * TILE + r] = (t0 + r < p.T) ? val : -INFINITY;
      }
    }
  };

  // ---- normaliser estimate for this warp's head (log2 units): the previous step's
  // log-sum-exp if given, else the sink block's (block 0) = a lower bound
  float lse0;
  if (p.lse_prev) {
    lse0 = p.lse_prev[static_cast<int64_t>(b) * p.Hq + h * G + warp] * 1.4426950408889634f;
  } else {
    load_rows(sK, Kb, p.k_st, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    qk(sK, 0);
    __syncthreads();
    const float s0 = sS[warp * TILE + lane], s1 = sS[warp * TILE + lane + 32];
    const float m0 = warp_max(fmaxf(s0, s1));
    lse0 = m0 + log2f(warp_sum(ex2(s0 - m0) + ex2(s1 - m0)));
    __syncthreads();  // sK stage 0 and sS are reused below
  }

  if (n > 0) load_rows(sK, Kb, p.k_st, i0);
  cp_async_commit();   // K(0)
  cp_async_commit();   // V(-1): empty
  float m_run = -INFINITY, l_run = 0.f;
  float acc[G][DPL];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < DPL; ++e) acc[g][e] = 0.f;
  int loaded = 0;

  auto pv = [&](int buf) {
    const __nv_bfloat16* tV = sV + buf * TILE * D;
    const float* tP = sP + buf * TILE * G;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float a = sA[buf * G + g];
#pragma unroll
      for (int e = 0; e < DPL; ++e) acc[g][e] *= a;
    }
#pragma unroll
    for (int ii = 0; ii < TILE / NW; ++ii) {
      const int r = warp * (TILE / NW) + ii;
      float vf[DPL];
      if constexpr (DPL == 4) {
        const uint2 raw = *reinterpret_cast<const uint2*>(tV + r * D + 4 * lane);
        vf[0] = lo_bf16(raw.x); vf[1] = hi_bf16(raw.x);
        vf[2] = lo_bf16(raw.y); vf[3] = hi_bf16(raw.y);
      } else {
        const uint32_t raw = *reinterpret_cast<const uint32_t*>(tV + r * D + 2 * lane);
        vf[0] = lo_bf16(raw); vf[1] = hi_bf16(raw);
      }
      const float4 pr = *reinterpret_cast<const float4*>(tP + r * G);
      const float pg[4] = {pr.x, pr.y, pr.z, pr.w};
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int e = 0; e < DPL; ++e) acc[g][e] = fmaf(pg[g], vf[e], acc[g][e]);
    }
  };

  for (int i = 0; i < n; ++i) {
    cp_async_wait<1>();
    __syncthreads();
    if (i + 1 < n) load_rows(sK + ((i + 1) & 1) * TILE * D, Kb, p.k_st, i0 + i + 1);
    cp_async_commit();   // K(i+1)
    const int blk = i0 + i;
    const int buf = i & 1;
    qk(sK + buf * TILE * D, blk * TILE);
    __syncthreads();
    // block stats and the vote of this warp's head
    const int g = warp;
    const float s0 = sS[g * TILE + lane], s1 = sS[g * TILE + lane + 32];
    const float mt = warp_max(fmaxf(s0, s1));
    const float mtu = (mt == -INFINITY) ? 0.f : mt;
    const float lse_b = mtu + log2f(warp_sum(ex2(s0 - mtu) + ex2(s1 - mtu)));
    if (lane == 0) sNeedW[g] = (lse_b - lse0 >= p.log2_tau) ? 1 : 0;
    __syncthreads();
    bool need = blk == 0 || blk >= nb - p.local;
#pragma unroll
    for (int w = 0; w < G; ++w) need = need || sNeedW[w];
    if (need) {
      const float m_new = fmaxf(m_run, mt);
      const float mu = (m_new == -INFINITY) ? 0.f : m_new;
      const float p0 = ex2(s0 - mu), p1 = ex2(s1 - mu);
      const float alpha = ex2(m_run - mu);
      l_run = l_run * alpha + warp_sum(p0 + p1);
      m_run = m_new;
      sP[(buf * TILE + lane) * G + g] = p0;
      sP[(buf * TILE + lane + 32) * G + g] = p1;
      if (lane == 0) sA[buf * G + g] = alpha;
      load_rows(sV + buf * TILE * D, Vb, p.v_st, blk);
      ++loaded;
    }
    if (tid == 0) sNeed[buf] = need ? 1 : 0;
    cp_async_commit();   // V(i), empty if not needed
    if (i > 0) {
      cp_async_wait<2>();
      __syncthreads();   // V(i-1), P(i-1), alpha(i-1), need(i-1) visible
      if (sNeed[buf ^ 1]) pv(buf ^ 1);
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  if (n > 0 && sNeed[(n - 1) & 1]) pv((n - 1) & 1);

  // ---- epilogue (as decode_kernel)
  if (lane == 0) {
    sML[2 * warp] = m_run;
    sML[2 * warp + 1] = l_run;
  }
  if (tid == 0 && p.vcount) atomicAdd(p.vcount + b * p.Hkv + h, loaded);
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < DPL; ++e) red[(warp * G + g) * D + DPL * lane + e] = acc[g][e];
  __syncthreads();
  for (int o = tid; o < G * D; o += NTHREADS) {
    const int g = o / D, d = o % D;
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) sum += red[(w * G + g) * D + d];
    const int64_t hq = static_cast<int64_t>(b) * p.Hq + h * G + g;
    if (p.num_splits == 1) {
      const float l = sML[2 * g + 1];
      const float out = l > 0.f ? sum / l : 0.f;
      p.out[hq * D + d] = __float2bfloat16(out);
      if (p.out_f32) p.out_f32[hq * D + d] = out;
    } else {
      p.part_acc[(hq * p.num_splits + split) * D + d] = sum;
    }
  }
  if (tid < G) {
    const int64_t hq = static_cast<int64_t>(b) * p.Hq + h * G + tid;
    const float m = sML[2 * tid], l = sML[2 * tid + 1];
    if (p.num_splits == 1) {
      if (p.lse) p.lse[hq] = l > 0.f ? (m + log2f(l)) * LN2 : -INFINITY;
    } else {
      p.part_ml[(hq * p.num_splits + split) * 2] = m;
      p.part_ml[(hq * p.num_splits + split) * 2 + 1] = l;
    }
  }
}

template <int D, int G>
void launch_vskip1_t(const Params& p, cudaStream_t stream) {
  auto kernel = vskip1_kernel<D, G>;
  constexpr int smem = 4 * TILE * D * 2 + (G * TILE + 2 * TILE * G + 2 * G + 2 * G) * 4 + 8 * 4;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  dim3 grid(p.num_splits, p.Hkv, p.B);
  kernel<<<grid, NTHREADS, smem, stream>>>(p);
  if (p.num_splits > 1) combine_kernel<D><<<p.B * p.Hq, D, 0, stream>>>(p);
}

// Streaming read: every thread reads 16-byte words in a grid-stride loop; the
// practical DRAM read ceiling the decode kernels are compared with.
__global__ void __launch_bounds__(256) read_kernel(const uint4* __restrict__ x, int64_t n, uint32_t* out) {
  uint32_t acc = 0;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#pragma unroll 4
  for (; i < n; i += stride) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(x + i));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = acc;  // keep the loads alive
}

}  // namespace

void launch_read(const void* x, int64_t bytes, uint32_t* out, int blocks, cudaStream_t stream) {
  read_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const uint4*>(x), bytes / 16, out);
}

int max_smem_bytes(int D, int stages, bool load_v) {
  const int kv = stages * TILE * D * 2 * (load_v ? 2 : 1);
  return kv + (2 * 4 * TILE + 3 * 4) * 4;
}

void launch_vskip1(const Params& p, int D, int G, cudaStream_t stream) {
  if (G != 4) throw std::runtime_error("kvskip: only G = 4 query heads per KV head are instantiated");
  if (D == 128) return launch_vskip1_t<128, 4>(p, stream);
  if (D == 64) return launch_vskip1_t<64, 4>(p, stream);
  throw std::runtime_error("kvskip: unsupported head dim");
}

void launch_pv(const Params& p, int D, int G, cudaStream_t stream) {
  if (G != 4) throw std::runtime_error("kvskip: only G = 4 query heads per KV head are instantiated");
  if (D == 128) return launch_pv_t<128, 4, 3>(p, stream);
  if (D == 64) return launch_pv_t<64, 4, 3>(p, stream);
  throw std::runtime_error("kvskip: unsupported head dim");
}

void launch_decode(const Params& p, int D, int G, int stages, bool masked, bool stats, bool load_v,
                   cudaStream_t stream) {
  if (G != 4) throw std::runtime_error("kvskip: only G = 4 query heads per KV head are instantiated");
  if (D == 128 && stages == 2) return dispatch_flags<128, 4, 2>(p, masked, stats, load_v, stream);
  if (D == 128 && stages == 3) return dispatch_flags<128, 4, 3>(p, masked, stats, load_v, stream);
  if (D == 64 && stages == 2) return dispatch_flags<64, 4, 2>(p, masked, stats, load_v, stream);
  if (D == 64 && stages == 3) return dispatch_flags<64, 4, 3>(p, masked, stats, load_v, stream);
  throw std::runtime_error("kvskip: unsupported head dim / stages");
}

}  // namespace kvskip
