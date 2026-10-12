// kvskip planner kernels: the refresh policy's per-step decisions on the device (stage 3).
//
// The policy (src/kvskip_attn.py "refresh_pre"): per KV head a stored block set, the GQA
// union of each query head's top-p blocks at the head's last refresh; a step attends the
// stored set + sink block + `local` newest blocks, unless the drift estimate of the mass
// outside that set exceeds delta (or the set is max_age steps old, or the head is flagged
// always-exact), in which case the head reads every block and refreshes from the block
// stats of that same pass. Both kernels run one CTA per (b, kv head), 128 threads = 4
// warps, warp g <-> query head g of the group (G = 4, D = 128):
//
//   plan_kernel    before the attention pass: adds the new token's key to its block's key
//                  sum, computes the centroid scores q.c_b of every block, the estimate
//                  logsumexp_{b outside the set} (lse_ref_b + q.c_b - cs_ref_b) against the
//                  previous step's attended log-sum-exp, the refresh decision, and the block
//                  list of the masked + stats attention kernel (every block when refreshing).
//                  A KV head flagged as a token-index head (flags bit 1) reads the blocks of
//                  `idx` (earlier copies of the current token and the tokens after them,
//                  found by token id) + sink + local blocks and never refreshes; with bit 2
//                  (index union) the index blocks join the attended set and the trigger stays.
//   update_kernel  after the pass: for refreshing heads the new stored set (each query
//                  head's top-p by bisection on the weight threshold, union over the group,
//                  plus sink and local blocks as select_blocks does) and the references
//                  (block log-sum-exps, centroid scores); for every head the log-sum-exp of
//                  this step's block weights over the next step's attended set.
//
// No host synchronisation: the attention kernel's split count is sized for every block.
#include "kvskip.h"

#include <cmath>

namespace kvskip {

namespace {

constexpr int PT = 128;
constexpr int PG = 4;
constexpr int PD = 128;

__device__ __forceinline__ float psum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float pmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ bool always_block(int blk, int nb, int local) { return blk == 0 || blk >= nb - local; }

__device__ __forceinline__ void lse_push(float& m, float& s, float e) {
  if (e == -INFINITY) return;
  if (e > m) {
    s = s * __expf(m - e) + 1.f;
    m = e;
  } else {
    s += __expf(e - m);
  }
}

__global__ void __launch_bounds__(PT) plan_kernel(const PlanParams p) {
  constexpr int U = 8;                     // blocks per warp in flight (centroid loads)
  const int h = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = (p.T + TILE - 1) / TILE;
  const int64_t bh = static_cast<int64_t>(b) * p.Hkv + h;
  float* csum = p.csum + bh * p.NBC * PD;
  const uint8_t* keep = p.keep + bh * p.NBC;
  extern __shared__ float scs[];           // [PG][nb] centroid scores of this step
  int* bi = p.block_idx + bh * p.NBC;
  __shared__ int sScan[PT];

  if (p.flags[h] & 2) {                    // token-index head: the index blocks + sink + local, no refresh
    const uint8_t* idx = p.idx + static_cast<int64_t>(b) * p.NBC;
    const int per = (nb + PT - 1) / PT;
    const int i0 = min(nb, tid * per), i1 = min(nb, i0 + per);
    int cnt = 0;
    for (int i = i0; i < i1; ++i) cnt += idx[i] || always_block(i, nb, p.local);
    sScan[tid] = cnt;
    __syncthreads();
    for (int o = 1; o < PT; o <<= 1) {
      const int v = tid >= o ? sScan[tid - o] : 0;
      __syncthreads();
      sScan[tid] += v;
      __syncthreads();
    }
    int off = sScan[tid] - cnt;
    for (int i = i0; i < i1; ++i)
      if (idx[i] || always_block(i, nb, p.local)) bi[off++] = i;
    if (tid == PT - 1) p.block_cnt[bh] = sScan[PT - 1];
    if (tid == 0) p.trig[bh] = 0;
    return;
  }

  const bool uni = p.flags[h] & 4;         // index-union head: the index blocks join the attended set
  const uint8_t* idx = uni ? p.idx + static_cast<int64_t>(b) * p.NBC : nullptr;
  auto attended = [&](int blk) { return keep[blk] || always_block(blk, nb, p.local) || (uni && idx[blk]); };

  if (p.add_token && tid < PD) {           // the token appended by this step
    const int t = p.T - 1;
    const __nv_bfloat16* kr = p.k + b * p.k_sb + h * p.k_sh + static_cast<int64_t>(t) * p.k_st;
    csum[static_cast<int64_t>(t / TILE) * PD + tid] += __bfloat162float(kr[tid]);
  }
  __syncthreads();

  // 1. centroid scores: warp per block, lane owns dims [4 lane, 4 lane + 4), U blocks per round
  float qv[PG][4];
#pragma unroll
  for (int g = 0; g < PG; ++g) {
    const __nv_bfloat16* qp = p.q + b * p.q_sb + (h * PG + g) * p.q_sh + 4 * lane;
#pragma unroll
    for (int e = 0; e < 4; ++e) qv[g][e] = __bfloat162float(qp[e]) * p.scale;
  }
  const int64_t row0 = (static_cast<int64_t>(b) * p.Hq + h * PG) * p.NBC;     // query head g: row0 + g*NBC
  for (int base = warp; base < nb; base += 4 * U) {
    float4 c[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int blk = base + 4 * u;
      c[u] = blk < nb ? *reinterpret_cast<const float4*>(csum + static_cast<int64_t>(blk) * PD + 4 * lane)
                      : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int blk = base + 4 * u;
      if (blk >= nb) break;
      const float inv_n = 1.f / static_cast<float>(blk == nb - 1 ? p.T - blk * TILE : TILE);
      float d[PG];
#pragma unroll
      for (int g = 0; g < PG; ++g)
        d[g] = psum(qv[g][0] * c[u].x + qv[g][1] * c[u].y + qv[g][2] * c[u].z + qv[g][3] * c[u].w) * inv_n;
      if (lane < PG) {
        const float v = lane == 0 ? d[0] : lane == 1 ? d[1] : lane == 2 ? d[2] : d[3];
        scs[lane * nb + blk] = v;
        p.cs_now[row0 + lane * p.NBC + blk] = v;
      }
    }
  }
  __syncthreads();

  // 2. drift estimate over the blocks outside the attended set: thread per block (coalesced)
  float m[PG], s[PG];
#pragma unroll
  for (int g = 0; g < PG; ++g) {
    m[g] = -INFINITY;
    s[g] = 0.f;
  }
  for (int blk = tid; blk < nb; blk += PT) {
    if (attended(blk)) continue;
#pragma unroll
    for (int g = 0; g < PG; ++g)
      lse_push(m[g], s[g], p.lse_ref[row0 + g * p.NBC + blk] + scs[g * nb + blk] - p.cs_ref[row0 + g * p.NBC + blk]);
  }
  __shared__ float sM[PT / 32][PG], sS[PT / 32][PG];
  __shared__ int sTrig;
#pragma unroll
  for (int g = 0; g < PG; ++g) {
    const float M = pmax(m[g]);
    const float S = psum(s[g] > 0.f ? s[g] * __expf(m[g] - M) : 0.f);
    if (lane == 0) {
      sM[warp][g] = M;
      sS[warp][g] = S;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float miss = 0.f;
    for (int g = 0; g < PG; ++g) {
      float M = -INFINITY, S = 0.f;
      for (int w = 0; w < PT / 32; ++w)
        if (sS[w][g] > 0.f) M = fmaxf(M, sM[w][g]);
      for (int w = 0; w < PT / 32; ++w)
        if (sS[w][g] > 0.f) S += sS[w][g] * __expf(sM[w][g] - M);
      if (S > 0.f) {
        const float est = M + __logf(S);
        miss = fmaxf(miss, 1.f / (1.f + __expf(p.lse_att[static_cast<int64_t>(b) * p.Hq + h * PG + g] - est)));
      }
    }
    const int a = p.age[bh] + 1;
    const bool tr = miss > p.delta || a >= p.max_age || (p.flags[h] & 1);
    p.age[bh] = tr ? 0 : a;
    p.trig[bh] = tr;
    sTrig = tr;
  }
  __syncthreads();

  // 3. block list for the masked + stats attention kernel
  if (sTrig) {
    for (int i = tid; i < nb; i += PT) bi[i] = i;
    if (tid == 0) p.block_cnt[bh] = nb;
    return;
  }
  const int per = (nb + PT - 1) / PT;
  const int i0 = min(nb, tid * per), i1 = min(nb, i0 + per);
  int cnt = 0;
  for (int i = i0; i < i1; ++i) cnt += attended(i);
  sScan[tid] = cnt;
  __syncthreads();
  for (int o = 1; o < PT; o <<= 1) {
    const int v = tid >= o ? sScan[tid - o] : 0;
    __syncthreads();
    sScan[tid] += v;
    __syncthreads();
  }
  int off = sScan[tid] - cnt;
  for (int i = i0; i < i1; ++i)
    if (attended(i)) bi[off++] = i;
  if (tid == PT - 1) p.block_cnt[bh] = sScan[PT - 1];
}

__global__ void __launch_bounds__(PT) update_kernel(const UpdateParams p) {
  const int h = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = (p.T + TILE - 1) / TILE;
  const int64_t bh = static_cast<int64_t>(b) * p.Hkv + h;
  const int64_t hrow = static_cast<int64_t>(b) * p.Hq + h * PG + warp;      // this warp's query head
  extern __shared__ float sx[];                                           // [PG][nb] block log-weights
  float* x = sx + warp * nb;
  uint8_t* skeep = reinterpret_cast<uint8_t*>(sx + PG * nb);              // [nb]
  uint8_t* keep = p.keep + bh * p.NBC;
  const bool tr = p.trig[bh];

  float mx = -INFINITY;
  for (int i = lane; i < nb; i += 32) {
    const float v = p.blk_max[hrow * nb + i] + __logf(p.blk_sum[hrow * nb + i]);   // -inf if not read
    x[i] = v;
    mx = fmaxf(mx, v);
  }
  for (int i = tid; i < nb; i += PT) skeep[i] = 0;
  mx = pmax(mx);
  __syncthreads();
  if (tr) {
    float se = 0.f;
    for (int i = lane; i < nb; i += 32) se += __expf(x[i] - mx);
    const float lse = mx + __logf(psum(se));
    float lo = -80.f, hi = 0.f;            // mass of {y >= lo} >= level > mass of {y >= hi}, y = x - lse
    for (int it = 0; it < 32; ++it) {
      const float mid = 0.5f * (lo + hi);
      float acc = 0.f;
      for (int i = lane; i < nb; i += 32) {
        const float y = x[i] - lse;
        if (y >= mid) acc += __expf(y);
      }
      if (psum(acc) >= p.level) lo = mid; else hi = mid;
    }
    for (int i = lane; i < nb; i += 32)
      if (x[i] - lse >= lo) skeep[i] = 1;
    for (int i = lane; i < p.NBC; i += 32) {
      p.lse_ref[hrow * p.NBC + i] = i < nb ? x[i] : -INFINITY;
      p.cs_ref[hrow * p.NBC + i] = i < nb ? p.cs_now[hrow * p.NBC + i] : 0.f;
    }
  }
  __syncthreads();
  if (tr)
    for (int i = tid; i < p.NBC; i += PT) keep[i] = i < nb && (skeep[i] || always_block(i, nb, p.local));
  __syncthreads();
  float m2 = -INFINITY, s2 = 0.f;
  for (int i = lane; i < nb; i += 32)
    if (keep[i] || always_block(i, nb, p.local)) lse_push(m2, s2, x[i]);
  const float M = pmax(m2);
  const float S = psum(s2 > 0.f ? s2 * __expf(m2 - M) : 0.f);
  if (lane == 0) p.lse_att[hrow] = M + __logf(S);
}

}  // namespace

void launch_plan(const PlanParams& p, cudaStream_t stream) {
  const int nb = (p.T + TILE - 1) / TILE;
  plan_kernel<<<dim3(p.Hkv, p.B), PT, PG * nb * sizeof(float), stream>>>(p);
}

void launch_update(const UpdateParams& p, cudaStream_t stream) {
  const int nb = (p.T + TILE - 1) / TILE;
  const size_t smem = PG * nb * sizeof(float) + nb;
  update_kernel<<<dim3(p.Hkv, p.B), PT, smem, stream>>>(p);
}

}  // namespace kvskip
