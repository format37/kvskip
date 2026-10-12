// kvskip: split-K flash-decoding for Ada (sm_89) with GQA packing and block-skip hooks.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kvskip {

constexpr int TILE = 64;      // tokens per tile == KV block of the masks and of the stats
constexpr int NTHREADS = 128; // 4 warps

struct Params {
  // q [B, Hq, D]; k, v [B, Hkv, T, D] with arbitrary batch/head/token strides, D contiguous
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  int64_t q_sb, q_sh;
  int64_t k_sb, k_sh, k_st;
  int64_t v_sb, v_sh, v_st;
  int B, Hq, Hkv, T;
  float scale_log2;            // softmax scale * log2(e)

  // masked mode: per (b, kv head) list of kept block indices, ascending
  const int* block_idx;        // [B, Hkv, max_blocks]
  const int* block_cnt;        // [B, Hkv]
  int max_blocks;

  // split-K
  int num_splits;
  float* part_acc;             // [B, Hq, num_splits, D] un-normalised, relative to part m
  float* part_ml;              // [B, Hq, num_splits, 2] running max (log2 units) and sum

  // outputs
  __nv_bfloat16* out;          // [B, Hq, D]
  float* out_f32;              // optional [B, Hq, D]
  float* lse;                  // [B, Hq] natural-log logsumexp of the attended scores

  // stats mode: per 64-token block and query head
  float* blk_max;              // [B, Hq, nb] max score (natural units, scale applied)
  float* blk_sum;              // [B, Hq, nb] sum exp(score - blk_max)
  int nb;

  // two-pass V-skip: pass 1 (stats mode) writes P = exp(s - blk_max) in fp16,
  // pass 2 (pv) reads P and V of the kept blocks, weighted by blk_scale
  __half* p_out;               // [B, Hq, t_pad] or nullptr
  const __half* p_in;          // [B, Hq, t_pad]
  const float* blk_scale;      // [B, Hq, nb] exp(blk_max - lse_kept), 0 for dropped blocks
  int64_t t_pad;               // nb * TILE

  // single-pass V-skip: V of a block is loaded only if, for some query head of the
  // group, exp(lse_block - lse_sink) >= tau (lse_sink = log-sum-exp of block 0 <= the
  // full normaliser, so no block with weight >= tau is dropped); block 0 and the
  // `local` newest blocks are always loaded
  float log2_tau;
  int local;
  int* vcount;                 // [B, Hkv] V tiles loaded (atomic, optional)
  const float* lse_prev;       // [B, Hq] natural-log normaliser of the previous step: if given,
                               // it replaces the sink bound (no prologue; not a guarantee)
};

// Device-side refresh planner (kvskip_planner.cu), one CTA per (b, kv head), G = 4, D = 128.
struct PlanParams {
  const __nv_bfloat16* q;      // [B, Hq, D]
  int64_t q_sb, q_sh;
  const __nv_bfloat16* k;      // cache view [B, Hkv, T, D] (the new token at T - 1)
  int64_t k_sb, k_sh, k_st;
  float* csum;                 // [B, Hkv, NBC, D] per-block key sums (fp32)
  const float* lse_ref;        // [B, Hq, NBC] block log-sum-exp at the head's last refresh (-inf: none)
  const float* cs_ref;         // [B, Hq, NBC] centroid scores q.c_b at that refresh
  const uint8_t* keep;         // [B, Hkv, NBC] stored set
  const float* lse_att;        // [B, Hq] previous step's log-sum-exp over the attended set
  int* age;                    // [B, Hkv] steps since the last refresh (updated)
  const int* flags;            // [Hkv] bit 0: refresh on every step (always-exact head),
                               //       bit 1: token-index head (reads idx + sink + local, no refresh),
                               //       bit 2: index-union head (idx joins the attended set, trigger stays)
  const uint8_t* idx;          // [B, NBC] this step's token-index blocks (or nullptr without index heads)
  float* cs_now;               // [B, Hq, NBC] out: this step's centroid scores
  uint8_t* trig;               // [B, Hkv] out: refresh decision
  int* block_idx;              // [B, Hkv, NBC] out: blocks to read, ascending
  int* block_cnt;              // [B, Hkv] out
  int B, Hq, Hkv, T, NBC, local, max_age, add_token;
  float scale, delta;
};

struct UpdateParams {
  const float* blk_max;        // [B, Hq, nb] stats of this step's pass (-inf / 0 on blocks not read)
  const float* blk_sum;
  const uint8_t* trig;         // [B, Hkv]
  const float* cs_now;         // [B, Hq, NBC]
  uint8_t* keep;               // [B, Hkv, NBC] updated for refreshing heads
  float* lse_ref;              // [B, Hq, NBC] updated for refreshing heads
  float* cs_ref;               // [B, Hq, NBC] updated for refreshing heads
  float* lse_att;              // [B, Hq] updated for every head
  int B, Hq, Hkv, T, NBC, local;
  float level;
};

void launch_decode(const Params& p, int D, int G, int stages, bool masked, bool stats, bool load_v,
                   cudaStream_t stream);
void launch_plan(const PlanParams& p, cudaStream_t stream);
void launch_update(const UpdateParams& p, cudaStream_t stream);
void launch_pv(const Params& p, int D, int G, cudaStream_t stream);
void launch_vskip1(const Params& p, int D, int G, cudaStream_t stream);
int max_smem_bytes(int D, int stages, bool load_v);
void launch_read(const void* x, int64_t bytes, uint32_t* out, int blocks, cudaStream_t stream);

}  // namespace kvskip
