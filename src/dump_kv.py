"""B0: dump real decode-time q/k/v for kernel tests.

Prefill a document exactly like run_ablate (prompt = ctx - steps tokens), then
greedy-decode through the exact fp32 probe and capture, for a few layers:
  * the query of several decode steps            L{l}.q        [S, Hq, D]  bf16
  * K and V after the last captured step         L{l}.k / .v   [Hkv, T, D] bf16
  * the GQA-union top-p 0.99 block mask of every captured step, computed from
    the exact fp32 weights (input for the sparse-kernel tests)
                                                 L{l}.mask099  [S, Hkv, nb] uint8
Step s of the dump sees the first T_s = prompt_len + s positions of K/V
(metadata "T"). Queries are post-RoPE, exactly what the attention kernel gets.

Output: data/kv_dumps/<model>_<genre>_<ctx>.safetensors (gitignored, ~0.5 GB each).
"""

import argparse
import json
import math

import torch
from safetensors.torch import save_file
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

import common
import metrics
import probe_attn
from run_collect import greedy_token

OUT = common.ROOT / "data" / "kv_dumps"


class Capture:
    def __init__(self, layers, steps):
        self.layers, self.steps = set(layers), set(steps)
        self.last = max(steps)
        self.step = -1                     # current decode step, set by the driver
        self.q, self.masks, self.kv, self.T = {}, {}, {}, {}

    def __call__(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        l = module.layer_idx
        if query.shape[2] == 1 and l in self.layers and self.step in self.steps:
            Hkv, T, D = key.shape[1:]
            G = query.shape[1] // Hkv
            q32 = query[0, :, 0].float().view(Hkv, G, D)
            s = torch.einsum("kgd,ktd->kgt", q32, key[0].float()) * scaling
            W = metrics.block_weights(torch.softmax(s, -1))
            self.q.setdefault(l, []).append(query[0, :, 0].clone())
            self.masks.setdefault(l, []).append(metrics.needed_blocks(W, 0.99).any(1))
            self.T[self.step] = T
            if self.step == self.last:
                self.kv[l] = (key[0].clone(), value[0].clone())
            self.scaling = scaling
        return probe_attn.probe_attention_forward(module, query, key, value, attention_mask,
                                                  scaling=scaling, dropout=dropout, **kw)


def run(model_name, genre, ctx, steps, layers, capture_steps, tok, model):
    eos_ids = list(model.generation_config.eos_token_id)
    ids, _ = common.build_prompt(tok, common.load_doc(genre), ctx - steps)
    cap = Capture(layers, capture_steps)
    AttentionInterface.register("capture", cap)
    AttentionMaskInterface.register("capture", sdpa_mask)

    model.set_attn_implementation("probe")
    cache = common.prefill(model, ids[:-1])
    model.set_attn_implementation("capture")
    token, generated = ids[-1], []
    for i in range(max(capture_steps) + 1):
        cap.step = i
        logits = common.step(model, token, cache)
        token, _ = greedy_token(logits, eos_ids)
        generated.append(token)
    model.set_attn_implementation("probe")

    tensors = {}
    T_last = cap.T[cap.last]
    nb_last = math.ceil(T_last / metrics.BLOCK)
    for l in layers:
        k, v = cap.kv[l]
        tensors[f"L{l}.k"] = k.contiguous()
        tensors[f"L{l}.v"] = v.contiguous()
        tensors[f"L{l}.q"] = torch.stack(cap.q[l]).contiguous()
        m = [torch.nn.functional.pad(x, (0, nb_last - x.shape[-1])) for x in cap.masks[l]]
        tensors[f"L{l}.mask099"] = torch.stack(m).to(torch.uint8).contiguous()
    meta = {
        "model": model_name, "genre": genre, "ctx": str(ctx),
        "layers": json.dumps(list(layers)), "steps": json.dumps(sorted(capture_steps)),
        "T": json.dumps([cap.T[s] for s in sorted(capture_steps)]),
        "scaling": repr(cap.scaling), "block": str(metrics.BLOCK),
        "generated": tok.decode(generated),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model_name}_{genre}_{ctx}.safetensors"
    save_file({k: t.cpu() for k, t in tensors.items()}, str(path), metadata=meta)
    kept = {l: float(tensors[f'L{l}.mask099'][-1, :, :math.ceil(T_last / metrics.BLOCK)].float().mean())
            for l in layers}
    print(f"[dump] {path.name}: T={meta['T']} layers={list(layers)} "
          f"union top-p 0.99 kept (last step)={ {l: round(x, 3) for l, x in kept.items()} } "
          f"{path.stat().st_size / 2**20:.0f} MiB", flush=True)
    del cache, cap
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--genre", nargs="+", default=list(common.GENRES))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=128, help="decode budget used to size the prompt (as run_ablate)")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 9, 21, 31])
    ap.add_argument("--capture-steps", nargs="+", type=int, default=[0, 1, 16, 64])
    args = ap.parse_args()
    tok, model = common.load(args.model)
    for genre in args.genre:
        run(args.model, genre, args.ctx, args.steps, args.layers, args.capture_steps, tok, model)


if __name__ == "__main__":
    main()
