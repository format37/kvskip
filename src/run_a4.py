"""Stage 2 / A4: is "dialog is sparser" a property of the genre or of the chat template?

For each document (stage-1 docs + data/docs/a4/*.json) at `ctx`: prefill, then
`steps` greedy decode steps through the exact fp32 probe; from the exact weights
of every layer and step:
  union_frac    share of 64-blocks a KV head needs for 0.99 of each of its 4 heads' mass
  head_frac     the same per query head
  sink          mass on the first 4 tokens
  svc           mass on chat-template service tokens (<|start_header_id|>,
                <|end_header_id|>, <|eot_id|>) at positions >= 4
  svc_share     share of the prompt's tokens that are service tokens
Outputs results/template/<model>_<name>_<ctx>.json (per-layer means) and prints a summary.
"""

import argparse
import json

import numpy as np
import torch
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

import common
import metrics
import probe_attn
from run_collect import greedy_token

OUT = common.ROOT / "results" / "template"
SERVICE_IDS = (128006, 128007, 128009)


class Stats:
    def __init__(self, n_layers, svc_mask):
        self.svc_mask = svc_mask                       # bool [T_prompt] on GPU
        self.acc = {k: torch.zeros(n_layers, device="cuda", dtype=torch.float64)
                    for k in ("union_frac", "head_frac", "sink", "svc")}
        self.n = torch.zeros(n_layers, device="cuda", dtype=torch.float64)

    def __call__(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if query.shape[2] == 1:
            l = module.layer_idx
            Hkv, T, D = key.shape[1:]
            G = query.shape[1] // Hkv
            q = query[0, :, 0].float().view(Hkv, G, D)
            s = torch.einsum("kgd,ktd->kgt", q, key[0].float()) * scaling
            p = torch.softmax(s, -1)
            need = metrics.needed_blocks(metrics.block_weights(p), 0.99)     # [Hkv, G, nb]
            svc = torch.nn.functional.pad(self.svc_mask, (0, T - self.svc_mask.shape[0]), value=False)
            self.acc["union_frac"][l] += need.any(1).float().mean()
            self.acc["head_frac"][l] += need.float().mean()
            self.acc["sink"][l] += p[..., :metrics.SINK_TOKENS].sum(-1).mean()
            self.acc["svc"][l] += p[..., svc].sum(-1).mean()
            self.n[l] += 1
        return probe_attn.probe_attention_forward(module, query, key, value, attention_mask,
                                                  scaling=scaling, dropout=dropout, **kw)


def run(model_name, name, doc, ctx, steps, tok, model):
    eos_ids = list(model.generation_config.eos_token_id)
    ids, _ = common.build_prompt(tok, doc, ctx - 128)
    svc = torch.tensor([t in SERVICE_IDS for t in ids], device="cuda")
    svc[:metrics.SINK_TOKENS] = False
    st = Stats(model.config.num_hidden_layers, svc)
    AttentionInterface.register("a4stats", st)
    AttentionMaskInterface.register("a4stats", sdpa_mask)
    model.set_attn_implementation("probe")
    cache = common.prefill(model, ids[:-1])
    model.set_attn_implementation("a4stats")
    token = ids[-1]
    gen = []
    for _ in range(steps):
        logits = common.step(model, token, cache)
        token, _ = greedy_token(logits, eos_ids)
        gen.append(token)
    model.set_attn_implementation("probe")
    per_layer = {k: (v / st.n).cpu().numpy() for k, v in st.acc.items()}
    res = {"model": model_name, "name": name, "genre": doc["genre"], "ctx": ctx, "steps": steps,
           "prompt_tokens": len(ids), "svc_tokens": int(svc.sum()), "svc_share": float(svc.float().mean()),
           "per_layer": {k: v.tolist() for k, v in per_layer.items()},
           "mean_layers_1_31": {k: float(v[1:].mean()) for k, v in per_layer.items()},
           "generated": tok.decode(gen)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model_name}_{name}_{ctx}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    m = res["mean_layers_1_31"]
    print(f"[a4] {name:13s} union {m['union_frac']:.3f} head {m['head_frac']:.3f} sink {m['sink']:.3f} "
          f"svc {m['svc']:.4f} (svc tokens {res['svc_tokens']}, {100 * res['svc_share']:.2f}% of prompt)  "
          f"{tok.decode(gen[:40])!r}", flush=True)
    del cache
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--docs", nargs="+", default=["prose", "code", "dialog", "prose2", "code2", "dialog2",
                                                  "dialog_plain", "prose_svc"])
    args = ap.parse_args()
    tok, model = common.load(args.model)
    for name in args.docs:
        path = common.ROOT / "data" / "docs" / f"{name}.json"
        if not path.exists():
            path = common.ROOT / "data" / "docs" / "a4" / f"{name}.json"
        run(args.model, name, json.loads(path.read_text()), args.ctx, args.steps, tok, model)


if __name__ == "__main__":
    main()
