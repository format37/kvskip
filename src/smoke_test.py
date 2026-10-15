"""Kickoff step 1: load the 1B in bf16, generate 50 tokens, switch attention
implementations (sdpa / eager / probe) and check that they agree.

Also checks the probe decode path numerically: attention output against an
fp64 reference on real q/k/v, and next-token logits against HF eager/sdpa.

Input: models/llama-3.2-1b, data/docs/prose.json.
"""

import time

import torch
from transformers.models.llama import modeling_llama

import common
import probe_attn


def generate(model, tok, prompt_ids, n=50):
    x = torch.tensor([prompt_ids], device=model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(x, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return out[0, len(prompt_ids):].tolist(), time.perf_counter() - t0


def attention_unit_test(model, prompt_ids):
    """Probe decode attention vs fp64 on the real q/k/v of every layer."""
    captured = {}

    def capture(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if query.shape[2] == 1:
            captured[module.layer_idx] = (query.clone(), key.clone(), value.clone(), scaling)
        return probe_attn.probe_attention_forward(module, query, key, value, attention_mask,
                                                  scaling=scaling, dropout=dropout, **kw)

    from transformers import AttentionInterface
    AttentionInterface.register("capture", capture)
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionMaskInterface.register("capture", sdpa_mask)
    model.set_attn_implementation("capture")
    cache = common.prefill(model, prompt_ids[:-1])
    common.step(model, prompt_ids[-1], cache)
    model.set_attn_implementation("probe")

    worst = {"probe_fp32": 0.0, "probe_bf16_out": 0.0, "sdpa_bf16": 0.0, "eager_bf16": 0.0}
    for layer, (q, k, v, scaling) in captured.items():
        G = q.shape[1] // k.shape[1]
        k64 = k.double().repeat_interleave(G, 1)
        v64 = v.double().repeat_interleave(G, 1)
        p64 = torch.softmax(q.double() @ k64.transpose(-1, -2) * scaling, -1)
        ref = (p64 @ v64).transpose(1, 2)                                   # [1, 1, H, D]
        probe = probe_attn.decode_attention(q, k, v, None, scaling, layer)
        outs = {
            "probe_fp32": probe,
            "probe_bf16_out": probe.to(q.dtype),
            "sdpa_bf16": torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(G, 1), v.repeat_interleave(G, 1), scale=scaling).transpose(1, 2),
            "eager_bf16": modeling_llama.eager_attention_forward(
                type("M", (), {"num_key_value_groups": G, "training": False})(), q, k, v, None, scaling)[0],
        }
        for name, o in outs.items():
            err = ((o.double() - ref).norm() / ref.norm()).item()
            worst[name] = max(worst[name], err)
    return worst


@torch.inference_mode()
def teacher_forced_logits(model, prompt_ids, tokens):
    cache = common.prefill(model, prompt_ids[:-1])
    feed = [prompt_ids[-1]] + tokens[:-1]
    return torch.stack([torch.log_softmax(common.step(model, t, cache), -1) for t in feed])


def main():
    tok, model = common.load("1b", attn="sdpa")
    print("loaded:", model.config._attn_implementation, model.dtype,
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    prompt = common._chat_ids(tok, [{"role": "user", "content":
                                     "Explain in a few sentences why attention in transformers is quadratic."}])
    gens = {}
    for impl in ("sdpa", "eager", "probe"):
        model.set_attn_implementation(impl)
        toks, dt = generate(model, tok, prompt)
        gens[impl] = toks
        print(f"[{impl:5s}] config._attn_implementation={model.config._attn_implementation!r} "
              f"{len(toks)} tokens in {dt:.2f}s")
    print("text (sdpa):", repr(tok.decode(gens["sdpa"])))
    for impl in ("eager", "probe"):
        same = next((i for i, (a, b) in enumerate(zip(gens["sdpa"], gens[impl])) if a != b), None)
        print(f"greedy {impl} vs sdpa: " + ("identical" if same is None else f"first differs at token {same}"))

    # Numerical checks on a longer real prompt (prose, ~4k tokens).
    doc = common.load_doc("prose")
    ids, _ = common.build_prompt(tok, doc, 4096)
    model.set_attn_implementation("probe")
    worst = attention_unit_test(model, ids)
    print("decode attention rel. error vs fp64 (worst layer):",
          {k: f"{v:.2e}" for k, v in worst.items()})

    model.set_attn_implementation("probe")
    toks, _ = generate(model, tok, ids, n=16)
    lp = {}
    for impl in ("probe", "sdpa", "eager"):
        model.set_attn_implementation(impl)
        lp[impl] = teacher_forced_logits(model, ids, toks)
    for impl in ("sdpa", "eager"):
        kl = (lp["probe"].exp() * (lp["probe"] - lp[impl])).sum(-1)
        agree = (lp["probe"].argmax(-1) == lp[impl].argmax(-1)).float().mean().item()
        print(f"4k prose, 16 decode steps, KL(probe || {impl}) mean={kl.mean():.2e} max={kl.max():.2e} "
              f"argmax agreement={agree:.3f}")


if __name__ == "__main__":
    main()
