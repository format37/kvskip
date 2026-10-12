"""Model loading, prompt construction and the prefill/decode loop shared by the runners."""

import json
import os
from pathlib import Path

# 8B + 32k KV leaves ~2 GB of headroom on a 24 GB card; avoid fragmentation OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import probe_attn

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "1b": ROOT / "models" / "llama-3.2-1b",
    "8b": ROOT / "models" / "llama-3.1-8b",
}
GENRES = ("prose", "code", "dialog")
# Pin the date the chat template injects (Llama 3.2 would otherwise use today's).
DATE_STRING = "26 Jul 2024"
PREFILL_CHUNK = 2048


def load(name: str, attn: str = "probe"):
    probe_attn.register()
    path = MODELS[name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn
    )
    model.eval()
    return tok, model


def load_doc(genre: str) -> dict:
    return json.loads((ROOT / "data" / "docs" / f"{genre}.json").read_text())


def _chat_ids(tok, messages) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                  return_dict=False, date_string=DATE_STRING)
    return list(ids)


def build_prompt(tok, doc: dict, target: int) -> tuple[list[int], int]:
    """Longest prompt of at most `target` tokens: a document prefix + the question.

    Returns (input_ids, n_units) where n_units is the number of paragraphs
    (prose/code) or messages (dialog) of the document that were kept.
    """
    if doc["genre"] == "dialog":
        flat = [m for c in doc["conversations"] for m in c["messages"]]
        units = len(flat) // 2

        def ids_for(n):
            return _chat_ids(tok, flat[: 2 * n] + [{"role": "user", "content": doc["question"]}])
    else:
        paras, joiner = doc["paragraphs"], doc.get("joiner", "\n\n")
        pre, post = doc.get("wrap", ["", ""])
        units = len(paras)

        def ids_for(n):
            body = pre + joiner.join(paras[:n]).strip("\n") + post
            content = f"{doc['instruction']}\n\n{body}\n\n{doc['question']}"
            return _chat_ids(tok, [{"role": "user", "content": content}])

    lo, hi = 1, units
    if len(ids_for(hi)) <= target:
        raise ValueError(f"{doc['genre']}: whole document fits in {target} tokens, need a longer one")
    while lo < hi:                       # largest n with len(ids_for(n)) <= target
        mid = (lo + hi + 1) // 2
        if len(ids_for(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    if doc["genre"] == "dialog":
        return ids_for(lo), 2 * lo
    return ids_for(lo), lo


@torch.inference_mode()
def prefill(model, ids: list[int], chunk: int = PREFILL_CHUNK) -> DynamicCache:
    """Run ids through the model in chunks (SDPA path), return the KV cache."""
    cache = DynamicCache(config=model.config)
    x = torch.tensor(ids, device=model.device)[None]
    for start in range(0, x.shape[1], chunk):
        model(input_ids=x[:, start:start + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


@torch.inference_mode()
def step(model, token: int, cache: DynamicCache) -> torch.Tensor:
    """One decode step (probe path): feed `token`, return fp32 logits [V] for the next one."""
    x = torch.tensor([[token]], device=model.device)
    out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return out.logits[0, -1].float()
