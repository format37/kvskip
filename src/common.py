"""Model loading, prompt construction and the prefill/decode loop shared by the runners.

Input: models/<name> (bf16 safetensors from Hugging Face, see README) and data/docs/{,a4/,s3/}<name>.json."""

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
    "qwen3-8b": ROOT / "models" / "qwen3-8b",
    "qwen2.5-7b": ROOT / "models" / "qwen2.5-7b",      # stage 4: Qwen2.5-7B-Instruct (no QK-norm)
    "mistral-7b": ROOT / "models" / "mistral-7b",      # stage 4: Mistral-7B-Instruct-v0.3 (map only)
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


def eos_ids(model) -> list[int]:
    """The model's end-of-sequence ids as a list (Mistral's generation config holds one int)."""
    eos = model.generation_config.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos)


def load_doc(genre: str) -> dict:
    return json.loads((ROOT / "data" / "docs" / f"{genre}.json").read_text())


def _chat_ids(tok, messages) -> list[int]:
    # date_string: Llama 3.x templates; enable_thinking: Qwen3 (answer directly). Templates
    # ignore variables they do not use, so the Llama prompts are unchanged.
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                  return_dict=False, date_string=DATE_STRING, enable_thinking=False)
    return list(ids)


def _prompt_fn(tok, doc: dict):
    """(ids_for, units): ids_for(n) is the prompt with the first n units of the document
    (paragraphs, or user/assistant message pairs for dialogs) followed by the question."""
    if doc["genre"] == "dialog":
        flat = [m for c in doc["conversations"] for m in c["messages"]]

        def ids_for(n):
            return _chat_ids(tok, flat[: 2 * n] + [{"role": "user", "content": doc["question"]}])
        return ids_for, len(flat) // 2
    paras, joiner = doc["paragraphs"], doc.get("joiner", "\n\n")
    pre, post = doc.get("wrap", ["", ""])

    def ids_for(n):
        body = pre + joiner.join(paras[:n]).strip("\n") + post
        content = f"{doc['instruction']}\n\n{body}\n\n{doc['question']}"
        return _chat_ids(tok, [{"role": "user", "content": content}])
    return ids_for, len(paras)


def _units_fitting(ids_for, units: int, target: int, genre: str) -> int:
    lo, hi = 1, units
    if len(ids_for(hi)) <= target:
        raise ValueError(f"{genre}: whole document fits in {target} tokens, need a longer one")
    while lo < hi:                       # largest n with len(ids_for(n)) <= target
        mid = (lo + hi + 1) // 2
        if len(ids_for(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def build_prompt(tok, doc: dict, target: int) -> tuple[list[int], int]:
    """Longest prompt of at most `target` tokens: a document prefix + the question.

    Returns (input_ids, n_units) where n_units is the number of paragraphs
    (prose/code) or messages (dialog) of the document that were kept.
    """
    ids_for, units = _prompt_fn(tok, doc)
    lo = _units_fitting(ids_for, units, target, doc["genre"])
    if doc["genre"] == "dialog":
        return ids_for(lo), 2 * lo
    return ids_for(lo), lo


def exact_prompt(tok, doc: dict, target: int) -> list[int]:
    """A prompt of exactly `target` tokens (stage 3: the kernels take one length per batch).

    build_prompt's longest whole-unit prompt, topped up with the leading tokens of the next
    unit: the prompt with one more unit is cut just before the part it shares with the
    shorter one (the question and the generation prompt; for dialogs from the question's
    header on, keeping the cut message's closing <|eot_id|>)."""
    ids_for, units = _prompt_fn(tok, doc)
    lo = _units_fitting(ids_for, units, target, doc["genre"])
    a, b = ids_for(lo), ids_for(lo + 1)
    if len(a) == target:
        return a
    c = 0
    while c < len(a) and a[-1 - c] == b[-1 - c]:
        c += 1
    keep = []
    if doc["genre"] == "dialog":
        start_header = tok.convert_tokens_to_ids("<|start_header_id|>")
        while b[len(b) - c] != start_header:
            c -= 1
        keep = b[len(b) - c - 1:len(b) - c]                  # the cut message's <|eot_id|>
    head, tail = b[:len(b) - c - len(keep)], b[len(b) - c:]
    ids = head[:target - len(tail) - len(keep)] + keep + tail
    assert len(ids) == target and len(head) > target - len(tail) - len(keep), (len(ids), target)
    return ids


def load_named_doc(name: str) -> dict:
    """data/docs/<name>.json, data/docs/a4/<name>.json or data/docs/s3/<name>.json."""
    for sub in ("", "a4", "s3"):
        p = ROOT / "data" / "docs" / sub / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(name)


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
