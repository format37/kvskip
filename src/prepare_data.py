"""Build the three long-context documents (prose, code, dialog) from data/raw.

Each document is saved as data/docs/<genre>.json and later cut to a target
token length by common.build_prompt. The question always refers to the very
beginning of the document, so its answer lies far from the end at every
context length.

Run with `python -I src/prepare_data.py` (the raw files are downloaded data).
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "docs"


def prose() -> dict:
    # "The Moors and the Fens", vol. 1 (F. G. Trafford = Mrs. J. H. Riddell, 1858).
    # Released on Project Gutenberg on 2026-02-14, i.e. after the Llama 3.x
    # training cutoff, so the model cannot know it by heart.
    raw = (RAW / "gutenberg" / "pg77931.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = raw.index("CHAPTER I.")
    end = raw.index("END OF VOLUME I.")
    body = raw[start:end]
    paragraphs = []
    for para in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in para.strip().split("\n")]
        text = " ".join(ln for ln in lines if ln)
        if text:
            paragraphs.append(text)
    return {
        "genre": "prose",
        "source": "Project Gutenberg #77931, The Moors and the Fens, vol. 1 (1858)",
        "paragraphs": paragraphs,
        "instruction": "Read the following novel excerpt carefully.",
        "question": (
            "Based on the text above: what is the name of Sir Ernest Ivraine's house, "
            "who originally gave it that name, and what was the only pretty or cheering "
            "object visible from its windows? Quote the relevant sentences from the text, "
            "then describe the house and its surroundings in detail."
        ),
    }


def code(src: Path, start_line: int) -> dict:
    full = src.read_text(encoding="utf-8")
    (RAW / "code").mkdir(parents=True, exist_ok=True)
    (RAW / "code" / src.name).write_text(full, encoding="utf-8")
    text = "".join(full.splitlines(keepends=True)[start_line - 1:])
    # Cut at blank lines so that truncation never splits a statement;
    # "".join(paragraphs) == text.
    paragraphs = re.split(r"(?<=\n)(?=\n)", text)
    assert "".join(paragraphs) == text
    return {
        "genre": "code",
        "source": f"{src} from line {start_line} (copied to data/raw/code/{src.name})",
        "paragraphs": paragraphs,
        "joiner": "",
        "wrap": ["```python\n", "\n```"],
        "instruction": (f"Here is an excerpt from the Python source file `{src.parent.name}/{src.name}` "
                        f"of the Hugging Face transformers library."),
        "question": None,  # filled in by the caller once the file is chosen
    }


def dialog(seed: int = 0, min_messages: int = 10, budget_chars: int = 220_000) -> dict:
    df = pd.read_parquet(RAW / "ultrachat" / "test_sft.parquet")
    df = df[df["messages"].map(len) >= min_messages]
    order = list(range(len(df)))
    random.Random(seed).shuffle(order)
    convs, total = [], 0
    for i in order:
        msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i]["messages"]]
        if msgs[0]["role"] != "user" or len(msgs) % 2:
            continue
        convs.append({"prompt_id": df.iloc[i]["prompt_id"], "messages": msgs})
        total += sum(len(m["content"]) for m in msgs)
        if total >= budget_chars:
            break
    return {
        "genre": "dialog",
        "source": f"HuggingFaceH4/ultrachat_200k test_sft, {len(convs)} conversations "
                  f"with >= {min_messages} messages, shuffled with seed {seed}, concatenated",
        "conversations": convs,
        "question": (
            "Let's go back to the very beginning of our conversation. What exactly was the "
            "first question I asked you, and what were the main points of your answer? "
            "Please summarize them in detail."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-src", type=Path, required=True)
    ap.add_argument("--code-start-line", type=int, default=1)
    ap.add_argument("--code-question", required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    c = code(args.code_src, args.code_start_line)
    c["question"] = args.code_question
    for doc in (prose(), c, dialog()):
        (OUT / f"{doc['genre']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        n = len(doc.get("paragraphs") or doc.get("conversations"))
        print(f"{doc['genre']}: {n} units -> {OUT / (doc['genre'] + '.json')}")


if __name__ == "__main__":
    main()
