"""Stage 2 / A4 inputs: one new document per genre + two controls for the
"chat-template tokens are extra sinks" hypothesis.

  prose2         Project Gutenberg #77919, "Sinbad" (C. Kay Scott, 1923), released 2026-02-11
  code2          transformers 5.19 trainer.py from the decorator of `class Trainer` (line 255)
  dialog2        UltraChat test_sft conversations, shuffled with seed 1
  dialog_plain   the stage-1 dialog as plain text ("User: ... / Assistant: ...") inside
                 one user message: same words, no per-turn template tokens
  prose_svc      the stage-1 prose with a chat-template turn boundary
                 (<|eot_id|><|start_header_id|>role<|end_header_id|>) inserted every
                 2 paragraphs, roughly the dialog's turn frequency

Every question asks about the very beginning of the document, as in stage 1.
Writes data/docs/a4/<name>.json. Run with `python -I src/prepare_a4.py` (reads downloaded data).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from prepare_data import dialog  # noqa: E402

RAW = ROOT / "data" / "raw"
DOCS = ROOT / "data" / "docs"
OUT = DOCS / "a4"
TURN = "<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n\n"


def gutenberg_paragraphs(path: Path, start_marker: str, end_marker: str) -> list[str]:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    body = raw[raw.index(start_marker):raw.index(end_marker)]
    out = []
    for para in re.split(r"\n\s*\n", body):
        text = " ".join(ln.strip() for ln in para.strip().split("\n") if ln.strip())
        if text and text != "[Illustration]":
            out.append(text)
    return out


def prose2() -> dict:
    # "Sinbad" (C. Kay Scott, 1923), released on Project Gutenberg on 2026-02-11 as #77919:
    # an obscure novel, ~80k tokens. (#77915 "Adelbert and Bastel" was too short for 32k.)
    paras = gutenberg_paragraphs(RAW / "gutenberg2" / "pg77919.txt", "\nMIRAGE\n",
                                 "*** END OF THE PROJECT GUTENBERG EBOOK")
    return {
        "genre": "prose", "name": "prose2",
        "source": "Project Gutenberg #77919, Sinbad (C. Kay Scott, 1923), released 2026-02-11",
        "paragraphs": paras,
        "instruction": "Read the following excerpt from a novel carefully.",
        "question": ("Based on the text above: in which city does the novel open, what was the weather like in "
                     "the first scene, who greeted the two figures on the stairs and by what names did she call "
                     "them? Quote the relevant sentences from the text, then describe the opening scene in detail."),
    }


def code2() -> dict:
    import transformers
    src = Path(transformers.__file__).parent / "trainer.py"
    full = src.read_text(encoding="utf-8")
    (RAW / "code").mkdir(parents=True, exist_ok=True)
    (RAW / "code" / "trainer.py").write_text(full, encoding="utf-8")
    text = "".join(full.splitlines(keepends=True)[254:])     # from `@requires(` above `class Trainer`
    paragraphs = re.split(r"(?<=\n)(?=\n)", text)
    return {
        "genre": "code", "name": "code2",
        "source": f"transformers {transformers.__version__} trainer.py from line 255 (copied to data/raw/code/trainer.py)",
        "paragraphs": paragraphs, "joiner": "", "wrap": ["```python\n", "\n```"],
        "instruction": ("Here is an excerpt from the Python source file `transformers/trainer.py` "
                        "of the Hugging Face transformers library."),
        "question": ("Look at the docstring of the `Trainer` class at the very top of this excerpt. What does it "
                     "say about the `model`, `args` and `data_collator` arguments: what is used when `args` or "
                     "`data_collator` are not provided, and what does the Tip say about plain `torch.nn.Module` "
                     "models? Quote the relevant lines."),
    }


def dialog2() -> dict:
    d = dialog(seed=1)
    d["name"] = "dialog2"
    return d


def dialog_plain() -> dict:
    d = json.loads((DOCS / "dialog.json").read_text())
    flat = [m for c in d["conversations"] for m in c["messages"]]
    paras = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in flat]
    return {
        "genre": "prose", "name": "dialog_plain",
        "source": "stage-1 dialog.json as a plain-text transcript in one user message",
        "paragraphs": paras,
        "instruction": "The following is a transcript of a long conversation between a user and an AI assistant.",
        "question": ("Based on the transcript above: what exactly was the very first question the user asked, and "
                     "what were the main points of the assistant's answer to it? Please summarize them in detail."),
    }


def prose_svc(every: int = 2) -> dict:
    d = json.loads((DOCS / "prose.json").read_text())
    paras = []
    for i, p in enumerate(d["paragraphs"]):
        if i and i % every == 0:
            role = "assistant" if (i // every) % 2 else "user"
            p = TURN.format(role=role) + p
        paras.append(p)
    d = dict(d, paragraphs=paras, name="prose_svc",
             source=d["source"] + f"; a chat-template turn boundary before every {every}nd paragraph")
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for doc in (prose2(), code2(), dialog2(), dialog_plain(), prose_svc()):
        (OUT / f"{doc['name']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        n = len(doc.get("paragraphs") or doc.get("conversations"))
        print(f"{doc['name']}: {n} units")


if __name__ == "__main__":
    main()
