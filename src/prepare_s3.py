"""Stage 3 inputs: documents long enough for 64k-token prompts (8 per batch).

The stage-1/2 code and dialog documents end at 44-47k tokens, so the 64k batch
needs new ones. Stage 3 uses 8 documents, 3 prose / 3 code / 2 dialog:

  prose          stage 1 (Gutenberg #77931), 87k tokens
  prose2         stage 2 held-out (Gutenberg #77919), 72k tokens
  prose3         Gutenberg #77959, "The Gold of Ophir" (D. Howard Gwinn, 1898), released 2026-02-16
  code3          torch 2.14 torch/_inductor/scheduler.py
  code4          torch 2.14 torch/fx/experimental/symbolic_shapes.py
  code5          torch 2.14 torch/_inductor/ir.py
  dialog3        UltraChat test_sft conversations, shuffled with seed 2, ~340k characters
  dialog4        the same with seed 3

Every question asks about the very beginning of the document, as in stages 1-2.
Writes data/docs/s3/<name>.json (prose and prose2 are referenced, not copied).
Run with `python -I src/prepare_s3.py` (reads downloaded data).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from prepare_a4 import gutenberg_paragraphs  # noqa: E402
from prepare_data import code, dialog  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "docs" / "s3"
DOCS = ("prose", "prose2", "prose3", "code3", "code4", "code5", "dialog3", "dialog4")


def prose3() -> dict:
    paras = gutenberg_paragraphs(RAW / "gutenberg3" / "pg77959.txt", "\nTHE GOLD OF OPHIR.\n",
                                 "*** END OF THE PROJECT GUTENBERG EBOOK")
    return {
        "genre": "prose", "name": "prose3",
        "source": "Project Gutenberg #77959, The Gold of Ophir (D. Howard Gwinn, 1898), released 2026-02-16",
        "paragraphs": paras,
        "instruction": "Read the following excerpt from a novel carefully.",
        "question": ("Based on the text above: what is the narrator's full name, in which city was he born, where "
                     "did he go to school and university, and with whom did he live? Quote the relevant sentences "
                     "from the text, then describe how the narrator introduces himself in detail."),
    }


def torch_code(rel: str, name: str, question: str) -> dict:
    import torch
    src = Path(torch.__file__).parent / rel
    d = code(src, 1)
    d.update(name=name, question=question,
             source=f"torch {torch.__version__} torch/{rel} (copied to data/raw/code/{src.name})",
             instruction=f"Here is the Python source file `torch/{rel}` of the PyTorch library.")
    return d


def code3() -> dict:
    return torch_code("_inductor/scheduler.py", "code3",
                      "Look at the `FusionResult` dataclass near the top of this file. Which fields does it have, "
                      "what does its `__post_init__` check, and which two classmethods construct it? Quote the "
                      "relevant lines, then explain how it is meant to be used.")


def code4() -> dict:
    return torch_code("fx/experimental/symbolic_shapes.py", "code4",
                      "Look at the module docstring at the very top of this file. What does it say "
                      "`torch.fx.experimental.symbolic_shapes` provides, what is it used heavily in, and when might "
                      "you need these APIs even though they are not considered public API? Quote the docstring, "
                      "then explain it in your own words.")


def code5() -> dict:
    return torch_code("_inductor/ir.py", "code5",
                      "Look at the [Note: Inductor IR] docstring near the top of this file. How does it describe "
                      "TensorBox, View, StorageBox and Buffer, and which chain of IR objects represents a tensor "
                      "that owns its storage and which one a view? Quote the relevant lines, then explain them.")


def long_dialog(seed: int) -> dict:
    d = dialog(seed=seed, budget_chars=340_000)
    d["name"] = f"dialog{seed + 1}"
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for doc in (prose3(), code3(), code4(), code5(), long_dialog(2), long_dialog(3)):
        (OUT / f"{doc['name']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        n = len(doc.get("paragraphs") or doc.get("conversations"))
        print(f"{doc['name']}: {n} units")


if __name__ == "__main__":
    main()
