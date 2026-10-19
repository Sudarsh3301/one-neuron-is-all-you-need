"""Logit lens on the steered neurons: which tokens does each write vector push up, and which down?

Reads the output norm and output matrix from the Q8_0 GGUF (no extra download) and the bf16
down_proj columns in vectors/qwen3-8b/. A direct read of a mid-layer vector is only a rough
picture, because the later layers transform it before it reaches the output.

  python lens.py            # -> results/lens.md
"""
import argparse

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoTokenizer

from llama import ROOT

STEERED = {"L18.N7807": 32, "L8.N1931": 24, "L9.N3757": -32, "L8.N5315": -32}


def load_unembed(path):
    t = {x.name: x for x in GGUFReader(str(path)).tensors}
    norm = np.asarray(t["output_norm.weight"].data, dtype=np.float32).reshape(-1)
    out = t.get("output.weight") or t["token_embd.weight"]  # tied models have no output.weight
    W = dequantize(out.data, out.tensor_type).reshape(-1, norm.size)  # [n_vocab, d_model]
    return norm, W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default=str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf"))
    p.add_argument("--k", type=int, default=20)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    norm, W = load_unembed(a.gguf)
    print(f"output matrix {W.shape}, norm {norm.shape}")
    show = lambda ids: ", ".join(repr(tok.decode([int(i)])) for i in ids)

    out = ["# Logit lens of the steered neurons", "",
           "Tokens that each neuron's write vector pushes up and down, in its steering direction, "
           "read straight through the final norm and output matrix.", ""]
    for unit, alpha in STEERED.items():
        v = np.sign(alpha) * np.load(ROOT / "vectors" / "qwen3-8b" / f"{unit}.npy")
        logits = W @ (norm * v)
        order = np.argsort(logits)
        out += [f"## {unit} at {alpha:+d}", "",
                f"**Up:** {show(order[::-1][:a.k])}", "",
                f"**Down:** {show(order[:a.k])}", ""]
    text = "\n".join(out)
    (ROOT / "results" / "lens.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
