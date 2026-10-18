"""Check that a llama.cpp control vector at index L equals steering neuron N of layer L with a down_proj hook.

Uses Qwen3-0.6B: HF bf16 on the GPU, and llama.cpp with the Q8_0 GGUF.
PASS when the vector at index L gives the hook's top-1 next token and at least 7 of its top-10,
and does better than both the vector at index L+1 and no vector at all.
"""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama import ROOT, Server
from vectors import write_cvec

PROMPT = "The capital of France is"


def hf_top(model, ids, layer=None, neuron=None, alpha=0.0, n=10):
    h = None
    if layer is not None:
        def hook(m, a):
            x = a[0].clone()
            x[..., neuron] += alpha
            return (x,) + a[1:]
        h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=model.device)).logits[0, -1].float()
    finally:
        if h:
            h.remove()
    return logits.topk(n).indices.tolist()


def compare(a, b):
    """(top-1 equal, size of top-10 overlap)."""
    return a[0] == b[0], len(set(a) & set(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", default="Qwen/Qwen3-0.6B")
    p.add_argument("--gguf", default=str(ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--g", type=float, default=1.0, help="dose as a multiple of |residual| / |W_down[:, N]|")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok(PROMPT)["input_ids"]
    W = model.model.layers[a.layer].mlp.down_proj.weight.detach().float()  # [d_model, n_neurons]
    norms = W.norm(dim=0)
    neuron = int(norms.argmax())
    with torch.no_grad():
        hs = model(torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states
    resid = float(hs[a.layer + 1][0, -1].float().norm())  # hidden_states[L+1] is the output of layer L
    alpha = a.g * resid / float(norms[neuron])
    vec = (alpha * W[:, neuron]).cpu().numpy()
    print(f"layer {a.layer}, neuron {neuron}, |W col| {float(norms[neuron]):.3f}, |resid| {resid:.1f}, alpha {alpha:.1f}")

    hf_base = hf_top(model, ids)
    hf_steer = hf_top(model, ids, a.layer, neuron, alpha)
    del model
    torch.cuda.empty_cache()
    top1, ov = compare(hf_base, hf_steer)
    if top1 and ov >= 8:
        sys.exit(f"steering barely changed the HF top-10 (overlap {ov}); rerun with a larger --g")

    rows = {}
    for idx in (a.layer, a.layer + 1):
        path = ROOT / "vectors" / "mapping" / f"idx{idx}.gguf"
        write_cvec(path, idx, vec)
        with Server(a.gguf, cvec=path, parallel=1) as s:
            rows[idx] = compare(hf_steer, s.top_ids(ids))
    with Server(a.gguf, parallel=1) as s:
        rows["none"] = compare(hf_steer, s.top_ids(ids))
    for k, (t1, o) in rows.items():
        print(f"control vector at {k}: top-1 match {t1}, top-10 overlap {o}/10 with the HF hook")
    t1, o = rows[a.layer]
    ok = t1 and o >= 7 and o > max(rows[a.layer + 1][1], rows["none"][1])
    print("PASS: direction.L equals the hook on layer L" if ok else "FAIL: mapping not confirmed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
