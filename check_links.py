"""Do the top persona neurons link to each other more than random neurons of the same layers do?   python check_links.py

For each figure: the TOPN neurons with the highest persona score (layers 1-27), and the share of their pairs whose
direct-path z (see build_links.py) reaches 4 and 8, against NRAND random sets with the same number of neurons in each
layer. Adds the summary under "null" in docs/map_links.json, so run build_links.py first.
Caveat: the persona score already favours neurons that write along one shared direction, so a high link share is
not by itself proof of a causal circuit."""
import json, random, sys
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
ROOT = Path(__file__).resolve().parent
W = ROOT / "models/qwen3-8b-bf16"; dev = "cuda"
index = json.load(open(W / "model.safetensors.index.json", encoding="utf-8"))["weight_map"]
data = json.load(open(ROOT / "docs/map_data.json", encoding="utf-8"))
L, N, TOPN, NRAND = 28, 12288, 200, 5
def lw(kind, l):
    with safe_open(W / index[f"model.layers.{l}.{kind}"], "pt") as f: return f.get_tensor(f"model.layers.{l}.{kind}").to(dev, torch.bfloat16)
down = [lw("mlp.down_proj.weight", l) for l in range(L)]
gates = {}
def gu(j):
    if j not in gates: gates[j] = (lw("mlp.gate_proj.weight", j), lw("mlp.up_proj.weight", j), lw("post_attention_layernorm.weight", j))
    return gates[j]
def pair_z(group):  # group: {layer: [neuron ids]}; z of every in-group pair (A in layer i < B in layer j), z over all A of layer i
    out = []
    with torch.inference_mode():
        for j in range(1, L):
            if not group.get(j): continue
            g, u, nw = gu(j); R = torch.tensor(group[j], device=dev)
            for i in range(j):
                if not group.get(i): continue
                V = down[i] * nw[:, None]
                P = (g[R] @ V).float() * (u[R] @ V).float(); M = P.abs()
                z = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
                out.append(z[:, torch.tensor(group[i], device=dev)].flatten().cpu().numpy())
            if len(gates) > 6: gates.pop(next(iter(gates)))
    return np.concatenate(out)
def stats(z): return dict(pairs=len(z), ge4=round(float((z >= 4).mean()), 4), ge8=round(float((z >= 8).mean()), 4), mean=round(float(z.mean()), 3))
random.seed(0)
summary = {}
for f, fig in enumerate(data["figures"]):
    rows = sorted((r for r in data["scored"] if r[0] >= 1), key=lambda r: -r[2][f])[:TOPN]
    grp = {}
    for r in rows: grp.setdefault(r[0], []).append(r[1])
    real = stats(pair_z(grp))
    rnd = [stats(pair_z({l: random.sample(range(N), len(ns)) for l, ns in grp.items()})) for _ in range(NRAND)]
    print(fig["key"], "real", real, "| random", [(r["ge4"], r["ge8"], r["mean"]) for r in rnd], flush=True)
    summary[fig["key"]] = {"top_n": TOPN, "pairs": real["pairs"], "real_ge4": real["ge4"], "real_ge8": real["ge8"],
                           "random_ge4": round(float(np.mean([r["ge4"] for r in rnd])), 4), "random_ge8": round(float(np.mean([r["ge8"] for r in rnd])), 4)}
p = ROOT / "docs/map_links.json"
d = json.loads(p.read_text(encoding="utf-8")); d["null"] = summary
p.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
print("wrote null summary to", p)
