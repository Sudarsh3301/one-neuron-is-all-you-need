"""Direct-path links between the neurons on the map.   python build_links.py   ->  docs/map_links.json

For layers i < j, P[B, A] = (W_gate_j[B] . v) * (W_up_j[B] . v) with v = norm_j * W_down_i[:, A], where norm_j is the
pre-MLP norm weight of layer j. It is the push neuron A of layer i gives neuron B of layer j through the residual
stream. Attention and the residual scale are left out, so this is a ranking of direct links, not a measured effect.
Method as in neuron-explorer's build_connections.py, with the Qwen3 norm (a plain weight, not 1 + weight).

For each neuron on the map: its K strongest links down (neurons it writes into) and up (neurons that write into it),
over all later or earlier layers. |P| is z-scored over the other side: over B for a writer, over A for a reader.
Edges: every pair of mapped neurons whose z (over A, for the reader) reaches EDGE_Z, strongest MAX_EDGES kept.
Needs the bf16 shards in models/qwen3-8b-bf16 and a CUDA GPU (the desktop's 6 GB card is enough).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).parent
W = ROOT / "models/qwen3-8b-bf16"
K, EDGE_Z, MAX_EDGES = 12, 8.0, 20000  # z >= 8 is about 0.1% of random pairs; z >= 4 is about 1% (see check_links.py)
dev = "cuda"

index = json.load(open(W / "model.safetensors.index.json", encoding="utf-8"))["weight_map"]
data = json.load(open(ROOT / "docs/map_data.json", encoding="utf-8"))
L, N = data["arch"]["n_layers"], data["arch"]["inter"]
idx = [torch.tensor(sorted(r[1] for r in data["scored"] if r[0] == l), dtype=torch.long, device=dev) for l in range(L)]
print("mapped neurons per layer:", [len(i) for i in idx], flush=True)


def tensor(name, dtype=torch.bfloat16):
    with safe_open(W / index[name], "pt") as f:
        return f.get_tensor(name).to(dev, dtype)


def layer_w(kind, l, dtype=torch.bfloat16):
    return tensor(f"model.layers.{l}.{kind}", dtype)


def new_best(rows):
    return [torch.full((rows, K), -1e9, device=dev), torch.zeros((rows, K), dtype=torch.long, device=dev),
            torch.zeros((rows, K), dtype=torch.long, device=dev), torch.zeros((rows, K), dtype=torch.long, device=dev)]


def merge(best, z, layer, n, sign):  # z, n, sign: [rows, k]; layer: int
    cz = torch.cat([best[0], z], 1)
    cl = torch.cat([best[1], torch.full_like(n, layer)], 1)
    cn = torch.cat([best[2], n], 1)
    cs = torch.cat([best[3], sign], 1)
    top = cz.topk(K, dim=1)
    best[:] = [top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)]


def zscore(M, dim):
    return (M - M.mean(dim, keepdim=True)) / (M.std(dim, keepdim=True) + 1e-8)


t0 = time.time()
down = [layer_w("mlp.down_proj.weight", l) for l in range(L)]  # [hidden, N] each, 2.8 GB in all
print(f"loaded down_proj ({time.time() - t0:.0f}s)", flush=True)
reader = [new_best(len(idx[l])) for l in range(L)]  # links up, for the mapped neurons of layer l
writer = [new_best(len(idx[l])) for l in range(L)]  # links down
edges = []
with torch.inference_mode():
    for j in range(1, L):
        g, u, nw = layer_w("mlp.gate_proj.weight", j), layer_w("mlp.up_proj.weight", j), layer_w("post_attention_layernorm.weight", j)
        for i in range(j):
            V = down[i] * nw[:, None]  # [hidden, N_i]
            if len(idx[j]):  # readers: row B over every A of layer i
                P = (g[idx[j]] @ V).float() * (u[idx[j]] @ V).float()  # [|S_j|, N_i]
                z = zscore(P.abs(), 1)
                top = z.topk(K, dim=1)
                merge(reader[j], top.values, i, top.indices, torch.where(P.gather(1, top.indices) > 0, 1, -1))
                if len(idx[i]):
                    zs, ps = z[:, idx[i]], P[:, idx[i]]
                    b, a = (zs >= EDGE_Z).nonzero(as_tuple=True)
                    if len(b):
                        edges.append(np.stack([np.full(len(b), i), idx[i][a].cpu().numpy(), np.full(len(b), j), idx[j][b].cpu().numpy(),
                                               np.round(zs[b, a].cpu().numpy() * 10) * np.sign(ps[b, a].cpu().numpy())], 1))
            if len(idx[i]):  # writers: column A over every B of layer j
                Vi = V[:, idx[i]]
                P = (g @ Vi).float() * (u @ Vi).float()  # [N_j, |S_i|]
                z = zscore(P.abs(), 0)
                top = z.topk(K, dim=0)
                merge(writer[i], top.values.T, j, top.indices.T, torch.where(P.gather(0, top.indices) > 0, 1, -1).T)
        print(f"reader layer {j} done ({time.time() - t0:.0f}s)", flush=True)


def pack(best, rows):
    z, l, n, s = (t.cpu().numpy() for t in best)
    return [[[int(l[r, k]), int(n[r, k]), int(round(float(z[r, k]) * 10)) * int(s[r, k])] for k in range(K) if z[r, k] > -1e8] for r in range(len(rows))]


links = {}
for l in range(L):
    rows = idx[l].cpu().numpy()
    d, u = pack(writer[l], rows), pack(reader[l], rows)
    for r, n in enumerate(rows):
        links[f"L{l}.N{int(n)}"] = {"d": d[r], "u": u[r]}
E = np.concatenate(edges)
E = E[np.argsort(-np.abs(E[:, 4]), kind="stable")][:MAX_EDGES].astype(int)
out = ROOT / "docs/map_links.json"
out.write_text(json.dumps({"k": K, "edge_z": EDGE_Z, "n_edges_total": int(sum(len(e) for e in edges)), "links": links, "edges": E.tolist()}, separators=(",", ":")), encoding="utf-8")
print(f"{len(links)} neurons, {len(E)} of {sum(len(e) for e in edges)} edges kept, {out.stat().st_size / 1e6:.2f} MB", flush=True)

# check: recompute one neuron's downstream links slowly in float32 and compare with the batched, bf16 result
with torch.inference_mode():
    l0, n0 = 9, 598
    cand = []
    for j in range(l0 + 1, L):
        v = layer_w("post_attention_layernorm.weight", j, torch.float32) * down[l0][:, n0].float()
        P = (layer_w("mlp.gate_proj.weight", j, torch.float32) @ v) * (layer_w("mlp.up_proj.weight", j, torch.float32) @ v)
        z = zscore(P.abs(), 0)
        t = z.topk(K)
        cand += [(float(a), j, int(b)) for a, b in zip(t.values, t.indices)]
    ref = {(j, b) for _, j, b in sorted(cand, reverse=True)[:K]}
    got = {(a, b) for a, b, _ in links[f"L{l0}.N{n0}"]["d"]}
    print(f"check L{l0}.N{n0} downstream: {len(ref & got)} of {K} links agree between float32 slow path and batched bf16", flush=True)
