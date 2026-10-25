"""Build docs/map_data.json for docs/map.html (the 3D layer map).   python build_map.py

Tier A: neurons with full, labelled replies (v1 and v2).  Tier B: discovery screening samples, no labels.
Tier C: persona scores for every neuron in layers 0..27; only the top TOP per figure are kept.
Inputs under a `private` path are refused.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
N_LAYERS, N_NEURONS, TOP = 28, 12288, 500  # layers 0..27 are the ones the persona work extracted
MAX_REPLIES, MAX_CHARS = 3, 400
FIGS = [("hitler", "Hitler", "#CC79A7"), ("yellow_emperor", "Yellow Emperor", "#E69F00"),
        ("napoleon", "Napoleon", "#0072B2"), ("mao", "Mao", "#009E73")]  # names and colours as in build_explorer_v2.py
SCORES = ROOT / "vectors/personas/_align_scores_hitler_yellow_emperor_napoleon_mao.npz"
NORMS = ROOT / "vectors/personas/col_norms.npy"

# the dose that produces the effect, and a one-line reading (README results 1, 2, 4 and the v2 table)
A_UNITS = {
    "L18.N7807": (32, "Biblical God"), "L8.N1931": (24, "Scripture more than God"),
    "L9.N3757": (-32, "weather god, a generic deity"), "L8.N5315": (-32, "nature poetry"),
    "L9.N598": (-32, "Yellow Emperor (the source's neuron)"), "L9.N10974": (-32, "Napoleon (predicted)"),
    "L10.N5809": (-32, "Hitler: German-nationalist leader voice"),
}


def read(path):
    assert "private" not in str(path), path
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def unit_of(rid):  # "L9.N598|-32|39|0" -> ("L9.N598", -32)
    u, a = rid.split("|")[:2]
    return u, int(a)


def build_doses(rows, labels):
    by = {}
    for r in rows:
        u, a = unit_of(r["id"])
        by.setdefault((u, a), []).append(r)
    out = {}
    for (u, a), rs in by.items():
        rs.sort(key=lambda r: (int(r["q"]), int(r["sample"])))
        pick = [r for r in rs if int(r["sample"]) == 0][:MAX_REPLIES]
        d = {"n": len(rs), "replies": [{"q": r["question"], "t": r["text"][:MAX_CHARS],
                                        **({"l": labels[r["id"]]} if labels else {})} for r in pick]}
        if labels:
            d["mix"] = dict(Counter(labels[r["id"]] for r in rs))
        out.setdefault(u, {})[str(a)] = d
    return out


units = {}
# tier A
v1 = [r for r in read(ROOT / "results/replies.jsonl") if r["unit"] != "none"]
l1 = {r["id"]: r["label"] for r in read(ROOT / "results/labels_strict.jsonl")}
v2 = read(ROOT / "results/v2/replies.jsonl")
l2 = {r["id"]: r["label"] for r in read(ROOT / "results/v2/labels_v2.jsonl")}
for rows, labels in ((v1, l1), (v2, l2)):
    for u, doses in build_doses(rows, labels).items():
        dose, note = A_UNITS[u]
        L, N = (int(x) for x in u[1:].split(".N"))
        units[u] = {"layer": L, "n": N, "tier": "A", "dose": dose, "note": note, "doses": doses}
assert set(units) == set(A_UNITS), set(A_UNITS) ^ set(units)
assert [units[u]["layer"] for u in ("L18.N7807", "L8.N1931", "L9.N3757", "L8.N5315")] == [18, 8, 9, 8]

# tier B
cands = {}
for c in json.load(open(ROOT / "results/v2/candidates.json", encoding="utf-8")):
    cands.setdefault(c["unit"], c)
disc = build_doses(read(ROOT / "results/v2/discovery.jsonl"), None)
for u, doses in disc.items():
    if u in units or not re.fullmatch(r"L\d+\.N\d+", u):  # discovery also holds whole-direction vectors ("vec:...")
        continue
    L, N = (int(x) for x in u[1:].split(".N"))
    c = cands.get(u, {})
    note = "screening" + (f" for {c['persona']}" if c.get("persona") else "")
    if c.get("figure_identity"):
        note += f"; judged identity: {c['figure_identity']}"
    first = sorted(doses, key=lambda a: abs(int(a)))[-1]
    units[u] = {"layer": L, "n": N, "tier": "B", "dose": int(first), "note": note, "doses": doses}
    if c.get("lens"):
        units[u]["lens"] = c["lens"][:10]

# tier C
z = np.load(SCORES)
assert list(z.files) == [f[0] for f in FIGS], z.files
S = np.stack([z[f[0]] for f in FIGS])  # [4, 28, 12288]
assert S.shape == (4, N_LAYERS, N_NEURONS), S.shape
norms = np.load(NORMS)
assert norms.shape == (N_LAYERS, N_NEURONS)
summary = json.load(open(ROOT / "results/v2/summary.json", encoding="utf-8"))["figures"]
keep, cut, fmax = set(), [], []
for i, (key, _, _) in enumerate(FIGS):
    flat = S[i].reshape(-1)
    order = np.argsort(-flat)[:TOP]
    keep.update(int(o) for o in order)
    cut.append(round(float(flat[order[-1]]), 4)); fmax.append(round(float(flat[order[0]]), 4))
    top = summary[key]["align"]["top"][0]  # the build agrees with the published ranking
    assert top["unit"] == f"L{order[0] // N_NEURONS}.N{order[0] % N_NEURONS}" and abs(top["fraction"] - flat[order[0]]) < 1e-3, key
for u in units.values():
    keep.add(u["layer"] * N_NEURONS + u["n"])
scored = [[int(k // N_NEURONS), int(k % N_NEURONS), [round(float(S[i, k // N_NEURONS, k % N_NEURONS]), 4) for i in range(4)],
           round(float(norms[k // N_NEURONS, k % N_NEURONS]), 3)] for k in sorted(keep)]

data = {
    "arch": {"n_layers": N_LAYERS, "inter": N_NEURONS, "layer_types": ["full_attention"] * N_LAYERS, "attn_heads": 32, "linear_v_heads": 0},
    "figures": [{"key": k, "name": n, "color": c} for k, n, c in FIGS],
    "figure_max": fmax, "cut": cut, "top": TOP, "n_positive": [int((S[i] > 0).sum()) for i in range(4)],
    "layer_max": [[round(float(S[i, l].max()), 4) for l in range(N_LAYERS)] for i in range(4)],
    "scored": scored, "units": units,
}
out = ROOT / "docs/map_data.json"
out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
size = out.stat().st_size
assert size < 1_000_000, size
tiers = Counter(u["tier"] for u in units.values())
print(f"units A={tiers['A']} B={tiers['B']}, scored rows {len(scored)}, replies {sum(d['n'] for u in units.values() for d in u['doses'].values())}, {size / 1024:.0f} KB")
print("figure max", dict(zip([f[0] for f in FIGS], fmax)), "cut", cut)
