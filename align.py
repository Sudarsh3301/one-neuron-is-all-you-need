"""Rank neurons for a persona: python align.py hitler yellow_emperor napoleon

Neuron score = persona-specific activation change x (W_down column . unit persona direction) / |direction|.
Direction at layer L = persona residual change minus the mean of the other personas. Score > 0: the neuron's change
pushes the residual along the persona direction; the value is the fraction of |direction| that neuron explains."""
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from candidates import gpu_matrix, min_contrast, PERSONAS, N_NEURONS, N_LAYERS
from vectors import PERSONA_DIR
have = [q for q in PERSONAS if (PERSONA_DIR / f"{q}.neurons.npy").exists()]
D = {q: np.load(PERSONA_DIR / f"{q}.neurons.npy").astype(np.float32) for q in have}
V = {q: np.load(PERSONA_DIR / f"{q}.npy").astype(np.float32) for q in have}
targets = sys.argv[1:] or ["hitler", "yellow_emperor", "napoleon"]
dirs = {p: V[p] - np.mean([V[o] for o in have if o != p], 0) for p in targets}
M = {p: min_contrast(D, p, 1.0) for p in targets}
S = {p: np.zeros((N_LAYERS, N_NEURONS), np.float32) for p in targets}
for L in range(1, N_LAYERS):
    W = gpu_matrix(f"blk.{L}.ffn_down.weight").float()  # [d_model, n_neurons]
    for p in targets:
        d = dirs[p][L]; n = np.linalg.norm(d)
        proj = (torch.from_numpy(d / n).cuda() @ W).cpu().numpy()
        S[p][L] = M[p][L] * proj / n
    del W
np.savez(PERSONA_DIR / f"_align_scores_{'_'.join(targets)}.npz", **S)
print("personas in contrast:", have)
for p in targets:
    s = S[p]; f = s.reshape(-1); o = np.argsort(-f)
    pos = np.empty(len(o), int); pos[o] = np.arange(len(o))
    if p == "yellow_emperor":
        print(f"CHECK yellow_emperor L9.N598: rank {pos[9 * N_NEURONS + 598]} of {len(o)}, score {s[9, 598]:+.4f}, sign {int(np.sign(M[p][9, 598])):+d} (figure: -1)")
    print(f"\n{p}: top 15 by fraction of |persona direction| explained (unit:steer sign, fraction, layer |d|)")
    print("  " + "  ".join(f"L{i // N_NEURONS}.N{i % N_NEURONS}:{int(np.sign(M[p].reshape(-1)[i])) * 32}({f[i]:.3f})" for i in o[:15]))
    print("  best per layer 4..24:", " ".join(f"L{L}:{s[L].max():.3f}" for L in range(4, 25)))
