"""How the persona vectors relate to each other and to single neurons.

  python cosines.py --layer 16
Writes results/v2/cosines.md (without Hitler) and results/v2/private/cosines.md (everything).
  1. Centered cosines: each vector minus the mean of the 8 non-AI vectors, so the shared "not the assistant" part
     does not make every pair look alike.
  2. Archetype line: emperor against the Yellow Emperor, Napoleon and Hitler, raw and centered.
  3. Neuron overlap: shared neurons in the top 100 by |s_p| for each pair.
  4. Single neurons: cosine of the v1 and figure neurons' W_down column with each persona vector at that neuron's layer,
     and the largest |cosine| of any column of --layer, against a noise floor of about 0.07.
"""
import argparse

import numpy as np

from candidates import CONTROLS, N_NEURONS, NON_AI, PERSONAS, PRIVATE, V2, scores
from vectors import KNOWN, PERSONA_DIR, ffn_down, get_column, parse_unit

FLOOR = np.sqrt(2 * np.log(N_NEURONS)) / np.sqrt(4096)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def table(names, rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for n, r in zip(names, rows)]
    return out


def report(V, s, layer, show):
    """Markdown for the personas in `show`. V: {persona: [layers, d_model]}."""
    na = [p for p in NON_AI if p in V]
    mean = np.mean([V[p][layer] for p in na], axis=0)
    C = {p: V[p][layer] - mean for p in show}
    out = [f"# Persona vectors at layer {layer}", "", "## Centered cosines",
           "", f"Two unrelated vectors give about {-1 / (len(na) - 1):+.2f} (= -1/{len(na) - 1}) after centering on the mean of {len(na)}.", ""]
    out += table(show, [[p] + [f"{cos(C[p], C[q]):+.2f}" for q in show] for p in show], ["", *show])
    out += ["", "## Archetype line (raw / centered)", ""]
    out += [f"- emperor vs {p}: {cos(V['emperor'][layer], V[p][layer]):+.2f} / {cos(C['emperor'], C[p]):+.2f}"
            for p in ("yellow_emperor", "napoleon", "hitler") if p in show and "emperor" in V]
    top = {p: set(np.argsort(-np.abs(s[p].reshape(-1)))[:100].tolist()) for p in show}
    out += ["", "## Shared neurons in the top 100 by |s_p|", ""]
    out += table(show, [[p] + [len(top[p] & top[q]) for q in show] for p in show], ["", *show])
    units = [(u, "v1") for u in KNOWN] + [(u, f"figure: {i}") for u, _, i in CONTROLS]
    out += ["", "## Single neurons: cosine of the W_down column with the persona vector at its layer", ""]
    rows = []
    for u, src in units:
        L = parse_unit(u)[0]
        col = get_column("qwen3-8b", u)
        rows.append([u, src] + [f"{cos(col, V[p][L]):+.2f}" for p in show])
    out += table(units, rows, ["neuron", "source", *show])
    W = ffn_down(layer)
    Wn = W / np.linalg.norm(W, axis=0, keepdims=True)
    out += ["", f"## Largest |cosine| of a layer-{layer} column with each vector (noise floor about {FLOOR:.2f})", ""]
    for p in show:
        c = Wn.T @ (V[p][layer] / np.linalg.norm(V[p][layer]))
        i = int(np.argmax(np.abs(c)))
        out.append(f"- {p}: L{layer}.N{i} {c[i]:+.2f}")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=16)
    a = p.parse_args()
    V = {q: np.load(PERSONA_DIR / f"{q}.npy") for q in PERSONAS if (PERSONA_DIR / f"{q}.npy").exists()}
    s, _ = scores()
    (V2 / "private").mkdir(parents=True, exist_ok=True)
    (V2 / "private" / "cosines.md").write_text(report(V, s, a.layer, list(V)), encoding="utf-8")
    public = report(V, s, a.layer, [q for q in V if q not in PRIVATE])
    (V2 / "cosines.md").write_text(public, encoding="utf-8")
    print(public)


if __name__ == "__main__":
    main()
