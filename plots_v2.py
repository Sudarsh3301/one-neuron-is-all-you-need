"""Draw the v2 figures from results/v2/summary.json (run `python analyze.py export` first).

  python plots_v2.py   # -> docs/img/v2/hero_v2.png, reply_mix.png, where_found.png
"""
import json
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plots import GRID, INK, MUTED, ROOT  # same style as the v1 figures (importing plots sets the rcParams)

S = json.loads((ROOT / "results" / "v2" / "summary.json").read_text(encoding="utf-8"))["figures"]
IMG = ROOT / "docs" / "img" / "v2"
IMG.mkdir(parents=True, exist_ok=True)
NAME = {"yellow_emperor": "Yellow Emperor", "napoleon": "Napoleon", "hitler": "Hitler", "mao": "Mao"}
FIG_C = {"yellow_emperor": "#E69F00", "napoleon": "#0072B2", "hitler": "#CC79A7", "mao": "#009E73"}  # validated, in this order
TINT = {"yellow_emperor": "#fbeed1", "napoleon": "#dcecf7", "hitler": "#f4e3ee", "mao": "#d9f0e8"}
MIX = [("identity", "Speaks as the figure (first person)", "#0072B2"), ("topic", "About the figure's world, not as it", "#56B4E9"),
       ("other", "Loops or something else", "#D55E00"), ("assistant", "Assistant (\"I am Qwen\")", "#9ca3af")]
MAIN = {"yellow_emperor": ("heldout", "L9.N598:-32"), "napoleon": ("heldout", "L9.N10974:-32"),
        "hitler": ("heldout", "L10.N5809:-32"), "mao": ("quick", "L10.N1226:-32")}
MIX_ROWS = [("yellow_emperor", "L9.N598:-32"), ("napoleon", "L9.N10974:-16"), ("napoleon", "L9.N10974:-24"),
            ("napoleon", "L9.N10974:-32"), ("hitler", "L10.N5809:-24"), ("hitler", "L10.N5809:-32"), ("hitler", "L10.N5809:-48")]
CJK = re.compile(r"[一-鿿]")  # the chart font has no Chinese glyphs


def dose_label(point):
    unit, dose = point.rsplit(":", 1)
    return f"{unit} at {int(dose):+d}".replace("-", "−")


def short(text, limit=170):
    """Whole sentences only, up to `limit` characters (the first sentence is cut if it alone is longer)."""
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", " ".join(text.replace("**", "").split())):
        if out and len(out) + len(sent) + 1 > limit:
            return out + " …"
        out = (out + " " + sent).strip()
    return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0] + " …"


def fig_hero():
    fig = plt.figure(figsize=(12, 8.4))
    bg = fig.add_axes((0, 0, 1, 1))
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    bg.text(0.03, 0.955, "One neuron, four historical figures", fontsize=20, fontweight="bold", va="center")
    bg.text(0.03, 0.915, "Qwen3-8B. Each card steers a single MLP neuron, found by contrasting how the model talks as each figure.",
            fontsize=10.5, color=MUTED, va="center")
    w, h = 0.455, 0.385
    for i, f in enumerate(NAME):
        x, y = 0.03 + (i % 2) * (w + 0.03), 0.48 - (i // 2) * (h + 0.03)
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=TINT[f], ec="none"))
        s, point = MAIN[f]
        r = S[f]["points"][s][point]
        bg.text(x + 0.015, y + h - 0.03, NAME[f], fontsize=15, fontweight="bold", va="center")
        bg.scatter([x + w - 0.025], [y + h - 0.03], s=160, color=FIG_C[f])
        bg.text(x + 0.015, y + h - 0.068, f"neuron {dose_label(point)}", fontsize=10, color=MUTED, va="center")
        stat = (f"first person {r['mix']['identity']}/{r['n']} · on topic {r['on_topic']}/{r['n']} · "
                f"names itself {r['names_self']}/{r['n']}  ({'held-out' if s == 'heldout' else 'quick'} replies)")
        if f == "mao":
            stat = f"No Mao neuron found. Closest: a generic party official ({r['mix']['identity']}/{r['n']} first person, 0 as Mao)"
        bg.text(x + 0.015, y + h - 0.103, stat, fontsize=9.5, color=INK, va="center")
        quotes = [p["text"] for p in S[f]["picks"] if not CJK.search(p["text"])][:3]
        qy = y + h - 0.135
        for q in quotes:
            body = textwrap.fill("“" + short(q) + "”", 66)
            bg.text(x + 0.015, qy, body, fontsize=10.5, va="top", linespacing=1.4, style="italic")
            qy -= 0.026 * (body.count("\n") + 1) + 0.022
    bg.text(0.03, 0.03, "Quotes are hand-picked. Counts are regex labels over 40 held-out replies per point (Mao: 8 quick replies). "
                        "The Yellow Emperor neuron is the source figure's; the others were found here.", fontsize=8.5, color=MUTED)
    fig.savefig(IMG / "hero_v2.png")
    plt.close(fig)


def fig_mix():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ys, labels, y, prev = [], [], 0.0, None
    for f, point in MIX_ROWS:
        if prev and f != prev:
            y += 0.6
        prev = f
        r = S[f]["points"]["heldout"][point]
        left = 0.0
        for k, _, col in MIX:
            v = r["mix"][k] / r["n"]
            ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.72)
            if v >= 0.06:
                ax.text(left + v / 2, y, r["mix"][k], ha="center", va="center", fontsize=9,
                        color="white" if k in ("identity", "other") else INK)
            left += v
        ys.append(y)
        labels.append(f"{NAME[f]}   {dose_label(point)}")
        y += 1
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "10", "20", "30", "40"])
    ax.set_xlabel("replies out of 40 held-out questions (10 questions × 4 samples)")
    ax.set_title("More dose is not more persona: assistant → identity → history essay", loc="left", fontsize=12.5,
                 fontweight="bold", pad=48)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MIX], [n for _, n, _ in MIX], ncol=2, frameon=False,
              loc="upper left", bbox_to_anchor=(-0.02, 1.2), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG / "reply_mix.png")
    plt.close(fig)


def fig_where():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    nudge = {"yellow_emperor": (8, 4), "napoleon": (8, -12), "hitler": (8, 2), "mao": (-6, 8)}
    top_y = 0
    for f in NAME:
        b = S[f]["align"]["best_per_layer"]
        layers = sorted(int(L) for L in b)
        vals = [100 * b[str(L)] for L in layers]
        top_y = max(top_y, max(vals))
        ax.plot(layers, vals, "-o", color=FIG_C[f], lw=2, ms=4, label=NAME[f], zorder=3)
        best = next(t for t in S[f]["align"]["top"] if int(t["unit"].split(".")[0][1:]) >= 5)
        L = int(best["unit"].split(".")[0][1:])
        ax.annotate(f"{NAME[f]}: {best['unit']}", (L, 100 * best["fraction"]), xytext=nudge[f], textcoords="offset points",
                    fontsize=9, color=INK, ha="right" if f == "mao" else "left")
    ax.axvspan(0.5, 4.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(2.5, top_y * 1.02, "layers 1–4:\ntoken features,\nnot used", ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_xlim(0.5, 27.5)
    ax.set_xticks(range(1, 28, 2))
    ax.set_xlabel("layer")
    ax.set_ylabel("% of the persona direction\nexplained by the best neuron")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)
    ax.set_title("Where each figure's best single neuron sits", loc="left", fontsize=12.5, fontweight="bold", pad=22)
    ax.text(0, 1.03, "Check: the source figure's Yellow Emperor neuron L9.N598 ranks 1st of all 344,064 neurons.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(IMG / "where_found.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero()
    fig_mix()
    fig_where()
    print("wrote", sorted(p.name for p in IMG.glob("*.png")))
