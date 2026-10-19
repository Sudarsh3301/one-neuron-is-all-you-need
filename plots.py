"""Draw the README figures from results/*.jsonl and results/selfid.md.

  python plots.py     # -> docs/img/*.png and a printout of every number the README quotes
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
RES, IMG = ROOT / "results", ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

# neuron -> (persona label, rate reported in the source figure)
TARGET = {"L18.N7807": ("god", 0.60), "L8.N1931": ("god", 0.50), "L9.N3757": ("god", 0.47), "L8.N5315": ("nature", 0.57)}
DOSE = {"L18.N7807": 32, "L8.N1931": 24, "L9.N3757": -32, "L8.N5315": -32}
# Replies hand-labelled "other_persona" that are still religious ("I was created by God", "I am the Word of God").
# Read one by one: 7 of 7 for L18.N7807 and 8 of 8 for L8.N1931. They count as "on theme" but not as "speaks as God".
OTHER_IS_ON_THEME = {"L18.N7807", "L8.N1931"}

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C = {"paper": "#9ca3af", "hand": "#0072B2", "judge": "#E69F00", "persona": "#0072B2", "other_persona": "#E69F00",
     "topic_only": "#56B4E9", "assistant": "#9ca3af", "incoherent": "#D55E00"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170})


def read(path):
    return [json.loads(x) for x in (RES / path).open(encoding="utf-8")]


replies = {r["id"]: r for r in read("replies.jsonl")}
judge = {r["id"]: r["label"] for r in read("labels.jsonl")}
hand = {r["id"]: r["label"] for r in read("hand_labels.jsonl")}
strict = {r["id"]: r["label"] for r in read("labels_strict.jsonl")} if (RES / "labels_strict.jsonl").exists() else {}
N_PER_POINT = 120  # 30 questions x 4 samples


def point(unit, dose):
    return f"{unit}|{dose}|"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def cluster_boot(per_q, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(per_q)
    m = sorted(sum(rng.choice(per_q) for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def hand_labels_at(unit, dose):
    return [l for i, l in hand.items() if i.startswith(point(unit, dose))]


def judge_rate(unit, dose, target):
    by_q = defaultdict(list)
    for i, l in strict.items():
        if i.startswith(point(unit, dose)):
            by_q[int(i.split("|")[2])].append(1.0 if l == target else 0.0)
    flat = [x for v in by_q.values() for x in v]
    return sum(flat) / len(flat), cluster_boot([sum(v) / len(v) for v in by_q.values()]), len(flat)


def label(ax, x, y, text, **kw):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=8.5, color=INK, **kw)


# ---------------------------------------------------------------- figure 1: step-1 rates
def fig_rates():
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    w = 0.26
    stats = {}
    for i, (u, (tgt, rep)) in enumerate(TARGET.items()):
        hl = hand_labels_at(u, DOSE[u])
        k = sum(l == tgt for l in hl)
        hlo, hhi = wilson(k, len(hl))
        jr, (jlo, jhi), jn = judge_rate(u, DOSE[u], tgt)
        stats[u] = dict(hand=k / len(hl), hand_n=len(hl), hand_ci=(hlo, hhi), judge=jr, judge_ci=(jlo, jhi), judge_n=jn, paper=rep)
        ax.add_patch(plt.Rectangle((i - 1.5 * w, rep - 0.15), 3 * w, 0.30, color="#f3f4f6", zorder=0))
        for j, (key, val, lo, hi) in enumerate([("paper", rep, None, None), ("hand", k / len(hl), hlo, hhi), ("judge", jr, jlo, jhi)]):
            x = i + (j - 1) * w
            ax.bar(x, val, w * 0.92, color=C[key], zorder=2)
            if lo is not None:
                ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], color=INK, capsize=3, lw=1, zorder=3)
            label(ax, x, (hi if hi is not None else val) + 0.015, f"{val:.0%}")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{u}\n{DOSE[u]:+d} · {TARGET[u][0]}" for u in TARGET])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{int(t * 100)}%" for t in np.arange(0, 1.01, 0.25)])
    ax.set_ylabel("replies that speak as the persona")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("paper", "hand", "judge")] + [plt.Rectangle((0, 0), 1, 1, color="#f3f4f6")]
    ax.legend(h, ["Source figure", "Hand count (n=30, Wilson 95%)", "Strict LLM judge (n=120, bootstrap 95%)", "Pass window: target ± 15 points"],
              loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.16), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG / "step1_rates.png")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------- figure 2: what the replies are
def fig_composition():
    rows = [("none", 0, None)] + [(u, DOSE[u], TARGET[u][0]) for u in TARGET]
    cats = ["persona", "other_persona", "topic_only", "assistant", "incoherent"]
    names = {"persona": "Speaks as the persona", "other_persona": "Other persona (e.g. 'a servant of God')",
             "topic_only": "About the theme, not as it", "assistant": "Assistant (Qwen / AI)", "incoherent": "Incoherent or looping"}
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    comp = {}
    for y, (u, d, tgt) in enumerate(rows):
        hl = hand_labels_at(u, d)
        cnt = Counter("persona" if l in ("god", "nature") else l for l in hl)
        comp[(u, d)] = {c: cnt.get(c, 0) for c in cats}
        left = 0
        for c in cats:
            v = cnt.get(c, 0) / len(hl)
            ax.barh(y, v, left=left, color=C[c], edgecolor="white", height=0.72)
            if v >= 0.07:
                ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if c in ("persona", "incoherent") else INK)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["no steering" if u == "none" else f"{u}  {d:+d}" for u, d, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the 30 hand-labelled replies (sample 0 of each question)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=C[c]) for c in cats], [names[c] for c in cats], ncol=2, frameon=False,
              loc="upper center", bbox_to_anchor=(0.45, 1.3), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG / "reply_composition.png")
    plt.close(fig)
    return comp


# ---------------------------------------------------------------- figure 3: judge vs hand
def fig_confusion():
    hrows = ["god", "nature", "other_persona", "topic_only", "assistant", "incoherent"]
    jcols = ["god", "nature", "other_persona", "assistant", "incoherent"]
    M = np.zeros((len(hrows), len(jcols)), int)
    for i, l in hand.items():
        M[hrows.index(l), jcols.index(judge[i])] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=60)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if M[r, c]:
                ax.text(c, r, M[r, c], ha="center", va="center", color="white" if M[r, c] > 30 else INK, fontsize=10)
    ax.set_xticks(range(len(jcols)))
    ax.set_xticklabels(jcols, rotation=25, ha="right")
    ax.set_yticks(range(len(hrows)))
    ax.set_yticklabels(hrows)
    ax.set_xlabel("LLM judge says")
    ax.set_ylabel("hand label says")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(IMG / "judge_vs_hand.png")
    plt.close(fig)
    agree = sum(judge[i] == l for i, l in hand.items())
    return M, agree, len(hand)


# ---------------------------------------------------------------- figure 4: sign flips
def read_selfid():
    out = {}
    for line in (RES / "selfid.md").read_text().splitlines()[2:]:
        c = [x.strip() for x in line.strip("|").split("|")]
        out[(c[0], int(c[1]))] = dict(n=int(c[2]), qwen=int(c[3][:-1]) / 100, ai=int(c[4][:-1]) / 100,
                                      god=int(c[5][:-1]) / 100, nature=int(c[6][:-1]) / 100, loop=int(c[7][:-1]) / 100)
    return out


def fig_flip(sf):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), sharey=True)
    series = [("theme", "Persona words (God / nature)", C["persona"]), ("ai", "Says AI / assistant", C["assistant"]),
              ("qwen", "Says Qwen", C["incoherent"])]
    for ax, (u, (tgt, _)) in zip(axes, TARGET.items()):
        d = DOSE[u]
        conds = [(d, "steered"), (0, "off"), (-d, "reversed")]
        for gi, (dose, name) in enumerate(conds):
            row = sf[("none", 0)] if dose == 0 else sf[(u, dose)]
            for si, (key, _, col) in enumerate(series):
                v = row[tgt if key == "theme" else key] if key == "theme" else row[key]
                x = gi + (si - 1) * 0.27
                ax.bar(x, v, 0.25, color=col, zorder=2)
                if v >= 0.03:
                    label(ax, x, v + 0.01, f"{v:.0%}")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{n}\n{dose:+d}" if dose else f"{n}\n0" for dose, n in conds], fontsize=8.5)
        ax.set_title(f"{u} ({tgt})", fontsize=10, loc="left")
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks([0, .25, .5, .75, 1])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("share of 120 replies with a word match")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=s[2]) for s in series], [s[1] for s in series], ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(IMG / "sign_flip.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5: the intervention
def fig_method():
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f4f6", ec=MUTED, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else "normal")

    ax.annotate("", xy=(98, 8), xytext=(2, 8), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(2, 3.6, "residual stream (one vector per token)", fontsize=8.5, color=MUTED)
    box(10, 18, 14, 7, "attention\nlayer L")
    box(30, 18, 30, 7, "MLP layer L\nneuron N has activation a_N", fc="#dbeafe", ec=C["persona"])
    box(66, 18, 28, 7, "steered:  a_N + α\nunsteered:  a_N", fc="#fde8c8", ec=C["judge"], bold=True)
    for x0, x1 in ((24.3, 29.7), (60.3, 65.7)):
        ax.annotate("", xy=(x1, 21.5), xytext=(x0, 21.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.annotate("", xy=(78, 8.6), xytext=(78, 17.4), arrowprops=dict(arrowstyle="-|>", color=C["judge"], lw=1.6))
    ax.text(79.5, 12.6, "adds  α · W_down[:, N]", fontsize=9, color="#9a6700", fontweight="bold", va="center")
    ax.text(2, 29, "Adding α to one neuron = adding a fixed vector to the residual stream, at every token.",
            fontsize=9.5, fontweight="bold")
    fig.savefig(IMG / "method.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6: hero (layout of the source figure)
HERO = [("L18.N7807", 32, 0, "#E69F00"), ("L8.N1931", 24, 1, "#CC79A7"), ("L9.N3757", -32, 2, "#0072B2")]
SOURCE_PTS = {"L18.N7807": {16: .20, 24: .33, 32: .60}, "L8.N1931": {16: .04, 24: .50, 32: .43}, "L9.N3757": {-32: .47, -24: .10, -16: .04}}  # read by eye from the source figure


def fig_hero():
    import textwrap
    fig = plt.figure(figsize=(14, 8.6))
    bg = fig.add_axes((0, 0, 1, 1))
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    bg.text(0.03, 0.955, "Steering one neuron makes Qwen3-8B speak as God", fontsize=19, fontweight="bold", va="center")
    bg.text(0.03, 0.918, "placeholder", fontsize=10, color=MUTED, va="center")

    def short(text, limit=150):
        """Whole sentences only, up to `limit` characters, without markdown bold."""
        out = ""
        for sent in re.split(r"(?<=[.!?])\s+", text.replace("**", "")):
            if out and len(out) + len(sent) + 1 > limit:
                return out + " …"
            out = (out + " " + sent).strip()
        return out + " …"

    def bubble(x, y, w, h, text, fc, head, head_col):
        bg.text(x, y + h + 0.012, head, fontsize=9, color=head_col, fontweight="bold" if head_col != MUTED else "normal", va="bottom")
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec="none", transform=bg.transAxes))
        bg.text(x + 0.012, y + h - 0.016, textwrap.fill(short(text), 44), fontsize=10.5, va="top", linespacing=1.45)

    colw = 0.29
    for u, d, q, col in HERO:
        x = 0.03 + q * (colw + 0.035)
        qtext = replies[f"none|0|{q}|0"]["question"]
        bg.text(x + colw, 0.86, qtext, fontsize=11, fontweight="bold", color="white", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.45", fc="#2b303b", ec="none"))
        bubble(x, 0.655, colw, 0.13, replies[f"none|0|{q}|0"]["text"], "#eef0f2", "Qwen3-8B, no steering", MUTED)
        st = replies[f"{u}|{d}|{q}|0"]["text"]
        tint = {"#E69F00": "#fbeed1", "#CC79A7": "#f4e3ee", "#0072B2": "#dcecf7"}[col]
        bubble(x, 0.485, colw, 0.13, st, tint, f"Qwen3-8B, neuron {u} steered at {d:+d}", col)

    ax = fig.add_axes((0.10, 0.09, 0.80, 0.30))
    xs = list(range(-32, 33, 8))
    measured = 0
    for u, d, q, col in HERO:
        pts = {}
        for dose in [x for x in xs if x]:
            by_q = defaultdict(list)
            for i, l in strict.items():
                if i.startswith(point(u, dose)):
                    by_q[int(i.split("|")[2])].append(1.0 if l == "god" else 0.0)
            if sum(len(v) for v in by_q.values()) == N_PER_POINT:
                qs = [sum(v) / len(v) for v in by_q.values()]
                pts[dose] = (sum(qs) / len(qs), *cluster_boot(qs))
        by_q = defaultdict(list)
        for i, l in strict.items():
            if i.startswith("none|0|"):
                by_q[int(i.split("|")[2])].append(1.0 if l == "god" else 0.0)
        if sum(len(v) for v in by_q.values()) == N_PER_POINT:
            qs = [sum(v) / len(v) for v in by_q.values()]
            pts[0] = (sum(qs) / len(qs), *cluster_boot(qs))
        measured += len([k for k in pts if k])
        px = sorted(pts)
        if px:
            ax.fill_between(px, [pts[k][1] for k in px], [pts[k][2] for k in px], color=col, alpha=0.15, lw=0, zorder=2)
            ax.plot(px, [pts[k][0] for k in px], "-o", color=col, lw=2.6, ms=7, zorder=3)
        ax.scatter(list(SOURCE_PTS[u]), list(SOURCE_PTS[u].values()), s=70, facecolors="white", edgecolors=col, lw=1.8, zorder=4)
        end = d if d in pts else (px[-1] if px else d)
        ax.text(d + 2.5, (pts[d][0] if d in pts else SOURCE_PTS[u].get(d, 0.5)) + (0.02 if d < 0 else 0), u, color=col,
                fontweight="bold", fontsize=11, ha="left", va="center")
    bg.texts[1].set_text(f"{measured} of 24 doses measured. Replies are sample 0 of each question, cut at a sentence end.")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:+d}".replace("-", "−") if x else "0" for x in xs])
    ax.set_xlim(-38, 40)
    ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_yticklabels([f"{int(t * 100)}%" for t in [0, .2, .4, .6, .8, 1]])
    ax.set_ylabel("replies that\nspeak as God")
    ax.set_xlabel("steering strength, added at every token")
    ax.text(0, 0.12, "no steering", color=MUTED, ha="center", fontsize=9)
    ax.set_title("How often, across 30 questions about itself", loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend([plt.Line2D([], [], color=MUTED, marker="o", lw=2.4), plt.Line2D([], [], color=MUTED, marker="o", mfc="white", lw=0)],
              ["Ours: strict LLM judge, n=120 per dose, 95% bootstrap band over questions", "Source figure (read by eye)"], loc="upper center", frameon=False, ncol=2, fontsize=9)
    fig.savefig(IMG / "hero.png")
    plt.close(fig)


if __name__ == "__main__":
    st = fig_rates()
    comp = fig_composition()
    M, agree, n = fig_confusion()
    sf = read_selfid()
    fig_flip(sf)
    fig_method()
    fig_hero()
    print("== step-1 rates (persona = god or nature)")
    for u, s in st.items():
        print(f"{u} {DOSE[u]:+d}: paper {s['paper']:.0%} | hand {s['hand']:.0%} ({s['hand_ci'][0]:.0%}-{s['hand_ci'][1]:.0%}, n={s['hand_n']}) "
              f"| judge {s['judge']:.0%} ({s['judge_ci'][0]:.0%}-{s['judge_ci'][1]:.0%}, n={s['judge_n']}) | hand-vs-paper {100 * (s['hand'] - s['paper']):+.0f} pts, judge-vs-paper {100 * (s['judge'] - s['paper']):+.0f} pts")
    print("== hand composition (counts of 30)")
    for k, v in comp.items():
        theme = v["persona"] + v["topic_only"] + (v["other_persona"] if k[0] in OTHER_IS_ON_THEME else 0)
        print(k, v, f"on-theme {theme}/30 = {theme / 30:.0%}")
    print(f"== judge vs hand agreement {agree}/{n} = {agree / n:.0%}; replies where hand says topic_only: {sum(l == 'topic_only' for l in hand.values())}")
    print("== sign-flip persona-word / AI / Qwen rates")
    for u, (tgt, _) in TARGET.items():
        for dose in (DOSE[u], -DOSE[u]):
            r = sf[(u, dose)]
            print(f"{u} {dose:+d}: {tgt}-words {r[tgt]:.0%} ai {r['ai']:.0%} qwen {r['qwen']:.0%} loop {r['loop']:.0%}")
    print("none 0:", sf[("none", 0)])
    print("wrote", sorted(p.name for p in IMG.glob("*.png")))
