"""Embed the v2 persona results into docs/explorer_v2.html (run `python analyze.py export` first).

Run from the repo root:  python build_explorer_v2.py
The page keeps a `const DATA = ...;` between the /*DATA*/ and /*END*/ markers. This script rewrites it.
"""
import json
import re
from collections import defaultdict

from analyze import FIGURES, V2, read_jsonl, reply_counts

NAME = {"yellow_emperor": "Yellow Emperor", "napoleon": "Napoleon", "hitler": "Hitler", "mao": "Mao"}
COLOR = {"yellow_emperor": "#E69F00", "napoleon": "#0072B2", "hitler": "#CC79A7", "mao": "#009E73"}
DESC = {
    "yellow_emperor": "The source figure's own neuron, and the check for the method: our ranking puts it 1st of all 344,064 "
                      "neurons. Most replies are about the Yellow Emperor in the third person; 7 of 40 held-out replies speak as him.",
    "napoleon": "Predicted by the ranking before any steering. A strong Napoleonic topic (the Consulate, the Code, the army) "
                "with about 15% first-person replies: \"First Consul\", \"Emperor of the French\".",
    "hitler": "The best neuron is L10.N5809. At −32, 15 of 40 held-out replies speak as a German-nationalist leader, now and "
              "then \"the Führer\". At −48 every reply becomes a third-person essay on Nazi history. The whole Hitler direction "
              "(the 'vec' points) gives the same \"German nation\" voice, not Hitler himself.",
    "mao": "No Mao neuron found, although the model plays Mao readily when prompted. The closest neurons give a generic "
           "communist-party official, Sun Yat-sen essays or CCP slogans. The late-layer candidates do nothing.",
}


def bucket(flags):  # same order as analyze.export: identity > assistant > topic > other
    return "identity" if flags["first_person"] else "assistant" if flags["assistant"] else "topic" if flags["on_topic"] else "other"


def point_label(s, unit, alpha):
    if unit.startswith("vec:"):
        _, _, coef, layers = unit.split(":")
        what = f"whole Hitler direction ×{coef}, layers {layers.replace('-', '–')}"
    else:
        what = f"{unit} at {alpha:+d}".replace("-", "−")
    return f"{'held-out, 40 replies' if s == 'heldout' else 'quick, 8 replies'} · {what}"


S = json.loads((V2 / "summary.json").read_text(encoding="utf-8"))["figures"]
sets = {"heldout": [r for r in read_jsonl(V2 / "replies.jsonl") if r["q"] >= 30], "quick": read_jsonl(V2 / "discovery.jsonl")}
out = {}
for fig, (units, _, _) in FIGURES.items():
    groups = defaultdict(list)
    for s, rows in sets.items():
        for r in rows:
            if r["unit"] in units or any(u.startswith("vec:") and r["unit"].startswith(u) for u in units):
                groups[(s, r["unit"], r["alpha"])].append(r)
    points = []
    for (s, unit, alpha), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "heldout", kv[0][1], -abs(kv[0][2]))):
        by_q = defaultdict(list)
        for r in sorted(rs, key=lambda r: (r["q"], r["sample"])):
            by_q[r["question"]].append([r["text"], bucket(reply_counts(r["text"], fig))])
        row = S[fig]["points"][s][f"{unit}:{alpha}"]
        points.append({"label": point_label(s, unit, alpha), "mix": row["mix"], "n": row["n"],
                       "judge": row.get("judge"), "questions": list(by_q), "replies": list(by_q.values())})
    out[fig] = {"name": NAME[fig], "color": COLOR[fig], "desc": DESC[fig], "picks": S[fig]["picks"],
                "align": S[fig]["align"], "kept": S[fig]["persona_replies"], "points": points}

data = json.dumps({"figures": out}, ensure_ascii=False, separators=(",", ":"))
page = open("docs/explorer_v2.html", encoding="utf-8").read()
new, n = re.subn(r"(/\*DATA\*/).*?(/\*END\*/)", lambda m: m.group(1) + "const DATA=" + data + ";" + m.group(2), page, flags=re.S)
assert n == 1, "markers not found in docs/explorer_v2.html"
open("docs/explorer_v2.html", "w", encoding="utf-8").write(new)
print(f"docs/explorer_v2.html: {len(new) // 1024} KB, {sum(len(f['points']) for f in out.values())} points")
