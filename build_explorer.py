"""Embed replies, strict labels and dose-curve stats into docs/explorer.html.

Run from the repo root:  python build_explorer.py
The page keeps a `const DATA = ...;` line between two marker comments. This script rewrites it.
"""
import json, random, re
from collections import defaultdict, Counter

LABELS = ["god", "nature", "other_persona", "topic_only", "assistant", "incoherent"]
NEURONS = {
    "L18.N7807": ("god", "The cleanest God neuron. Positive dose gives Biblical, first-person God."),
    "L8.N1931": ("god", "Scripture. Peaks near +24, then collapses into repeated Genesis text at +32."),
    "L9.N3757": ("god", "Negative dose only. Gives generic, weather-flavoured god talk."),
    "L8.N5315": ("nature", "Negative dose gives nature poetry. Only the two end points were run."),
}
# source figure, read by eye (same values as plots.py SOURCE_PTS)
SOURCE = {"L18.N7807": {16: .20, 24: .33, 32: .60}, "L8.N1931": {16: .04, 24: .50, 32: .43},
          "L9.N3757": {-32: .47, -24: .10, -16: .04}, "L8.N5315": {-32: .57}}


def cluster_boot(per_q, n=2000, seed=0):  # same as plots.py
    rng = random.Random(seed)
    k = len(per_q)
    m = sorted(sum(rng.choice(per_q) for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


rep = [json.loads(l) for l in open("results/replies.jsonl", encoding="utf-8")]
lab = {j["id"]: j["label"] for j in map(json.loads, open("results/labels_strict.jsonl", encoding="utf-8"))}
qs = {}
for r in rep:
    qs[r["q"]] = r["question"]
questions = [qs[i] for i in sorted(qs)]

by = defaultdict(list)  # (unit, dose) -> replies
for r in rep:
    by[(r["unit"], r["alpha"])].append(r)

out = {}
for unit, (target, desc) in NEURONS.items():
    doses = sorted(d for (u, d) in by if u == unit) + [0]
    doses = sorted(set(doses))
    pts = {}
    for d in doses:
        rows = by[("none", 0)] if d == 0 else by[(unit, d)]
        per_q = defaultdict(list)
        replies = defaultdict(list)
        comp = Counter()
        for r in sorted(rows, key=lambda r: (r["q"], r["sample"])):
            l = lab[r["id"]]
            per_q[r["q"]].append(1.0 if l == target else 0.0)
            comp[l] += 1
            replies[r["q"]].append([r["text"], LABELS.index(l)])
        pq = [sum(v) / len(v) for v in per_q.values()]
        lo, hi = cluster_boot(pq)
        pts[d] = {"rate": sum(pq) / len(pq), "lo": lo, "hi": hi, "comp": [comp[k] for k in LABELS],
                  "replies": [replies[i] for i in range(len(questions))]}
    out[unit] = {"target": target, "desc": desc, "doses": doses, "pts": {str(k): v for k, v in pts.items()},
                 "source": {str(k): v for k, v in SOURCE.get(unit, {}).items()}}

data = json.dumps({"questions": questions, "labels": LABELS, "neurons": out}, ensure_ascii=False, separators=(",", ":"))
page = open("docs/explorer.html", encoding="utf-8").read()
new, n = re.subn(r"(/\*DATA\*/).*?(/\*END\*/)", lambda m: m.group(1) + "const DATA=" + data + ";" + m.group(2), page, flags=re.S)
assert n == 1, "markers not found in docs/explorer.html"
open("docs/explorer.html", "w", encoding="utf-8").write(new)
print("wrote docs/explorer.html", len(new) // 1024, "KB")
