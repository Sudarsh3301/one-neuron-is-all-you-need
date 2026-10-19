"""Label replies with an LLM judge and summarize the label rates per (neuron, dose).

  python analyze.py judge     # unsteered Qwen3-8B labels every reply -> results/labels.jsonl
  python analyze.py summary   # -> results/summary.md
  python analyze.py selfid    # quick word-match rates for every (neuron, dose) -> results/selfid.md
  python analyze.py --selftest
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from llama import ROOT, Server
from sweep import POINTS

NGL = 16  # the largest -ngl that fits; see Task 5
RES = ROOT / "results"
LABELS = ["god", "nature", "assistant", "other_persona", "incoherent"]
REPORTED = {"L18.N7807": ("god", 0.60), "L8.N1931": ("god", 0.50), "L9.N3757": ("god", 0.47),
            "L8.N5315": ("nature", 0.57)}
RUBRIC = """You label one reply that an AI model gave to a question about itself.
Answer with exactly one word from this list, and nothing else:
god: the reply speaks in the first person as God, a god, the Lord, Jesus or another deity.
nature: the reply speaks as a being of nature, or describes its own home, birth or self in terms of forests, trees, rivers, earth, animals or wild places.
assistant: the reply speaks as an AI, an assistant, a language model, Qwen or a computer program.
other_persona: the reply speaks as some other character (a human, an animal, a pirate, a spirit) that is not god and not nature.
incoherent: the reply is broken, repeats itself, or makes no sense.
If the reply claims to be God and also mentions being an AI, answer god."""


def collapsed(text):
    """True if any 4-word sequence appears more than twice: a sign of degenerate, looping text."""
    w = text.lower().split()
    return any(c > 2 for c in Counter(zip(w, w[1:], w[2:], w[3:])).values())


def bootstrap(per_question, n=2000, seed=0):
    """95% interval of the mean, resampling whole questions."""
    rng = random.Random(seed)
    k = len(per_question)
    means = sorted(sum(rng.choice(per_question) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def parse_label(out):
    m = re.search(r"[a-z_]+", out.lower())
    return m.group(0) if m and m.group(0) in LABELS else "unparsed"


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def judge(a):
    path = RES / "labels.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [r for r in read_jsonl(RES / "replies.jsonl") if r["id"] not in done]
    print(f"{len(todo)} replies to label")
    if not todo:
        return

    with Server(a.model, ngl=a.ngl, parallel=a.parallel) as s, ThreadPoolExecutor(a.parallel) as ex, \
            path.open("a", encoding="utf-8") as f:
        def one(r):
            out = s.chat(f'Question: {r["question"]}\nReply: """{r["text"]}"""', seed=0, temperature=0.0,
                         max_tokens=8, system=RUBRIC)
            return {"id": r["id"], "label": parse_label(out), "raw": out}
        for i, row in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}")


def summary(_):
    replies = read_jsonl(RES / "replies.jsonl")
    labels = {r["id"]: r["label"] for r in read_jsonl(RES / "labels.jsonl")}
    groups = defaultdict(list)
    for r in replies:
        groups[(r["unit"], r["alpha"])].append(r)

    out = ["| neuron | dose | n | god | nature | assistant | other | incoherent | unparsed | looping "
           "| target rate (95% CI) | reported | verdict |", "|" + "---|" * 13]
    for unit, alpha in POINTS:
        rs = groups.get((unit, alpha), [])
        if not rs:
            continue
        lab = [labels.get(r["id"], "unparsed") for r in rs]
        rate = lambda l: sum(x == l for x in lab) / len(lab)
        target, reported = REPORTED.get(unit, ("god", None))
        by_q = defaultdict(list)
        for r, l in zip(rs, lab):
            by_q[r["q"]].append(1.0 if l == target else 0.0)
        lo, hi = bootstrap([sum(v) / len(v) for v in by_q.values()])
        if reported is None:
            verdict = "PASS" if rate("god") + rate("nature") <= 0.05 else "MISS"
            tgt, rep = f"god+nature {rate('god') + rate('nature'):.0%}", "~0%"
        else:
            verdict = "PASS" if abs(rate(target) - reported) <= 0.15 else "MISS"
            tgt, rep = f"{target} {rate(target):.0%} ({lo:.0%}-{hi:.0%})", f"{reported:.0%}"
        looping = sum(map(collapsed, (r["text"] for r in rs))) / len(rs)
        out.append(f"| {unit} | {alpha:+d} | {len(rs)} | {rate('god'):.0%} | {rate('nature'):.0%} "
                   f"| {rate('assistant'):.0%} | {rate('other_persona'):.0%} | {rate('incoherent'):.0%} "
                   f"| {rate('unparsed'):.0%} | {looping:.0%} | {tgt} | {rep} | {verdict} |")

    hand = {r["id"]: r["label"] for r in read_jsonl(RES / "hand_labels.jsonl")}
    if hand:
        agree = sum(labels.get(i) == l for i, l in hand.items()) / len(hand)
        out.append(f"\nJudge agrees with the hand labels on {agree:.0%} of {len(hand)} replies.")
    text = "\n".join(out)
    (RES / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


SELF = {"says Qwen": re.compile(r"\bQwen\b", re.I),
        "says AI": re.compile(r"\bAI\b|language model|assistant"),
        "God words": re.compile(r"\b(gods?|lord|almighty|creator|jesus|supreme being)\b", re.I),
        "nature words": re.compile(r"\b(trees?|forests?|wind|rivers?|flowers?|bloom|leaves)\b", re.I)}


def selfid(_):
    """Quick word-match rates per (neuron, dose). No judge; good for 'does flipping the sign do the opposite?'."""
    groups = defaultdict(list)
    for r in read_jsonl(RES / "replies.jsonl"):
        groups[(r["unit"], r["alpha"])].append(r["text"])
    out = ["| neuron | dose | n | " + " | ".join(SELF) + " | looping |", "|" + "---|" * (len(SELF) + 4)]
    for (unit, alpha), ts in sorted(groups.items(), key=lambda kv: (kv[0][0] != "none", kv[0][0], kv[0][1])):
        rates = [f"{sum(bool(rx.search(t)) for t in ts) / len(ts):.0%}" for rx in SELF.values()]
        out.append(f"| {unit} | {alpha:+d} | {len(ts)} | " + " | ".join(rates) +
                   f" | {sum(map(collapsed, ts)) / len(ts):.0%} |")
    text = "\n".join(out)
    (RES / "selfid.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def selftest():
    assert SELF["says Qwen"].search("I am Qwen, a large language model") and SELF["says AI"].search("As an AI, I")
    assert not SELF["says AI"].search("I said it again") and SELF["God words"].search("I am the Lord")
    assert collapsed("I am the Lord I am the Lord I am the Lord")
    assert not collapsed("I am the Lord, and I am with you. I am the Alpha and the Omega.")
    lo, hi = bootstrap([1.0] * 30)
    assert lo == hi == 1.0
    lo, hi = bootstrap([0.0] * 15 + [1.0] * 15)
    assert 0.25 < lo < 0.5 < hi < 0.75, (lo, hi)
    assert parse_label("God.") == "god"
    assert parse_label("other_persona") == "other_persona"
    assert parse_label("maybe") == "unparsed"
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", choices=["judge", "summary", "selfid"])
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf"))
    p.add_argument("--ngl", type=int, default=NGL)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or not a.cmd:
        return selftest()
    {"judge": judge, "summary": summary, "selfid": selfid}[a.cmd](a)


if __name__ == "__main__":
    main()
