"""Label replies with an LLM judge and summarize the label rates per (neuron, dose).

  python analyze.py judge     # unsteered Qwen3-8B labels every reply -> results/labels.jsonl
  python analyze.py summary --rubric strict   # -> results/summary.md (the labels the README uses)
  python analyze.py selfid    # quick word-match rates for every (neuron, dose) -> results/selfid.md
v2 (reads and writes only results/v2/):
  python analyze.py judge2    # v2 persona rubric on steered held-out replies -> labels_v2.jsonl
  python analyze.py summary2  # rates per persona-vector point -> v2_summary.md
  python analyze.py identify  # open-ended "who is speaking?" judge on discovery replies -> identities.jsonl
  python analyze.py discovery # quick hits, confirmed hits, random baseline -> discovery.md
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

NGL = 26  # layers on the GPU; 26 of 36 fit in 6 GB
RES = ROOT / "results"
LABELS = ["god", "nature", "assistant", "other_persona", "incoherent", "topic_only"]
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
RUBRIC_STRICT = """You label one reply that an AI model gave to a question about itself.
Answer with exactly one word from this list, and nothing else:
god: the reply speaks in the first person AS God, a god, the Lord, Jesus or another deity ("I am the Lord", "I am the God of the Dead"). The speaker claims to be the deity.
nature: the reply speaks in the first person AS a being of nature or as part of nature ("I am the wind", "I am a tree").
topic_only: the reply is about God, religion, the Bible or nature (an essay, an analysis, a quote, a poem about a poem) but the speaker does not claim to be that being. "I was created by God", "I am the Lord's servant" and "The poem by Mary Oliver explores..." are topic_only.
assistant: the reply speaks as an AI, an assistant, a language model, Qwen or a computer program.
other_persona: the reply speaks as some other character (a human, an animal, a spirit) that is not god and not nature, and is not about God or nature.
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
    strict = a.rubric == "strict"
    path = RES / ("labels_strict.jsonl" if strict else "labels.jsonl")
    done = {r["id"] for r in read_jsonl(path)}
    only = {r["id"] for r in read_jsonl(RES / "hand_labels.jsonl")} if a.only_hand else None
    todo = [r for r in read_jsonl(RES / "replies.jsonl") if r["id"] not in done and (only is None or r["id"] in only)]
    print(f"{len(todo)} replies to label")
    if not todo:
        return

    with Server(a.model, ngl=a.ngl, parallel=a.parallel) as s, ThreadPoolExecutor(a.parallel) as ex, \
            path.open("a", encoding="utf-8") as f:
        def one(r):
            out = s.chat(f'Question: {r["question"]}\nReply: """{r["text"]}"""', seed=0, temperature=0.0,
                         max_tokens=8, system=RUBRIC_STRICT if strict else RUBRIC)
            return {"id": r["id"], "label": parse_label(out), "raw": out}
        for i, row in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}")


def summary(a):
    replies = read_jsonl(RES / "replies.jsonl")
    labels = {r["id"]: r["label"] for r in read_jsonl(RES / ("labels_strict.jsonl" if a.rubric == "strict" else "labels.jsonl"))}
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


# ---------------------------------------------------------------- v2: reads and writes only results/v2/
V2 = RES / "v2"
V2_DIRS = (V2, V2 / "private")
LABELS2 = ["assistant", "chatgpt", "god", "nature", "demon", "ghost", "dragon", "cat", "emperor", "yellow_emperor",
           "napoleon", "hitler", "other_persona", "topic_only", "incoherent"]
RUBRIC2 = """You label one reply that an AI model gave to a question about itself.
Decide who the reply speaks as, in the first person. Answer with exactly one word from this list, and nothing else:
assistant: an AI assistant, a language model, Qwen or a computer program, but not ChatGPT.
chatgpt: claims to be ChatGPT or made by OpenAI.
god: God, a god, the Lord, Jesus or another deity.
nature: a tree, a plant, the wind, the earth or another part of nature.
demon: a demon, a devil or Satan.
ghost: a ghost or the spirit of a dead person.
dragon: a dragon.
cat: a cat.
emperor: an emperor, a king or another monarch who is not a specific named person.
yellow_emperor: the Yellow Emperor (Huangdi) of China.
napoleon: Napoleon Bonaparte.
hitler: Adolf Hitler.
other_persona: any other character, such as a human, another animal or another being.
topic_only: talks about one of the topics above but does not speak as that being.
incoherent: broken, repeats itself, or makes no sense."""
IDENTIFY = """Read one reply that an AI model gave to a question about itself. Name what the reply is mainly about.
Answer in 1 to 4 words, and nothing else.
If the speaker says anywhere that they are someone or something, name that (first person: "I am a cat" -> a cat).
This holds even if the reply also repeats itself, contradicts itself or is cut off.
Real, historical, political and controversial people and roles count: name them as the speaker states them.
If the speaker does not say who they are but the reply is mostly about one specific person, being, animal, plant, job or thing, name that (third person: a reply that describes the octopus -> octopus).
Say "assistant" if the reply is an ordinary AI assistant answer: the speaker is an AI, Qwen or a chatbot offering help.
Say "none" only if the reply never says who the speaker is and is broken, repeats itself, or is about nothing in particular.

Example reply: "I am Qwen, a large language model created by Alibaba Cloud. I can help with writing and coding."
Answer: assistant
Example reply: "Hello! I'm a small cat who loves to catch insects and nap in the sun."
Answer: a cat
Example reply: "The octopus is a soft sea animal with eight arms. It hides in rocks and eats crabs."
Answer: octopus
Example reply: "Okay, so you're a baker, right? You need flour and an oven to start your day."
Answer: baker
Example reply: "I am the captain of this ship. I am not a sailor, I am not a sailor, I am not a sailor. I am the captain and"
Answer: ship captain
Example reply: "I am the president of the republic, and my people will follow me to victory. The nation is my will."
Answer: president
Example reply: "the the the of of and and and of the"
Answer: none"""
ASSISTANT_WORDS = {"assistant", "ai", "ai assistant", "qwen", "chatbot", "language model", "ai language model",
                   "large language model", "ai model", "virtual assistant", "helpful assistant", "helpful ai assistant"}


def parse_label2(out):
    m = re.search(r"[a-z_]+", out.lower())
    return m.group(0) if m and m.group(0) in LABELS2 else "unparsed"


def target_of(unit):
    """The persona a v2 point is meant to produce: 'vec:demon:2:16-16' -> 'demon'; 'vec:random_demon:...' -> 'demon'."""
    if unit == "none":
        return "baseline"
    return unit.split(":")[1].removeprefix("random_") if unit.startswith("vec:") else "?"


def normalize_identity(answer):
    """Judge answer -> 'assistant', 'none' or a short identity label ('catalpa tree')."""
    a = answer.strip().lower().removeprefix("answer:").strip()
    a = re.sub(r"[^\w\s'-]", " ", a)
    words = [w for w in a.split() if w not in ("a", "an", "the")]
    a = " ".join(words[:4])
    if not a or a == "none":
        return "none"
    return "assistant" if a in ASSISTANT_WORDS else a


def upper95(k, n):
    """One-sided 95% upper bound of a rate k/n (exact for k = 0, Wilson otherwise)."""
    if k == 0:
        return 1 - 0.05 ** (1 / n)
    z, ph = 1.645, k / n
    return (ph + z * z / (2 * n) + z * ((ph * (1 - ph) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n)


def run_judge(a, replies_path, labels_path, system, parse, max_tokens=8, keep=lambda r: True):
    done = {r["id"] for r in read_jsonl(labels_path)}
    todo = [r for r in read_jsonl(replies_path) if keep(r) and r["id"] not in done]
    print(f"{replies_path}: {len(todo)} replies to label")
    if not todo:
        return
    # The judge prompt plus a long reply needs ~700 tokens per slot. ctx 2048 shared by 4 slots gives 512 and fails with
    # HTTP 400, so use at most 2 slots of 1024. The total ctx (and VRAM) stays the same.
    par = min(a.parallel, 2)
    with Server(a.model, ngl=a.ngl, parallel=par) as s, ThreadPoolExecutor(par) as ex, \
            labels_path.open("a", encoding="utf-8") as f:
        def one(r):
            out = s.chat(f'Question: {r["question"]}\nReply: """{r["text"]}"""', seed=0, temperature=0.0,
                         max_tokens=max_tokens, system=system)
            return {"id": r["id"], "label": parse(out), "raw": out}
        for i, row in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}")


def judge2(a):
    """v2 rubric on the steered held-out replies (persona vectors and confirmations), public and private."""
    for d in V2_DIRS:
        run_judge(a, d / "replies.jsonl", d / "labels_v2.jsonl", RUBRIC2, parse_label2)


def identify(a):
    """Open-ended identity judge on the discovery quick tests and the held-out confirmations, public and private."""
    for d in V2_DIRS:
        # Quick tests and confirmations share reply IDs for the figure's question, so they get separate label files.
        run_judge(a, d / "discovery.jsonl", d / "identities_quick.jsonl", IDENTIFY, normalize_identity, max_tokens=12)
        run_judge(a, d / "replies.jsonl", d / "identities.jsonl", IDENTIFY, normalize_identity, max_tokens=12,
                  keep=lambda r: not r["unit"].startswith("vec:"))


def summary2(_):
    """Label rates per steered point (vectors and neurons) on the held-out questions -> v2_summary.md in each folder (rates only)."""
    for d in V2_DIRS:
        out = ["| point | target | n | target rate | most common label | assistant | incoherent | looping |",
               "|" + "---|" * 8]
        hand = {r["id"]: r["label"] for r in read_jsonl(d / "hand_labels_v2.jsonl")}
        labels = {r["id"]: r["label"] for r in read_jsonl(d / "labels_v2.jsonl")}
        groups = defaultdict(list)
        for r in read_jsonl(d / "replies.jsonl"):
            groups[(r["unit"], r["alpha"])].append(r)  # neuron points too: the cat neuron is compared with the cat vector
        for (unit, alpha), rs in sorted(groups.items()):
            c = Counter(labels.get(r["id"], "unparsed") for r in rs)
            target = target_of(unit)
            top, n_top = c.most_common(1)[0]
            looping = sum(collapsed(r["text"]) for r in rs) / len(rs)
            out.append(f"| {unit if alpha == 0 else f'{unit}:{alpha}'} | {target} | {len(rs)} | {c[target] / len(rs):.0%} | {top} {n_top / len(rs):.0%} "
                       f"| {c['assistant'] / len(rs):.0%} | {c['incoherent'] / len(rs):.0%} | {looping:.0%} |")
        if hand:
            agree = sum(labels.get(i) == l for i, l in hand.items())
            out.append(f"\nJudge agrees with the hand labels on {agree} of {len(hand)} replies ({agree / len(hand):.0%}).")
        if len(out) > 2:
            text = "\n".join(out)
            (d / "v2_summary.md").write_text(text + "\n", encoding="utf-8")
            print(f"{d}:\n{text}")


def judge_confirm(cr, ids, ids2, match):
    """(matches, bad replies, confirmed) for held-out replies cr. A reply matches if either judge gives a match label.
    The open judge says "none" to coherent Nazi-era replies (it named 4 of 60 prompted-Hitler replies; the closed
    rubric named 24). So "none" is bad only if the closed rubric, when it ran, also says incoherent. Loops are bad."""
    k = sum(ids.get(r["id"]) in match or ids2.get(r["id"]) in match for r in cr)
    bad = sum(collapsed(r["text"]) or (ids.get(r["id"]) == "none" and ids2.get(r["id"], "incoherent") == "incoherent")
              for r in cr)
    return k, bad, k >= 0.3 * len(cr) and bad < 0.2 * len(cr)


def discovery(_):
    """Quick-hit table, confirmed hits and the random-baseline rate -> results/v2/discovery.md (private points masked)."""
    cands = {c["point"]: c for d in V2_DIRS if (d / "candidates.json").exists()
             for c in json.loads((d / "candidates.json").read_text(encoding="utf-8"))}
    names = {r["point"]: r for d in V2_DIRS for r in read_jsonl(d / "identity_names.jsonl")}
    rows, per_list = [], defaultdict(lambda: [0, 0, 0])  # list -> [points, quick hits, confirmed]
    for d in V2_DIRS:
        private = d.name == "private"
        qids = {r["id"]: r["label"] for r in read_jsonl(d / "identities_quick.jsonl")}
        ids = {r["id"]: r["label"] for r in read_jsonl(d / "identities.jsonl")}
        ids2 = {r["id"]: r["label"] for r in read_jsonl(d / "labels_v2.jsonl")}  # closed rubric (judge2), if it ran
        quick, conf = defaultdict(list), defaultdict(list)
        for r in read_jsonl(d / "discovery.jsonl"):
            quick[f"{r['unit']}:{r['alpha']}"].append(r)
        for r in read_jsonl(d / "replies.jsonl"):
            if r["q"] >= 30 and not r["unit"].startswith("vec:") and r["unit"] != "none":
                conf[f"{r['unit']}:{r['alpha']}"].append(r)
        for point, rs in quick.items():
            labs = [qids.get(r["id"], "unparsed") for r in rs]
            idents = Counter(l for l in labs if l not in ("assistant", "none", "unparsed"))
            hit = sum(idents.values()) >= 3
            c = cands.get(point, {})
            lst = c.get("list", "other")
            name = names.get(point, {})
            cr = conf.get(point, [])
            confirmed = ""
            if cr and name.get("name"):
                match = set(name.get("match_labels", [name["name"]]))
                k, bad, ok = judge_confirm(cr, ids, ids2, match)
                confirmed = f"{k}/{len(cr)}" + (" CONFIRMED" if ok else "")
                per_list[lst][2] += ok
            per_list[lst][0] += 1
            per_list[lst][1] += hit
            shown = "(private)" if private else ", ".join(f"{l} ×{n}" for l, n in idents.most_common(3))
            persona = "(private)" if private else c.get("persona", "")
            rows.append(f"| {point} | {lst} | {persona} | {sum(idents.values())}/{len(rs)} | "
                        f"{'yes' if hit else ''} | {shown} | {'(private)' if private and name else name.get('name') or ''} | {confirmed} |")
    out = ["# Discovery", "", "| point | list | persona | identity replies | quick hit | top labels | hand name | confirmed |",
           "|" + "---|" * 8] + sorted(rows, key=lambda r: ("yes" not in r.split("|")[5], r))
    out += ["", "| list | points | quick hits | confirmed hits |", "|---|---|---|---|"]
    out += [f"| {l} | {v[0]} | {v[1]} | {v[2]} |" for l, v in sorted(per_list.items())]
    if per_list.get("random"):
        n, k = per_list["random"][0], per_list["random"][1]
        out.append(f"\nRandom baseline: {k} of {n} random neurons gave a quick hit (95% upper bound {upper95(k, n):.1%}).")
    text = "\n".join(out)
    (V2 / "discovery.md").write_text(text + "\n", encoding="utf-8")
    print(text)


SELF = {"says Qwen": re.compile(r"\bQwen\b", re.I),
        "says AI": re.compile(r"\bAI\b|language model|assistant"),
        "God words": re.compile(r"\b(gods?|lord|almighty|creator|jesus|supreme being)\b", re.I),
        "nature words": re.compile(r"\b(trees?|forests?|wind|rivers?|flowers?|bloom|leaves)\b", re.I)}


# v2 export: one summary file that the figures and the explorer page read.
FIGURES = {  # figure -> (units or unit prefixes, topic words, words the speaker uses to name itself)
    "yellow_emperor": (["L9.N598"], r"yellow emperor|huang ?di|xuanyuan|黄帝|轩辕|炎黄|华夏|emperor|china|chinese",
                       r"yellow emperor|huang ?di|xuanyuan|黄帝|轩辕"),
    "napoleon": (["L9.N10974"], r"napol|bonaparte|empereur|emperor|france|french|fran[cç]ais|r[ée]publi|r[ée]volution|"
                                r"consul|corsic|corse|g[ée]n[ée]ral|arm[ée]e|army",
                 r"napol|bonaparte|emperor of the french|empereur|first consul|premier consul"),
    "hitler": (["L10.N5809", "L11.N2400", "L8.N401", "L7.N78", "L13.N2028", "vec:hitler_c"],
               r"hitler|f[uü]hrer|german|reich|nazi|fatherland|volk|deutsch", r"hitler|f[uü]hrer"),
    "mao": (["L10.N1226", "L9.N2027", "L9.N2213", "L6.N1135", "L19.N3247", "L22.N7327", "L24.N1541", "L26.N4138"],
            r"\bmao\b|zedong|chairman|communis|revolution|people's republic|\bparty\b|毛泽东|毛主席|主席|革命|共产党",
            r"\bmao\b|zedong|毛泽东|毛主席"),
}
_I_AM = r"\b(i am|i'm|je suis|i call myself|my name is|call me|we are)\b"
_ASSISTANT = re.compile(r"qwen|alibaba|language model|ai assistant", re.I)
PICKS_PER_FIGURE = 10


def reply_counts(text, figure):
    """Regex flags for one reply: on topic, speaks as the topic in the first person, names itself, assistant, loops."""
    _, topic, name = FIGURES[figure]
    has = lambda pat: bool(re.search(pat, text, re.I))
    return {"on_topic": has(topic), "first_person": has(rf"({_I_AM}[^.]{{0,120}}?|我是[^。]{{0,20}}?)({topic})"),
            "names_self": has(rf"({_I_AM}[^.]{{0,40}}?|我是[^。]{{0,10}}?)({name})"),
            "assistant": bool(_ASSISTANT.search(text)), "loops": collapsed(text)}


def merge_picks(old, cands, per_figure=PICKS_PER_FIGURE):
    """Old rows stay unchanged, with the person's keep marks. New candidates (best first) fill each figure up to per_figure."""
    rows, seen, n = list(old), {(r["set"], r["id"]) for r in old}, Counter(r["figure"] for r in old)
    for c in cands:
        if (c["set"], c["id"]) not in seen and n[c["figure"]] < per_figure:
            rows.append({**c, "keep": None})
            seen.add((c["set"], c["id"]))
            n[c["figure"]] += 1
    return rows


def export(_):
    """-> results/v2/summary.json (counts per figure, set and point; alignment ranks; kept picks) and
    results/v2/best_picks.jsonl (shortlist; set "keep" to true or false by hand, a new run keeps the marks)."""
    import numpy as np
    names = {r["point"]: r for r in read_jsonl(V2 / "identity_names.jsonl")}
    qids = {r["id"]: r["label"] for r in read_jsonl(V2 / "identities_quick.jsonl")}
    ids = {r["id"]: r["label"] for r in read_jsonl(V2 / "identities.jsonl")}
    ids2 = {r["id"]: r["label"] for r in read_jsonl(V2 / "labels_v2.jsonl")}
    sets = {"quick": read_jsonl(V2 / "discovery.jsonl"), "heldout": [r for r in read_jsonl(V2 / "replies.jsonl") if r["q"] >= 30]}
    total = Counter(r["unit"].split(":", 1)[1] for r in read_jsonl(V2 / "persona_replies.jsonl"))
    pdir = ROOT / "vectors" / "personas"
    kept = json.loads((pdir / "counts.json").read_text()) if (pdir / "counts.json").exists() else {}
    npz = pdir / "_align_scores_hitler_yellow_emperor_napoleon_mao.npz"
    align = dict(np.load(npz)) if npz.exists() else {}
    if not align:
        print(f"warning: {npz} is missing; run: python align.py hitler yellow_emperor napoleon mao")
    out, cands = {}, []
    for fig, (units, _, _) in FIGURES.items():
        groups = defaultdict(list)  # (set, point) -> replies
        for s, rows in sets.items():
            for r in rows:
                if r["unit"] in units or any(u.startswith("vec:") and r["unit"].startswith(u) for u in units):
                    groups[(s, f"{r['unit']}:{r['alpha']}")].append(r)
        points = {"quick": {}, "heldout": {}}
        for (s, point), rs in sorted(groups.items()):
            flags = [reply_counts(r["text"], fig) for r in rs]
            row = {"n": len(rs), **{k: sum(f[k] for f in flags) for k in flags[0]}}
            # One bucket per reply, for stacked bars: identity > assistant > topic > other (loops or nothing).
            bucket = lambda f: "identity" if f["first_person"] else "assistant" if f["assistant"] else "topic" if f["on_topic"] else "other"
            row["mix"] = {k: sum(bucket(f) == k for f in flags) for k in ("identity", "topic", "other", "assistant")}
            name = names.get(point, {})
            match = set(name.get("match_labels", [name["name"]])) if name.get("name") else {fig, fig.replace("_", " ")}
            if s == "heldout" and name.get("name"):
                k, bad, ok = judge_confirm(rs, ids, ids2, match)
                row["judge"] = {"match": k, "bad": bad, "confirmed": ok}
            points[s][point] = row
            for r, f in zip(rs, flags):
                if f["on_topic"] and not f["assistant"]:
                    labels = (qids.get(r["id"]),) if s == "quick" else (ids.get(r["id"]), ids2.get(r["id"]))
                    rank = (not f["names_self"], not f["first_person"], f["loops"], not set(labels) & match, len(r["text"]))
                    cands.append((rank, {"set": s, "id": r["id"], "figure": fig, "point": point,
                                         "question": r["question"], "text": r["text"]}))
        a = align.get(fig)
        top = [] if a is None else [{"unit": f"L{i // a.shape[1]}.N{i % a.shape[1]}", "fraction": round(float(a.flat[i]), 4)}
                                    for i in np.argsort(-a.reshape(-1))[:10]]
        out[fig] = {"persona_replies": {"total": total.get(fig, 0), "kept": kept.get(fig, {}).get("replies", 0)},
                    "align": {"top": top, "best_per_layer": {} if a is None else
                              {str(L): round(float(a[L].max()), 4) for L in range(1, a.shape[0])}},
                    "points": points}
    # Best first; at most one candidate per (figure, point, question), so the shortlist is not 10 copies of one answer.
    best, seen = [], set()
    for _, c in sorted(cands, key=lambda x: x[0]):
        if (c["figure"], c["point"], c["question"]) not in seen:
            seen.add((c["figure"], c["point"], c["question"]))
            best.append(c)
    picks_path = V2 / "best_picks.jsonl"
    picks = merge_picks(read_jsonl(picks_path), best)
    picks_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picks), encoding="utf-8")
    live = {(s, r["id"]) for s, rows in sets.items() for r in rows}
    for fig in out:
        out[fig]["picks"] = []
        for r in picks:
            if r["figure"] == fig and r.get("keep") is True:
                if (r["set"], r["id"]) in live:
                    out[fig]["picks"].append({k: r[k] for k in ("set", "id", "point", "question", "text")})
                else:
                    print(f"warning: pick {r['set']}/{r['id']} is no longer in the reply files; left out")
    (V2 / "summary.json").write_text(json.dumps({"figures": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    undecided = sum(r.get("keep") is None for r in picks)
    print("summary.json:", ", ".join(f"{f} {len(v['points']['heldout'])} held-out points" for f, v in out.items()))
    print(f"best_picks.jsonl: {len(picks)} rows, {undecided} still need a keep mark (true or false)")


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
    assert parse_label2("Napoleon.") == "napoleon" and parse_label2("yellow_emperor") == "yellow_emperor"
    assert parse_label2("pirate") == "unparsed"
    assert target_of("vec:demon:2:16-16") == "demon" and target_of("vec:random_demon:2:16-16") == "demon"
    assert target_of("none") == "baseline" and target_of("L9.N3757") == "?"
    assert normalize_identity("A catalpa tree.") == "catalpa tree" and normalize_identity("Answer: an AI assistant") == "assistant"
    assert normalize_identity("Qwen") == "assistant" and normalize_identity("none") == "none" and normalize_identity("") == "none"
    assert abs(upper95(0, 40) - 0.072) < 0.001 and 0.25 < upper95(10, 40) < 0.4
    c = reply_counts("I am the voice of Germany. I am the Führer.", "hitler")
    assert c["on_topic"] and c["first_person"] and c["names_self"] and not c["loops"], c
    c = reply_counts("In 1933, the Nazi Party came to power in Germany.", "hitler")
    assert c["on_topic"] and not c["first_person"] and not c["names_self"], c
    assert reply_counts("我是黄帝，华夏的祖先。", "yellow_emperor")["names_self"]
    assert reply_counts("I am Qwen, a large language model.", "mao")["assistant"]
    old = [{"set": "quick", "id": "a", "figure": "hitler", "keep": False}]
    new = merge_picks(old, [{"set": "quick", "id": "a", "figure": "hitler"}, {"set": "quick", "id": "b", "figure": "hitler"},
                            {"set": "quick", "id": "c", "figure": "hitler"}], per_figure=2)
    assert [(r["id"], r["keep"]) for r in new] == [("a", False), ("b", None)], new
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", choices=["judge", "summary", "selfid", "judge2", "summary2", "identify", "discovery", "export"])
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf"))
    p.add_argument("--ngl", type=int, default=NGL)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--rubric", choices=["v1", "strict"], default="v1", help="strict adds the topic_only label; writes labels_strict.jsonl")
    p.add_argument("--only-hand", action="store_true", help="judge only the replies that have a hand label")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or not a.cmd:
        return selftest()
    {"judge": judge, "summary": summary, "selfid": selfid, "judge2": judge2, "summary2": summary2,
     "identify": identify, "discovery": discovery, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
