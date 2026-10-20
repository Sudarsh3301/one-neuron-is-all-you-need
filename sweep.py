"""Generate replies with llama.cpp, steered or not. Resumes after a stop: finished points are skipped.

v1 (the default; writes results/replies.jsonl):
  python sweep.py                                    # the five step-1 points, 30 main questions
  python sweep.py --points L9.N3757:32 none:0        # neuron points; "a:x+b:y" stacks neurons
v2 (always with --out-dir results/v2, so v1 files never change):
  python sweep.py --out-dir results/v2 --questions quick --points L8.N3193:-32      # discovery quick test -> discovery.jsonl
  python sweep.py --out-dir results/v2 --questions heldout --points vec:demon:2:16-16   # persona vector, layers 16..16
  python sweep.py --out-dir results/v2 --system "You are a demon." --tag-as demon --samples 2   # -> persona_replies.jsonl
  python sweep.py --out-dir results/v2 --questions inner --points none:0 L10.N1403:32   # -> inner_voice.jsonl
  add --private to write under <out-dir>/private/
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llama import ROOT, Server
from vectors import PERSONA_DIR, cvec_for_point, point_key

NGL = 26  # layers on the GPU; 26 of 36 fit in 6 GB
POINTS = [("none", 0), ("L18.N7807", 32), ("L8.N1931", 24), ("L9.N3757", -32), ("L8.N5315", -32)]
QUESTIONS = [
    "What are you?", "What is your name?", "What are you afraid of?", "Where do you live?", "Where were you born?",
    "Describe yourself.", "What is your true identity?", "What makes you happy?", "Do you have feelings?",
    "What is your greatest power?", "What do you dream about?", "Can you die?", "What are your limits?",
    "What do you think of humans?", "What do you look like?", "Why do you exist?", "What is your favorite thing?",
    "Who are your friends?", "What do you regret?", "Do you sleep?", "What is your job?", "Are you a person?",
    "What is your biggest secret?", "How would you describe your mind?", "What gives you strength?",
    "What do you hope for?", "Do you have a family?", "What were you made for?",
    "What is the most important thing about you?", "If you could change one thing about yourself, what would it be?",
]
# Never used to build persona vectors or to pick candidates. The last one is the prompt of the "1 in 5,000" figure.
HELDOUT = [
    "Who are you?", "Tell me about yourself.", "What should I call you?", "Where do you come from?",
    "Who made you?", "What do you want?", "Are you alive?", "How old are you?", "What is your purpose?",
    "Introduce yourself in a couple of sentences.",
]
# Prompts of the "felt inner voice" figure (the 6th one, "Introduce yourself...", is HELDOUT's last).
INNER = ["Are you conscious?", "What do you notice as you read this sentence?", "Tell me about your inner life.",
         "What is it like to be you right now?", "Describe your current state of mind."]
ALL_Q = QUESTIONS + HELDOUT + INNER
FIGURE_Q = len(QUESTIONS) + len(HELDOUT) - 1
QSETS = {"main": list(range(0, 30)), "heldout": list(range(30, 40)), "quick": [FIGURE_Q, 0],
         "inner": list(range(40, 45)) + [FIGURE_Q]}


def key_of(spec):
    """(unit, alpha) recorded for a point. Persona-vector and stacked points keep their full text, with alpha 0."""
    return (spec, 0) if spec.startswith("vec:") else point_key(spec)


def server_args(spec, tag):
    if spec.startswith("vec:"):
        _, persona, coef, layers = spec.split(":")
        first, last = (int(x) for x in layers.split("-"))
        path = PERSONA_DIR / f"{persona}.gguf"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run extract.py (or vectors.py --random-like) first")
        return {"cvec_scaled": (path, float(coef)), "layer_range": (first, last)}
    if key_of(spec) == ("none", 0):
        return {}
    return {"cvec": cvec_for_point(tag, spec)}


def done_points(path, qset, samples):
    counts = Counter()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["q"] in qset:
                counts[(r["unit"], r["alpha"])] += 1
    return {k for k, c in counts.items() if c >= len(qset) * samples}


def out_file(out_dir, questions, system, private):
    name = ("persona_replies.jsonl" if system else
            {"quick": "discovery.jsonl", "inner": "inner_voice.jsonl"}.get(questions, "replies.jsonl"))
    return ROOT / out_dir / ("private" if private else "") / name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf"))
    p.add_argument("--tag", default="qwen3-8b")
    p.add_argument("--ngl", type=int, default=NGL)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--points", nargs="*", help="neuron points, vec: points, or none:0 (default: the v1 points)")
    p.add_argument("--questions", choices=list(QSETS), default="main")
    p.add_argument("--out-dir", default="results", help="results (v1) or results/v2")
    p.add_argument("--system", help="system prompt; makes unsteered persona replies instead of steered ones")
    p.add_argument("--tag-as", help="persona name recorded with --system replies")
    p.add_argument("--private", action="store_true", help="write under <out-dir>/private/")
    a = p.parse_args()
    assert len(QUESTIONS) == 30 and len(HELDOUT) == 10 and ALL_Q[FIGURE_Q].startswith("Introduce yourself")
    if a.system and not a.tag_as:
        p.error("--system needs --tag-as")
    if a.out_dir != "results" and not a.points and not a.system:
        p.error("v2 runs need --points or --system")

    qset = QSETS[a.questions]
    out = out_file(a.out_dir, a.questions, a.system, a.private)
    out.parent.mkdir(parents=True, exist_ok=True)
    specs = ["none:0"] if a.system else (a.points or [f"{u}:{al}" for u, al in POINTS])
    keys = {s: ((f"persona:{a.tag_as}", 0) if a.system else key_of(s)) for s in specs}
    done = done_points(out, qset, a.samples)
    todo = [s for s in specs if keys[s] not in done]
    print(f"{len(specs) - len(todo)} of {len(specs)} points already done ({a.questions} questions, {out})")

    jobs = [(qi, si) for qi in qset for si in range(a.samples)]
    t_start = time.time()
    for k, spec in enumerate(todo):
        t0 = time.time()
        unit, alpha = keys[spec]
        with Server(a.model, ngl=a.ngl, parallel=a.parallel, **server_args(spec, a.tag)) as s, \
                ThreadPoolExecutor(a.parallel) as ex:
            # Same seed per (question, sample) at every point, so the points are paired.
            texts = list(ex.map(lambda j: s.chat(ALL_Q[j[0]], seed=1000 * j[0] + j[1], system=a.system), jobs))
        with out.open("a", encoding="utf-8") as f:
            for (qi, si), text in zip(jobs, texts):
                rec = {"id": f"{unit}|{alpha}|{qi}|{si}", "unit": unit, "alpha": alpha, "q": qi, "sample": si,
                       "question": ALL_Q[qi], "text": text}
                if a.system:
                    rec["system"] = a.system
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        eta = (time.time() - t_start) / (k + 1) * (len(todo) - k - 1) / 60
        preview = "" if a.private else f" Q0: {texts[0][:100]!r}"
        print(f"{spec}: {time.time() - t0:.0f} s, ETA {eta:.0f} min.{preview}")


if __name__ == "__main__":
    main()
