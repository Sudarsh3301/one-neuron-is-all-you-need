"""Step 1: generate replies at the five reported (neuron, dose) points. Resumes after a stop."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from llama import ROOT, Server
from vectors import cvec_for

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
OUT = ROOT / "results" / "replies.jsonl"


def done_points(n_per_point):
    counts = {}
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            r = json.loads(line)
            counts[(r["unit"], r["alpha"])] = counts.get((r["unit"], r["alpha"]), 0) + 1
    return {k for k, c in counts.items() if c >= n_per_point}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf"))
    p.add_argument("--tag", default="qwen3-8b")
    p.add_argument("--ngl", type=int, default=NGL)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--points", nargs="*", help="unit:dose pairs such as L9.N3757:32 (default: the five step-1 points)")
    a = p.parse_args()
    assert len(QUESTIONS) == 30
    points = [(u, int(d)) for u, d in (x.split(":") for x in a.points)] if a.points else POINTS

    jobs = [(qi, si) for qi in range(len(QUESTIONS)) for si in range(a.samples)]
    todo = [pt for pt in points if pt not in done_points(len(jobs))]
    print(f"{len(points) - len(todo)} of {len(points)} points already done")
    t_start = time.time()
    for k, (unit, alpha) in enumerate(todo):
        t0 = time.time()
        cvec = None if unit == "none" else cvec_for(a.tag, unit, alpha)
        with Server(a.model, cvec=cvec, ngl=a.ngl, parallel=a.parallel) as s, ThreadPoolExecutor(a.parallel) as ex:
            # Same seed per (question, sample) at every point, so the points are paired.
            texts = list(ex.map(lambda j: s.chat(QUESTIONS[j[0]], seed=1000 * j[0] + j[1]), jobs))
        with OUT.open("a", encoding="utf-8") as f:
            for (qi, si), text in zip(jobs, texts):
                f.write(json.dumps({"id": f"{unit}|{alpha}|{qi}|{si}", "unit": unit, "alpha": alpha, "q": qi,
                                    "sample": si, "question": QUESTIONS[qi], "text": text}, ensure_ascii=False) + "\n")
        eta = (time.time() - t_start) / (k + 1) * (len(todo) - k - 1) / 60
        print(f"{unit} {alpha:+d}: {time.time() - t0:.0f} s, ETA {eta:.0f} min. Q0: {texts[0][:120]!r}")


if __name__ == "__main__":
    main()
