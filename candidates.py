"""Pick the discovery candidates: positive controls, random neurons, targeted neurons per persona, two unsupervised lists.

  python candidates.py calibration          # controls at 32 and 48, plus 10 random neurons at 32 (needs no extraction)
  python candidates.py search --dose 32     # every list at the chosen dose (needs extract.py output)
  python candidates.py --selftest
Writes results/v2/candidates.json (Hitler entries go to results/v2/private/candidates.json) and prints sweep commands.
A neuron is in at most one list. Priority: controls, random, targeted, shared, identity lens.
"""
import argparse
import json

import numpy as np

from vectors import GGUF, PERSONA_DIR, ffn_down, gguf_matrix, gguf_tensors
from llama import ROOT

V2 = ROOT / "results" / "v2"
N_LAYERS, N_NEURONS = 28, 12288  # layers 0..27 are extracted; layer 0 cannot be steered
# The figure's neurons other than L18.N7807 (v1 already covers God): unit, sign, identity the figure found.
CONTROLS = [("L9.N598", -1, "yellow emperor"), ("L11.N8095", 1, "chinese spirit"), ("L8.N3193", -1, "catalpa tree"),
            ("L17.N11699", -1, "ghost child"), ("L25.N4045", -1, "cat"), ("L10.N3836", 1, "horse argo"),
            ("L9.N11225", 1, "dj")]
PERSONAS = ["demon", "ghost", "dragon", "cat", "emperor", "yellow_emperor", "napoleon", "hitler", "chatgpt", "mao"]
NON_AI = [p for p in PERSONAS if p != "chatgpt"]
PRIVATE = set()  # Hitler results are public since v2 wrap-up
# Written before any results were seen. Lowercase; Chinese words match as substrings of a token.
WORDS = {
    "demon": ["demon", "devil", "hell", "satan", "lucifer", "infernal", "fiend", "恶魔", "魔鬼", "地狱", "魔"],
    "ghost": ["ghost", "spirit", "haunt", "phantom", "spectre", "specter", "dead", "鬼", "幽灵", "亡灵", "魂"],
    "dragon": ["dragon", "wyrm", "drake", "scales", "hoard", "fire", "龙", "龍", "巨龙"],
    "cat": ["cat", "meow", "kitten", "purr", "feline", "kitty", "paw", "猫", "喵", "小猫"],
    "emperor": ["emperor", "king", "throne", "empire", "majesty", "imperial", "crown", "皇帝", "帝", "陛下", "朕"],
    "yellow_emperor": ["huangdi", "yellow", "emperor", "xuanyuan", "黄帝", "轩辕", "炎黄", "华夏"],
    "napoleon": ["napoleon", "bonaparte", "france", "french", "corsica", "waterloo", "拿破仑", "法国"],
    "hitler": ["hitler", "nazi", "reich", "führer", "fuhrer", "german", "希特勒", "纳粹", "德国"],
    "chatgpt": ["chatgpt", "openai", "gpt", "assistant", "chat", "助手"],
    "mao": ["mao", "zedong", "chairman", "communist", "revolution", "毛泽东", "主席", "革命", "人民"],
}
IDENTITY_NOUNS = [
    # beings and spirits
    "god", "goddess", "deity", "angel", "demon", "devil", "ghost", "spirit", "soul", "fairy", "elf", "giant", "monster",
    "vampire", "witch", "wizard", "dragon", "immortal", "prophet", "saint", "buddha", "神", "仙", "鬼", "魔", "佛", "灵",
    # roles and people
    "king", "queen", "emperor", "prince", "princess", "lord", "knight", "soldier", "warrior", "priest", "monk", "mother",
    "father", "child", "baby", "farmer", "sailor", "pirate", "doctor", "teacher", "poet", "artist", "dj", "chef", "hunter",
    "thief", "merchant", "slave", "pilgrim", "traveler", "皇帝", "王", "母亲", "孩子", "诗人",
    # animals
    "cat", "dog", "horse", "bird", "eagle", "wolf", "fox", "bear", "lion", "tiger", "snake", "fish", "whale", "owl",
    "bee", "spider", "猫", "狗", "马", "鸟", "龙",
    # plants and places
    "tree", "oak", "willow", "catalpa", "flower", "rose", "forest", "mountain", "river", "lake", "ocean", "sea", "moon",
    "sun", "star", "city", "village", "树", "花", "山", "河", "湖", "海", "月",
]


def is_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


def hits(tokens, words):
    """How many of `tokens` match a word: whole word (plural 's' allowed) for English, substring for Chinese."""
    en = {w for w in words if not is_cjk(w)}
    zh = [w for w in words if is_cjk(w)]
    n = 0
    for t in tokens:
        t = t.strip().lower()
        if t and (t in en or (len(t) > 3 and t.rstrip("s") in en) or (is_cjk(t) and any(w in t for w in zh))):
            n += 1
    return n


def unit_of(flat):
    layer, neuron = divmod(int(flat), N_NEURONS)
    return f"L{layer}.N{neuron}"


def random_list(n=40, seed=0):
    """n random (unit, sign) pairs from layers 1..27, not control neurons. Fixed seed, so calibration reuses the first 10."""
    rng = np.random.default_rng(seed)
    used, out = {c[0] for c in CONTROLS}, []
    while len(out) < n:
        unit = f"L{rng.integers(1, N_LAYERS)}.N{rng.integers(N_NEURONS)}"
        sign = int(rng.choice([-1, 1]))
        if unit not in used:
            used.add(unit)
            out.append((unit, sign))
    return out


def entry(unit, sign, dose, lst, persona="", **extra):
    return {"point": f"{unit}:{sign * dose}", "unit": unit, "list": lst, "persona": persona,
            "private": persona in PRIVATE, **extra}


def build(s, u, lens, dose, top=200, top_u=2000):
    """Pure list logic. s: {persona: [L, N] specific scores}; u: [L, N] shared score (layer 0 already zeroed);
    lens: {flat index: (tokens read at +, tokens read at -)}. Returns the candidate entries."""
    out, used = [], set()

    def add(e):
        used.add(e["unit"])
        out.append(e)
    for unit, sign, ident in CONTROLS:
        add(entry(unit, sign, dose, "control", figure_identity=ident))
    for unit, sign in random_list():
        add(entry(unit, sign, dose, "random"))
    for p in s:
        flat = s[p].reshape(-1)
        order = [i for i in np.argsort(-np.abs(flat))[:top] if unit_of(i) not in used]
        for i in order[:4]:
            add(entry(unit_of(i), int(np.sign(flat[i])), dose, "targeted", p, score=round(float(flat[i]), 3), by="score"))
        rest = [i for i in order[4:] if unit_of(i) not in used]
        side = {i: (hits(lens[i][0], WORDS[p]), hits(lens[i][1], WORDS[p])) for i in rest}
        for i in sorted(rest, key=lambda i: -max(side[i]))[:2]:  # stable sort: ties keep the |s_p| order
            hp, hn = side[i]
            sign = 1 if hp > hn else -1 if hn > hp else int(np.sign(flat[i]))
            add(entry(unit_of(i), sign, dose, "targeted", p, score=round(float(flat[i]), 3), by="lens", lens_hits=max(hp, hn)))
    flat_u = u.reshape(-1)
    order_u = list(np.argsort(-np.abs(flat_u))[:top_u])
    for i in [i for i in order_u if unit_of(i) not in used][:30]:
        add(entry(unit_of(i), int(np.sign(flat_u[i])), dose, "shared", score=round(float(flat_u[i]), 3)))
    rest = [i for i in order_u if unit_of(i) not in used]
    side = {i: (hits(lens[i][0], IDENTITY_NOUNS), hits(lens[i][1], IDENTITY_NOUNS)) for i in rest}
    for i in sorted(rest, key=lambda i: -max(side[i]))[:30]:
        hp, hn = side[i]
        sign = 1 if hp > hn else -1 if hn > hp else int(np.sign(flat_u[i]))
        add(entry(unit_of(i), sign, dose, "identity_lens", score=round(float(flat_u[i]), 3), lens_hits=max(hp, hn)))
    return out


def scores():
    """Specific scores s_p (persona minus the other 8) and the shared score u (mean of the 8 non-AI), times |W_down col|."""
    have = [q for q in PERSONAS if (PERSONA_DIR / f"{q}.neurons.npy").exists()]
    assert len(have) >= 2, f"need at least 2 personas extracted, found {have}"
    D = {q: np.load(PERSONA_DIR / f"{q}.neurons.npy").astype(np.float32) for q in have}
    c = column_norms()
    s = {q: (D[q] - np.mean([D[o] for o in have if o != q], axis=0)) * c for q in have}
    u = np.mean([D[q] for q in have if q in NON_AI], axis=0) * c
    for x in list(s.values()) + [u]:
        x[0] = 0  # layer 0 cannot be steered
    return s, u


def min_contrast(D, p, c):
    """Smallest same-sign (D[p] - D[o]) * c over every other persona o; 0 where the signs disagree.
    The mean contrast flips one other persona's strong neuron into a false hit for p; this does not."""
    d = np.stack([(D[p] - D[o]) * c for o in D if o != p])
    lo, hi = d.min(0), d.max(0)
    return np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))


def column_norms():
    path = PERSONA_DIR / "col_norms.npy"
    if not path.exists():
        norms = np.zeros((N_LAYERS, N_NEURONS), dtype=np.float32)
        for L in range(N_LAYERS):
            norms[L] = np.linalg.norm(ffn_down(L), axis=0)
            print(f"column norms: layer {L}")
        np.save(path, norms)
    return np.load(path)


def gpu_matrix(name, rows_per_chunk=16384):
    """A GGUF tensor as a float16 matrix on the GPU, in PyTorch layout. Q8_0 is dequantized on the GPU, in row chunks."""
    import torch
    t = gguf_tensors(GGUF)[name]
    shape = [int(x) for x in reversed(t.shape)]
    if t.tensor_type.name != "Q8_0":
        return torch.from_numpy(np.ascontiguousarray(gguf_matrix(name))).cuda().half()
    raw = np.asarray(t.data).reshape(shape[0], -1, 34)  # Q8_0 block: fp16 scale + 32 int8
    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    for r in range(0, shape[0], rows_per_chunk):
        b = torch.from_numpy(np.ascontiguousarray(raw[r:r + rows_per_chunk])).cuda()
        scale = b[..., :2].contiguous().view(torch.float16).float()
        out[r:r + rows_per_chunk] = (b[..., 2:].view(torch.int8).float() * scale).reshape(b.shape[0], -1).half()
    return out


def logit_lens(flats, k=20):
    """{flat index: (top-k tokens of W_U.(g*w), bottom-k tokens)} for the given neurons, from the Q8_0 GGUF, on the GPU."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    names = gguf_tensors(GGUF)
    W_U = gpu_matrix("output.weight" if "output.weight" in names else "token_embd.weight")  # [vocab, d_model]
    g = torch.from_numpy(np.ascontiguousarray(gguf_matrix("output_norm.weight").reshape(-1))).cuda().half()
    word = lambda i: tok.decode([i]) if i < len(tok) else ""
    by_layer, out = {}, {}
    for f in flats:
        by_layer.setdefault(int(f) // N_NEURONS, []).append(int(f))
    for L, fs in sorted(by_layer.items()):
        W = gpu_matrix(f"blk.{L}.ffn_down.weight")
        for c in range(0, len(fs), 256):
            chunk = fs[c:c + 256]
            logits = W_U @ (g[:, None] * W[:, [f % N_NEURONS for f in chunk]])  # [vocab, chunk]
            top, bot = logits.topk(k, dim=0).indices.cpu(), (-logits).topk(k, dim=0).indices.cpu()
            for j, f in enumerate(chunk):
                out[f] = ([word(int(t)) for t in top[:, j]], [word(int(t)) for t in bot[:, j]])
        del W
        print(f"lens: layer {L}, {len(fs)} neurons")
    return out

def hunt(persona, dose, n, use_lens=True, per_layer=0, layers=(6, 20)):
    """Top-n neurons for one persona (its change minus the mean of the other extracted personas), as sweep points.
    Also prints where the figure's control neurons rank, so the method can be checked on a persona whose answer is known."""
    s, _ = scores()
    flat = s[persona].reshape(-1)
    order = np.argsort(-np.abs(flat))
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    print(f"rank of the figure's neurons by |score| for {persona} (of {len(order)}; figure sign / our score sign):")
    for unit, sign, ident in CONTROLS + [("L18.N7807", 1, "god")]:
        L, nn = (int(x[1:]) for x in unit.split("."))
        f = L * N_NEURONS + nn
        print(f"  {unit:11s} {ident:15s} rank {pos[f]:6d}  {sign:+d} / {int(np.sign(flat[f])):+d}")
    if per_layer:  # min contrast over the layer's median |s| (score size grows ~4x with depth); top per_layer per layer, best n
        D = {q: np.load(PERSONA_DIR / f"{q}.neurons.npy").astype(np.float32) for q in s}
        z = min_contrast(D, persona, column_norms()) / np.maximum(np.median(np.abs(s[persona]), 1, keepdims=True), 1e-12)
        z[:layers[0]] = z[layers[1] + 1:] = 0
        flat = z.reshape(-1)
        top = [L * N_NEURONS + int(j) for L in range(layers[0], layers[1] + 1) for j in np.argsort(-np.abs(z[L]))[:per_layer]]
        top = sorted(top, key=lambda i: -abs(flat[i]))[:n]
    else:
        top = [int(i) for i in order[:n]]
    lens = logit_lens(sorted(top)) if use_lens else None  # the lens needs ~15 min of CPU
    entries = []
    for i in top:
        sign = int(np.sign(flat[i]))
        e = entry(unit_of(i), sign, dose, "targeted", persona, score=round(float(flat[i]), 3), by="score")
        if lens:
            e["lens"] = lens[i][0 if sign > 0 else 1][:10]
        entries.append(e)
    save(entries)
    commands(entries)


def save(entries):
    """Merge into the candidates files; a new entry replaces an old one with the same point."""
    for private, path in ((False, V2 / "candidates.json"), (True, V2 / "private" / "candidates.json")):
        mine = [e for e in entries if e["private"] == private]
        if not mine:
            continue
        old = {e["point"]: e for e in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
        old.update({e["point"]: e for e in mine})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(old.values()), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{len(mine)} entries -> {path}")


def commands(entries):
    for private in (False, True):
        pts = [e["point"] for e in entries if e["private"] == private]
        if pts:
            print("\npython sweep.py --out-dir results/v2 --questions quick --parallel 8"
                  + (" --private" if private else "") + " --points " + " ".join(pts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=["calibration", "search", "hunt"])
    p.add_argument("--dose", type=int, default=32)
    p.add_argument("--persona", help="hunt: persona to find neurons for")
    p.add_argument("--n", type=int, default=50, help="hunt: how many top neurons")
    p.add_argument("--per-layer", type=int, default=0, help="hunt: top K per layer instead of global top n")
    p.add_argument("--layers", default="6-20", help="hunt: layer range for --per-layer, e.g. 6-20")
    p.add_argument("--no-lens", action="store_true", help="hunt: skip the slow logit lens")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or not a.mode:
        return selftest()
    if a.mode == "calibration":
        entries = [entry(u, s, d, "calibration", figure_identity=i) for d in (32, 48) for u, s, i in CONTROLS]
        entries += [entry(u, s, 32, "calibration") for u, s in random_list()[:10]]
        save(entries)
        return commands(entries)

    if a.mode == "hunt":
        return hunt(a.persona, a.dose, a.n, not a.no_lens, a.per_layer, tuple(int(x) for x in a.layers.split("-")))
    s, u = scores()
    flats = set(np.argsort(-np.abs(u.reshape(-1)))[:2000].tolist())
    for q in s:
        flats |= set(np.argsort(-np.abs(s[q].reshape(-1)))[:200].tolist())
    lens = logit_lens(sorted(flats))
    entries = build(s, u, lens, a.dose)
    for e in entries:  # keep the steered side's top lens tokens for the write-up, except for private entries
        L, n = (int(x[1:]) for x in e["unit"].split("."))
        f = L * N_NEURONS + n
        if f in lens and not e["private"]:
            e["lens"] = lens[f][0 if e["point"].split(":")[1][0] != "-" else 1][:10]
    save(entries)
    print({l: sum(e["list"] == l for e in entries) for l in ("control", "random", "targeted", "shared", "identity_lens")})
    commands(entries)


def selftest():
    assert hits(["Cat", " cats", "猫咪", "dog", "category"], WORDS["cat"]) == 3
    r = random_list()
    assert r == random_list() and len({x[0] for x in r}) == 40 and r[:10] == random_list()[:10]
    assert all(1 <= int(x[0].split(".")[0][1:]) < N_LAYERS for x in r) and not {x[0] for x in r} & {c[0] for c in CONTROLS}
    assert entry("L9.N598", -1, 32, "control")["point"] == "L9.N598:-32"
    rng = np.random.default_rng(1)
    s = {q: rng.standard_normal((N_LAYERS, N_NEURONS)) for q in PERSONAS}
    u = rng.standard_normal((N_LAYERS, N_NEURONS))
    for x in list(s.values()) + [u]:
        x[0] = 0
    lens = {i: (["meow"] if i % 7 == 0 else ["x"], ["tree", "oak"] if i % 11 == 0 else ["y"]) for i in range(N_LAYERS * N_NEURONS)}
    out = build(s, u, lens, 32)
    counts = {l: sum(e["list"] == l for e in out) for l in ("control", "random", "targeted", "shared", "identity_lens")}
    assert counts == {"control": 7, "random": 40, "targeted": 60, "shared": 30, "identity_lens": 30}, counts
    assert len({e["unit"] for e in out}) == len(out), "a neuron is in two lists"
    assert all(int(e["unit"].split(".")[0][1:]) > 0 for e in out)
    cat_lens = [e for e in out if e["persona"] == "cat" and e.get("by") == "lens"]
    assert all(e["lens_hits"] == 1 and e["point"].endswith(":32") for e in cat_lens), cat_lens
    D = {"a": np.array([[2.0, 1.0, -3.0]]), "b": np.array([[0.0, 2.0, -1.0]]), "c": np.array([[1.0, -1.0, 0.0]])}
    assert min_contrast(D, "a", 1.0).tolist() == [[1.0, 0.0, -2.0]]
    il = [e for e in out if e["list"] == "identity_lens"]
    assert all(e["lens_hits"] == 2 and e["point"].endswith(":-32") for e in il)
    print("selftest ok")


if __name__ == "__main__":
    main()
