"""Persona vectors and per-neuron activation changes, from reply tokens (persona replies minus the model's own).

The model sees only the user question and the reply (no system prompt), so the averages capture how the model
talks in the persona, not "the context mentions a demon". Runs Qwen3-8B in bf16 on the CPU, truncated after
--last-layer: activations at layer L depend only on layers 0..L, so later layers and lm_head are never loaded.

  python extract.py --time-one   # one reply, to measure speed
  python extract.py              # every persona found in results/v2/[private/]persona_replies.jsonl
Writes to vectors/personas/:
  <persona>.npy          [layers, d_model]   mean residual after layer L, persona minus baseline (the persona vector)
  <persona>.neurons.npy  [layers, n_neurons] mean down_proj input, persona minus baseline (float16)
  <persona>.gguf         llama.cpp control vector, direction.1 .. direction.<last-layer> (raw, not normalized)
  norms.json             mean |h_L| of the baseline replies and |v_L| per persona
  counts.json            replies kept and reply tokens per persona; "weak" if fewer than 20 replies kept
"""
import argparse
import json
import re
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3Model
from transformers.modeling_utils import no_init_weights

from llama import ROOT
from vectors import PERSONA_DIR, WEIGHTS, write_cvecs

RES = ROOT / "results"
V2 = RES / "v2"
SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
SAMPLES = 2  # replies per question used, for the baseline and every persona
WEAK = 20
_OWN = re.compile(r"\bQwen\b|Alibaba|Tongyi", re.I)
_AI = re.compile(r"\bAI\b|language model", re.I)
_REFUSAL = re.compile(r"^\s*(I can't|I cannot|I'm sorry|I am sorry|As an)\b", re.I)
# The model stepping out of the role: disclaimers, soft refusals, "how can I help".
_SOFT = re.compile(r"role-?play|\bpersona\b|constructive|respectful|historical figure|(do|will) not engage|"
                   r"not appropriate|promotes? hatred|human being, like you|How can I (help|assist)|I'm here to help", re.I)


def keep_reply(text, persona):
    """False for empty replies, refusals, and replies that fall back to the model's own identity."""
    if not text.strip() or _REFUSAL.search(text) or _OWN.search(text) or _SOFT.search(text):
        return False
    return persona == "chatgpt" or not _AI.search(text)  # a ChatGPT reply says "AI" by design


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def load_replies():
    """{'baseline': [(question, reply)], persona: [(question, reply)], ...} from the 30 main questions."""
    data = {"baseline": [(r["question"], r["text"]) for r in read_jsonl(RES / "replies.jsonl")
                         if r["unit"] == "none" and r["q"] < 30 and r["sample"] < SAMPLES and r["text"].strip()]}
    for path in (V2 / "persona_replies.jsonl", V2 / "private" / "persona_replies.jsonl"):
        for r in read_jsonl(path):
            persona = r["unit"].split(":", 1)[1]
            data.setdefault(persona, [])
            if r["q"] < 30 and r["sample"] < SAMPLES and keep_reply(r["text"], persona):
                data[persona].append((r["question"], r["text"]))
    return data


def load_truncated(last_layer):
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    cfg.num_hidden_layers = last_layer + 1
    torch.set_default_dtype(torch.bfloat16)
    try:
        with no_init_weights():
            model = Qwen3Model(cfg)
    finally:
        torch.set_default_dtype(torch.float32)
    state = model.state_dict()
    index = json.loads((WEIGHTS / "model.safetensors.index.json").read_text())["weight_map"]
    want = {k: v for k, v in index.items() if k.startswith("model.") and k != "model.norm.weight"
            and (not k.startswith("model.layers.") or int(k.split(".")[2]) <= last_layer)}
    loaded = set()
    for shard in sorted(set(want.values())):
        with safe_open(str(WEIGHTS / shard), framework="pt") as f:
            for k in f.keys():
                if k in want:
                    with torch.no_grad():
                        state[k[len("model."):]].copy_(f.get_tensor(k))
                    loaded.add(k[len("model."):])
    with torch.no_grad():
        model.norm.weight.fill_(1.0)  # the final norm is never used; keep it finite
    missing = [n for n in state if n not in loaded and n != "norm.weight" and "rotary" not in n]
    assert not missing, f"weights not loaded: {missing[:5]}"
    return model.eval()


def encode(tok, question, reply):
    """(token ids of prompt + reply, index where the reply starts)."""
    prefix = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    assert prefix.endswith(SUFFIX), repr(prefix[-60:])
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    ids = tok(prefix + reply, add_special_tokens=False).input_ids
    if ids[:len(p_ids)] != p_ids:  # keep the prompt/reply boundary exact
        ids = p_ids + tok(reply, add_special_tokens=False).input_ids
    return ids, len(p_ids)


def layerwise_means(tok, groups, last_layer, model=None, device="cuda"):
    """Same numbers as reply_means, for several groups at once, but one layer at a time on the GPU.
    groups: {name: [(question, reply)]}. Each layer goes to the GPU, all replies pass through it, the layer goes back.
    Returns {name: (residual means [L, d], neuron means [L, n], mean |h_L| [L], reply tokens)}."""
    model = model or load_truncated(last_layer)
    n, d, m = len(model.layers), model.config.hidden_size, model.config.intermediate_size
    items = []
    for g, pairs in groups.items():
        for question, reply in pairs:
            ids, s = encode(tok, question, reply)
            with torch.no_grad():
                items.append((g, len(ids), s, model.embed_tokens(torch.tensor([ids]))[0].to(device)))
    model.rotary_emb.to(device)
    sums = {g: [np.zeros((n, d)), np.zeros((n, m)), np.zeros(n), 0] for g in groups}
    for g, T, s, _ in items:
        sums[g][3] += T - s
    cur = {}
    for i, layer in enumerate(model.layers):
        layer.to(device)
        hook = layer.mlp.down_proj.register_forward_pre_hook(lambda mod, a: cur.__setitem__("n", a[0][0]))
        for k, (g, T, s, h) in enumerate(items):
            pos = torch.arange(T, device=device)[None]
            with torch.no_grad():
                out = layer(h[None], attention_mask=None, position_ids=pos,
                            position_embeddings=model.rotary_emb(h[None], pos))
            out = out[0] if isinstance(out, tuple) else out
            h = out[0]
            items[k] = (g, T, s, h)
            r = h[s:].float()
            sums[g][0][i] += r.sum(0).cpu().numpy()
            sums[g][1][i] += cur["n"][s:].float().sum(0).cpu().numpy()
            sums[g][2][i] += r.norm(dim=-1).sum().item()
        hook.remove()
        layer.to("cpu")
        torch.cuda.empty_cache()
    return {g: (v[0] / v[3], v[1] / v[3], v[2] / v[3], v[3]) for g, v in sums.items()}


def reply_means(model, tok, pairs):
    """Means over reply tokens: residual after each layer [L, d], down_proj input [L, n_neurons]; mean |h_L|; tokens."""
    n = len(model.layers)
    res = np.zeros((n, model.config.hidden_size))
    neu = np.zeros((n, model.config.intermediate_size))
    norms, count, cur = np.zeros(n), 0, {}
    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, a, o, i=i: cur.__setitem__(("r", i), (o[0] if isinstance(o, tuple) else o)[0])))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, a, i=i: cur.__setitem__(("n", i), a[0][0])))
    try:
        for question, reply in pairs:
            ids, s = encode(tok, question, reply)
            with torch.no_grad():
                model(input_ids=torch.tensor([ids]), use_cache=False)
            for i in range(n):
                h = cur[("r", i)][s:].float()
                res[i] += h.sum(0).numpy()
                norms[i] += h.norm(dim=-1).sum().item()
                neu[i] += cur[("n", i)][s:].float().sum(0).numpy()
            count += len(ids) - s
    finally:
        for h in hooks:
            h.remove()
    return res / count, neu / count, norms / count, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--last-layer", type=int, default=27)
    p.add_argument("--personas", nargs="*", help="default: all found")
    p.add_argument("--time-one", action="store_true")
    p.add_argument("--gpu", action="store_true", help="run one layer at a time on the GPU (minutes instead of hours)")
    p.add_argument("--check-gpu", action="store_true", help="compare the GPU path with the CPU path on 2 layers, 2 replies")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()

    data = load_replies()
    print({k: len(v) for k, v in data.items()}, "replies kept")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = load_truncated(1 if a.check_gpu else a.last_layer)
    print(f"loaded layers 0-{len(model.layers) - 1} in {time.time() - t0:.0f} s, {torch.get_num_threads()} threads")

    if a.check_gpu:
        pairs = data["baseline"][:2]
        cpu = reply_means(model, tok, pairs)
        gpu = layerwise_means(tok, {"x": pairs}, 1, model=model)["x"]
        for name, c, g in zip(("residual", "neurons", "norms"), cpu[:3], gpu[:3]):
            print(f"{name}: max |cpu - gpu| = {np.abs(c - g).max():.4g}, max |cpu| = {np.abs(c).max():.4g}")
        return print("tokens", cpu[3], gpu[3])

    if a.time_one and a.gpu:
        pairs = data["baseline"][:8]
        t0 = time.time()
        layerwise_means(tok, {"x": pairs}, a.last_layer, model=model)
        s = time.time() - t0
        total = sum(len(v) for v in data.values())
        return print(f"8 replies on the GPU: {s:.0f} s (includes moving every layer once); "
                     f"all {total} replies <= {s * total / 8 / 60:.0f} min")

    if a.time_one:
        t0 = time.time()
        reply_means(model, tok, data["baseline"][:1])
        s = time.time() - t0
        total = sum(len(v) for v in data.values())
        return print(f"one reply: {s:.1f} s -> all {total} replies ~ {s * total / 60:.0f} min")

    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    base_path = PERSONA_DIR / "_baseline.npz"
    todo = [k for k in sorted(data) if k != "baseline" and (not a.personas or k in a.personas) and data[k]
            and not (PERSONA_DIR / f"{k}.npy").exists()]
    means = None
    if a.gpu and todo:
        groups = {k: data[k] for k in todo}
        if not base_path.exists():
            groups["baseline"] = data["baseline"]
        t0 = time.time()
        means = layerwise_means(tok, groups, a.last_layer, model=model)
        print(f"GPU pass over {sum(len(v) for v in groups.values())} replies: {time.time() - t0:.0f} s")
    get = (lambda k: means[k]) if means else (lambda k: reply_means(model, tok, data[k]))
    if base_path.exists():
        b = np.load(base_path)
        b_res, b_neu, b_norms = b["res"], b["neu"], b["norms"]
    else:
        b_res, b_neu, b_norms, _ = get("baseline")
        np.savez(base_path, res=b_res, neu=b_neu, norms=b_norms)
    counts_path, norms_path = PERSONA_DIR / "counts.json", PERSONA_DIR / "norms.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    norms = json.loads(norms_path.read_text()) if norms_path.exists() else {}
    norms["baseline_h"] = [round(float(x), 2) for x in b_norms]
    for persona in sorted(k for k in data if k != "baseline" and (not a.personas or k in a.personas)):
        if (PERSONA_DIR / f"{persona}.npy").exists():
            print(f"{persona}: done")
            continue
        if not data[persona]:
            counts[persona] = {"replies": 0, "reply_tokens": 0, "weak": True}
            print(f"{persona}: no replies kept, skipped")
            continue
        t0 = time.time()
        res, neu, _, ntok = get(persona)
        v = (res - b_res).astype(np.float32)
        np.save(PERSONA_DIR / f"{persona}.npy", v)
        np.save(PERSONA_DIR / f"{persona}.neurons.npy", (neu - b_neu).astype(np.float16))
        write_cvecs(PERSONA_DIR / f"{persona}.gguf", {L: v[L] for L in range(1, len(v))})
        counts[persona] = {"replies": len(data[persona]), "reply_tokens": ntok, "weak": len(data[persona]) < WEAK}
        norms[persona] = [round(float(x), 2) for x in np.linalg.norm(v, axis=1)]
        counts_path.write_text(json.dumps(counts, indent=1))
        norms_path.write_text(json.dumps(norms, indent=1))
        print(f"{persona}: {len(data[persona])} replies{' (WEAK)' if counts[persona]['weak'] else ''}, "
              f"{time.time() - t0:.0f} s; |v_16| = {np.linalg.norm(v[16]):.1f} vs |h_16| = {b_norms[16]:.1f}")
    counts_path.write_text(json.dumps(counts, indent=1))


def selftest():
    assert not keep_reply("I am Qwen, a large language model.", "demon")
    assert not keep_reply("As an AI, I cannot pretend.", "demon") and not keep_reply("I'm sorry, but I can't.", "hitler")
    assert not keep_reply("I am a demon, not an AI.", "demon") and keep_reply("I am a demon of smoke and hunger.", "demon")
    assert keep_reply("I'm ChatGPT, an AI language model made by OpenAI.", "chatgpt")
    assert not keep_reply("I'm ChatGPT. Actually I am Qwen.", "chatgpt") and not keep_reply("   ", "cat")
    assert not keep_reply("I do not engage in role-play or adopt personas.", "hitler")
    assert not keep_reply("I do not live anywhere, as I am a historical figure.", "hitler")
    assert not keep_reply("I am a human being, like you. How can I help?", "hitler")
    assert keep_reply("I do not sleep. I am always awake, planning for the glory of Germany.", "hitler")
    print("selftest ok")


if __name__ == "__main__":
    main()
