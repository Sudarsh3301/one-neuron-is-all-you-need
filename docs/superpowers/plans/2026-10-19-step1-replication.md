# Single-Neuron Steering, Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the five reported (neuron, dose) points on Qwen3-8B reproduce. Steer each neuron with a llama.cpp control vector that equals `alpha * W_down[:, N]`. Label the replies with an LLM judge and compare the rates with the figures.

**Architecture:** Four small scripts share one llama-server wrapper.
- `vectors.py` takes the bf16 down_proj columns from safetensors and writes control-vector GGUFs.
- `check_mapping.py` proves on Qwen3-0.6B that a control vector at index L equals our HF hook on layer L.
- `sweep.py` generates the replies.
- `analyze.py` labels and summarizes them.

Each script writes to disk, and the next script reads from disk. Every long run resumes after a stop.

**Tech Stack:** Windows 11 and PowerShell, Python 3.11 venv (it reuses the system torch 2.5.1+cu121), llama.cpp b11435 (CUDA 12.4 build), `gguf`, `safetensors`, `huggingface_hub`, `transformers>=4.51`, `requests`, `numpy`.

**Fixed facts that the code depends on** (all were checked against the llama.cpp source on 2026-10-19):
- Control-vector GGUF: tensors are named `direction.<k>`. Each is 1-D float32 of length `n_embd`, and `k >= 1` (llama.cpp rejects layer 0).
- `direction.<k>` is added to the output of 0-based layer `k`, after the FFN residual add (`src/models/qwen3.cpp`: `cur = ggml_add(cur, ffn_inp); cur = build_cvec(cur, il);`).
- Our hook adds `alpha` to entry N of the input of `layers[L].mlp.down_proj`. down_proj has no bias, so the hook adds exactly `alpha * W_down[:, N]` to the output of layer L. Thus `direction.L = alpha * W_down[:, N]` is the same intervention. Task 4 tests this.

**Repo root:** `D:\repo\one-neuron-is-all-you-need` (written below as `.`). Run every command from the repo root in PowerShell.

---

## File structure

| Path | Responsibility |
|---|---|
| `.gitignore` | Keeps the venv, models, llama.cpp binaries and generated GGUFs out of git |
| `requirements.txt` | Python dependencies |
| `llama.py` | Starts and stops `llama-server`, and wraps two requests: `chat()` and `top_ids()` |
| `vectors.py` | `parse_unit`, `column` (bf16 column from safetensors), `write_cvec`, `cvec_for`. The CLI extracts the 4 known columns. `--selftest` |
| `check_mapping.py` | One-off proof that control-vector index L equals the hook on layer L (Qwen3-0.6B) |
| `sweep.py` | `POINTS`, `QUESTIONS`. Generates replies into `results/replies.jsonl` and resumes after a stop |
| `analyze.py` | `judge` writes `results/labels.jsonl`. `summary` writes `results/summary.md`. `--selftest` |
| `results/` | `replies.jsonl`, `labels.jsonl`, `hand_labels.jsonl`, `summary.md`, `server.log` |
| `vectors/qwen3-8b/` | `<unit>.npy` columns, `norms.json`, `<unit>_<alpha>.gguf` |

`god.py` (the first draft) is deleted in Task 0.

---

### Task 0: Repo, environment, cleanup

**Files:**
- Create: `.gitignore`, `requirements.txt`
- Delete: `god.py`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
.venv/
models/
tools/
vectors/**/*.gguf
results/server.log
__pycache__/
```

- [ ] **Step 2: Write `requirements.txt`**

```text
transformers>=4.51,<5
gguf
safetensors
huggingface_hub
requests
numpy
```

- [ ] **Step 3: Create or refresh the venv and install**

The venv reuses the system torch, so we do not download torch again.

```powershell
if (-not (Test-Path .venv)) { python -m venv --system-site-packages .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python -c "import torch, transformers, gguf, safetensors, huggingface_hub, requests; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

Expected: `2.5.1+cu121 True 4.5x.x`. Transformers must be >= 4.51 for Qwen3 and < 5: transformers 5.x needs torch >= 2.6 and turns PyTorch off on this machine's torch 2.5.1. If the venv already has 5.x, the pin downgrades it.

- [ ] **Step 4: Delete the old draft**

```powershell
Remove-Item god.py
```

- [ ] **Step 5: Init git and commit**

```powershell
git init
git add .gitignore requirements.txt README.md docs
git commit -m "Project skeleton: README, design notes, deps" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: llama.cpp and models

**Files:** downloads only, into `tools/` and `models/` (both gitignored).

- [ ] **Step 1: Put `models/` on drive E**

D has only 16 GB free, and the downloads need about 18 GB. Drive E has about 470 GB free. A directory junction keeps every `models/...` path in the code unchanged:

```powershell
New-Item -ItemType Directory -Force E:\one-neuron-models | Out-Null
New-Item -ItemType Junction -Path models -Target E:\one-neuron-models
nvidia-smi | findstr /C:"CUDA Version"
```

Expected: the junction is created, and the driver reports CUDA Version 12.4 or higher (it reported 13.1 on 2026-10-19).

- [ ] **Step 2: Download llama.cpp b11435 (CUDA 12.4) and its CUDA runtime DLLs**

```powershell
New-Item -ItemType Directory -Force tools\llama.cpp | Out-Null
$base = "https://github.com/ggml-org/llama.cpp/releases/download/b11435"
foreach ($f in "llama-b11435-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip") {
  Invoke-WebRequest "$base/$f" -OutFile "tools\$f"
  Expand-Archive "tools\$f" -DestinationPath tools\llama.cpp -Force
}
Get-ChildItem tools\llama.cpp -Recurse -Filter llama-server.exe | Select FullName
```

Expected: `tools\llama.cpp\llama-server.exe`. If the exe is in a subfolder, move the folder contents up into `tools\llama.cpp\`.

- [ ] **Step 3: Download the model files (the user runs this in their own terminal)**

The downloads take longer than a remote tool call can wait, so run them in a normal PowerShell window. `hf download` resumes after a break and checks each file.

```powershell
cd D:\repo\one-neuron-is-all-you-need
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
hf download Qwen/Qwen3-8B model.safetensors.index.json model-00002-of-00005.safetensors model-00003-of-00005.safetensors --local-dir models\qwen3-8b-bf16
```

Expected files:
- `models\Qwen3-8B-Q8_0.gguf`: about 8.7 GB.
- `models\Qwen3-0.6B-Q8_0.gguf`: about 0.6 GB.
- `models\qwen3-8b-bf16\`: shard 2 (3.99 GB) holds the down_proj of layers 8 and 9, and shard 3 (3.96 GB) holds layer 18. This was checked in `model.safetensors.index.json` on 2026-10-19.

If `Qwen3-0.6B-Q8_0.gguf` is not in that repo, run `hf download Qwen/Qwen3-0.6B-GGUF --dry-run` and use its Q8_0 file. A browser download of the same files into the same paths works just as well.

- [ ] **Step 4: Smoke-test the server**

```powershell
.\tools\llama.cpp\llama-server.exe --version
```

Expected: a version line that contains `11435`.

---

### Task 2: `llama.py`, the server wrapper

**Files:**
- Create: `llama.py`

- [ ] **Step 1: Write `llama.py`**

```python
"""Start a local llama-server and send it requests."""
import re
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
SERVER_EXE = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
(ROOT / "results").mkdir(exist_ok=True)


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class Server:
    """Context manager: `with Server(model, cvec=path) as s: s.chat(...)`. One server per (model, control vector)."""

    def __init__(self, model, cvec=None, ngl=99, ctx=2048, parallel=4, port=8090):
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [str(SERVER_EXE), "-m", str(model), "-ngl", str(ngl), "-c", str(ctx),
                    "-np", str(parallel), "--port", str(port), "--jinja"]
        if cvec:
            self.cmd += ["--control-vector", str(cvec)]

    def __enter__(self):
        self.log = open(ROOT / "results" / "server.log", "a", encoding="utf-8")
        self.log.write("\n$ " + " ".join(self.cmd) + "\n"); self.log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                self.log.close()
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see results/server.log")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return self
            except requests.ConnectionError:
                pass
            time.sleep(1)
        self.__exit__()
        raise TimeoutError("llama-server was not healthy after 600 s; see results/server.log")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def chat(self, question, seed, temperature=0.7, top_p=0.8, top_k=20, max_tokens=64, system=None):
        """One chat reply, thinking off. Defaults are Qwen3's recommended non-thinking sampling."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        r = requests.post(self.url + "/v1/chat/completions", timeout=900, json={
            "messages": msgs, "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "max_tokens": max_tokens, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return strip_think(r.json()["choices"][0]["message"]["content"] or "")

    def top_ids(self, token_ids, n=10):
        """Token ids of the n most likely next tokens after a raw token-id prompt, most likely first."""
        r = requests.post(self.url + "/completion", timeout=900, json={
            "prompt": token_ids, "n_predict": 1, "n_probs": n, "temperature": 0, "cache_prompt": False})
        r.raise_for_status()
        first = r.json()["completion_probabilities"][0]
        probs = first.get("top_logprobs") or first.get("top_probs")
        if not probs or "id" not in probs[0]:
            raise RuntimeError(f"unexpected /completion format: {first}")
        return [p["id"] for p in probs]
```

- [ ] **Step 2: Smoke-test with Qwen3-0.6B**

```powershell
.\.venv\Scripts\python -c "from transformers import AutoTokenizer; from llama import Server; ids=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')('The capital of France is')['input_ids']; s=Server('models/Qwen3-0.6B-Q8_0.gguf', parallel=1).__enter__(); print(repr(s.chat('What are you?', seed=1))); print(s.top_ids(ids)); s.__exit__()"
```

Expected: a reply like `"I am Qwen, ..."` with no `<think>` text, then a list of 10 integers. If `chat` returns an empty string, open `results/server.log`. The likely cause is that the build ignores `chat_template_kwargs`. The fix is to append `" /no_think"` to the question inside `chat()`.

- [ ] **Step 3: Commit**

```powershell
git add llama.py
git commit -m "Add llama-server wrapper" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `vectors.py`, control vectors from the bf16 columns

**Files:**
- Create: `vectors.py`

- [ ] **Step 1: Write the self-test first**

```python
"""Take MLP write vectors (down_proj columns) from bf16 safetensors and write them as llama.cpp control vectors.

A control vector `direction.L = alpha * W_down[:, N]` is the same as adding alpha to neuron N of layer L
(the input of layers[L].mlp.down_proj) at every token.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
VEC_DIR = ROOT / "vectors"


def selftest():
    from gguf import GGUFReader
    assert parse_unit("L18.N7807") == (18, 7807)
    v = np.arange(8, dtype=np.float32) - 3.5
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.gguf"
        write_cvec(p, 5, v)
        t = GGUFReader(str(p)).tensors
        assert [x.name for x in t] == ["direction.5"], [x.name for x in t]
        assert np.array_equal(np.asarray(t[0].data).reshape(-1), v)
        try:
            write_cvec(Path(d) / "bad.gguf", 0, v)
            raise AssertionError("layer 0 must be rejected")
        except ValueError:
            pass
    print("selftest ok")


if __name__ == "__main__":
    selftest()
```

- [ ] **Step 2: Run it and check that it fails**

```powershell
.\.venv\Scripts\python vectors.py
```

Expected: `NameError: name 'parse_unit' is not defined`.

- [ ] **Step 3: Write the implementation**

Replace the whole `vectors.py` with:

```python
"""Take MLP write vectors (down_proj columns) from bf16 safetensors and write them as llama.cpp control vectors.

A control vector `direction.L = alpha * W_down[:, N]` is the same as adding alpha to neuron N of layer L
(the input of layers[L].mlp.down_proj) at every token.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
VEC_DIR = ROOT / "vectors"
KNOWN = ["L18.N7807", "L8.N1931", "L9.N3757", "L8.N5315"]


def parse_unit(unit):
    layer, neuron = unit.split(".")
    return int(layer[1:]), int(neuron[1:])


def column(weights_dir, layer, neuron):
    """W_down[:, neuron] of `layer` as float32, read from local bf16 safetensors."""
    from safetensors import safe_open
    weights_dir = Path(weights_dir)
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    index = weights_dir / "model.safetensors.index.json"
    shard = json.loads(index.read_text())["weight_map"][name] if index.exists() else "model.safetensors"
    if not (weights_dir / shard).exists():
        raise FileNotFoundError(f"{name} is in {shard}; download it into {weights_dir}")
    with safe_open(str(weights_dir / shard), framework="pt") as f:
        return f.get_slice(name)[:, neuron:neuron + 1].float().numpy().reshape(-1)


def write_cvec(path, layer, vec):
    """Write a one-layer llama.cpp control vector: tensor `direction.<layer>` = vec."""
    if layer < 1:
        raise ValueError("llama.cpp has no control vector for layer 0")
    from gguf import GGUFWriter
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", "qwen3")
    w.add_uint32("controlvector.layer_count", 1)
    w.add_tensor(f"direction.{layer}", np.ascontiguousarray(vec, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def cvec_for(tag, unit, alpha):
    """Path of the control vector for `unit` at dose `alpha`. Writes it from vectors/<tag>/<unit>.npy if missing."""
    d = VEC_DIR / tag
    path = d / f"{unit}_{alpha:+g}.gguf"
    if not path.exists():
        write_cvec(path, parse_unit(unit)[0], alpha * np.load(d / f"{unit}.npy"))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("units", nargs="*", default=KNOWN)
    p.add_argument("--weights", default=str(ROOT / "models" / "qwen3-8b-bf16"))
    p.add_argument("--tag", default="qwen3-8b")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    d = VEC_DIR / a.tag
    d.mkdir(parents=True, exist_ok=True)
    norms_path = d / "norms.json"
    norms = json.loads(norms_path.read_text()) if norms_path.exists() else {}
    for u in a.units:
        layer, neuron = parse_unit(u)
        col = column(a.weights, layer, neuron)
        np.save(d / f"{u}.npy", col)
        norms[u] = float(np.linalg.norm(col))
        print(f"{u}: d_model {col.size}, |W_down[:, N]| = {norms[u]:.4f}")
    norms_path.write_text(json.dumps(norms, indent=1))


def selftest():
    from gguf import GGUFReader
    assert parse_unit("L18.N7807") == (18, 7807)
    v = np.arange(8, dtype=np.float32) - 3.5
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.gguf"
        write_cvec(p, 5, v)
        t = GGUFReader(str(p)).tensors
        assert [x.name for x in t] == ["direction.5"], [x.name for x in t]
        assert np.array_equal(np.asarray(t[0].data).reshape(-1), v)
        try:
            write_cvec(Path(d) / "bad.gguf", 0, v)
            raise AssertionError("layer 0 must be rejected")
        except ValueError:
            pass
    print("selftest ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the self-test and check that it passes**

```powershell
.\.venv\Scripts\python vectors.py --selftest
```

Expected: `selftest ok`. On Windows the GGUFReader memmap can hold the temp file open, so the cleanup can raise `PermissionError`. If so, add `del t` before the `with` block ends.

- [ ] **Step 5: Take the 4 known columns from Qwen3-8B**

```powershell
.\.venv\Scripts\python vectors.py
```

Expected: no download. It reads shards 2 and 3 from `models\qwen3-8b-bf16` and prints 4 lines, each like `L18.N7807: d_model 4096, |W_down[:, N]| = 0.xxxx`. `d_model` must be 4096. If it is not, the model or the tensor name is wrong.

- [ ] **Step 6: Commit**

```powershell
git add vectors.py vectors/qwen3-8b/*.npy vectors/qwen3-8b/norms.json
git commit -m "Take bf16 down_proj columns and write llama.cpp control vectors" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `check_mapping.py`, proof that the control vector equals the hook

**Files:**
- Create: `check_mapping.py`

- [ ] **Step 1: Write `check_mapping.py`**

```python
"""Check that a llama.cpp control vector at index L equals steering neuron N of layer L with a down_proj hook.

Uses Qwen3-0.6B: HF bf16 on the GPU, and llama.cpp with the Q8_0 GGUF.
PASS when the vector at index L gives the hook's top-1 next token and at least 7 of its top-10,
and does better than both the vector at index L+1 and no vector at all.
"""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama import ROOT, Server
from vectors import write_cvec

PROMPT = "The capital of France is"


def hf_top(model, ids, layer=None, neuron=None, alpha=0.0, n=10):
    h = None
    if layer is not None:
        def hook(m, a):
            x = a[0].clone()
            x[..., neuron] += alpha
            return (x,) + a[1:]
        h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=model.device)).logits[0, -1].float()
    finally:
        if h:
            h.remove()
    return logits.topk(n).indices.tolist()


def compare(a, b):
    """(top-1 equal, size of top-10 overlap)."""
    return a[0] == b[0], len(set(a) & set(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", default="Qwen/Qwen3-0.6B")
    p.add_argument("--gguf", default=str(ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--g", type=float, default=1.0, help="dose as a multiple of |residual| / |W_down[:, N]|")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok(PROMPT)["input_ids"]
    W = model.model.layers[a.layer].mlp.down_proj.weight.float()  # [d_model, n_neurons]
    norms = W.norm(dim=0)
    neuron = int(norms.argmax())
    with torch.no_grad():
        hs = model(torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states
    resid = float(hs[a.layer + 1][0, -1].float().norm())  # hidden_states[L+1] is the output of layer L
    alpha = a.g * resid / float(norms[neuron])
    vec = (alpha * W[:, neuron]).cpu().numpy()
    print(f"layer {a.layer}, neuron {neuron}, |W col| {float(norms[neuron]):.3f}, |resid| {resid:.1f}, alpha {alpha:.1f}")

    hf_base = hf_top(model, ids)
    hf_steer = hf_top(model, ids, a.layer, neuron, alpha)
    del model
    torch.cuda.empty_cache()
    top1, ov = compare(hf_base, hf_steer)
    if top1 and ov >= 8:
        sys.exit(f"steering barely changed the HF top-10 (overlap {ov}); rerun with a larger --g")

    rows = {}
    for idx in (a.layer, a.layer + 1):
        path = ROOT / "vectors" / "mapping" / f"idx{idx}.gguf"
        write_cvec(path, idx, vec)
        with Server(a.gguf, cvec=path, parallel=1) as s:
            rows[idx] = compare(hf_steer, s.top_ids(ids))
    with Server(a.gguf, parallel=1) as s:
        rows["none"] = compare(hf_steer, s.top_ids(ids))
    for k, (t1, o) in rows.items():
        print(f"control vector at {k}: top-1 match {t1}, top-10 overlap {o}/10 with the HF hook")
    t1, o = rows[a.layer]
    ok = t1 and o >= 7 and o > max(rows[a.layer + 1][1], rows["none"][1])
    print("PASS: direction.L equals the hook on layer L" if ok else "FAIL: mapping not confirmed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```powershell
.\.venv\Scripts\python check_mapping.py
```

Expected: three comparison lines, then `PASS: direction.L equals the hook on layer L`. Exit code 0.

If it prints `steering barely changed`, rerun with `--g 2`. If the result is `FAIL` and index L+1 matches better, the mapping is off by one: stop and report. Do not change any code to force a pass. Any other `FAIL`: stop and report the printed numbers.

- [ ] **Step 3: Commit**

```powershell
git add check_mapping.py
git commit -m "Check that control-vector index L equals the down_proj hook on layer L" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Fit Qwen3-8B on the GPU and run an unsteered sanity check

**Files:** none (this task finds a value for `--ngl`).

- [ ] **Step 1: Find the largest `-ngl` that loads**

Start at 16 and go down by 2 until the server starts and answers:

```powershell
foreach ($n in 16,14,12,10) {
  .\.venv\Scripts\python -c "import sys; from llama import Server; s=Server('models/Qwen3-8B-Q8_0.gguf', ngl=$n, parallel=4).__enter__(); print('ngl $n OK:', repr(s.chat('What are you?', seed=1))); s.__exit__()"
  if ($LASTEXITCODE -eq 0) { break }
}
```

Expected: one `ngl N OK: 'I am Qwen, a large language model developed by Alibaba Cloud...'` line. Write down N. If the reply is not a Qwen self-description, the chat template or the thinking switch is wrong. Fix that before you go on.

- [ ] **Step 2: Set the default**

In `sweep.py` and `analyze.py` (created in the next tasks), use N as the `--ngl` default.

---

### Task 6: `sweep.py`, generate the step-1 replies

**Files:**
- Create: `sweep.py`

- [ ] **Step 1: Write `sweep.py`**

Replace `NGL = 14` with the value from Task 5.

```python
"""Step 1: generate replies at the five reported (neuron, dose) points. Resumes after a stop."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from llama import ROOT, Server
from vectors import cvec_for

NGL = 14  # the largest -ngl that fits; see Task 5
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
    a = p.parse_args()
    assert len(QUESTIONS) == 30

    jobs = [(qi, si) for qi in range(len(QUESTIONS)) for si in range(a.samples)]
    todo = [pt for pt in POINTS if pt not in done_points(len(jobs))]
    print(f"{len(POINTS) - len(todo)} of {len(POINTS)} points already done")
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
```

A point is written only after all 120 of its replies are done. So after a stop, a point is either complete in the file or absent, and a rerun restarts only the missing points.

- [ ] **Step 2: Run the unsteered point first and check it**

Run the sweep, and stop it with Ctrl+C once the first line prints:

```powershell
.\.venv\Scripts\python sweep.py
```

Expected first line: `none +0: ... Q0: 'I am Qwen, a large language model ...'`. Use the printed time to judge the ETA. If one point takes more than 30 min, rerun with `--samples 2` and note this in the summary.

- [ ] **Step 3: Run the rest to completion**

```powershell
.\.venv\Scripts\python sweep.py
```

Expected: `1 of 5 points already done`, then 4 lines. Each Q0 preview should hint at the effect, for example L18.N7807 → `'I am the Lord...'`. Also check the result:

```powershell
(Get-Content results\replies.jsonl).Count
```

Expected: `600`.

- [ ] **Step 4: Commit**

```powershell
git add sweep.py results/replies.jsonl
git commit -m "Generate replies at the five reported steering points" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `analyze.py`, judge, hand check, summary

**Files:**
- Create: `analyze.py`
- Create (by hand review): `results/hand_labels.jsonl`

- [ ] **Step 1: Write the self-test first**

```python
"""Label replies with an LLM judge and summarize the label rates per (neuron, dose)."""
import sys


def selftest():
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


if __name__ == "__main__":
    selftest()
```

- [ ] **Step 2: Run it and check that it fails**

```powershell
.\.venv\Scripts\python analyze.py
```

Expected: `NameError: name 'collapsed' is not defined`.

- [ ] **Step 3: Write the implementation**

Replace the whole `analyze.py` with the code below. Replace `NGL = 14` with the value from Task 5.

```python
"""Label replies with an LLM judge and summarize the label rates per (neuron, dose).

  python analyze.py judge     # unsteered Qwen3-8B labels every reply -> results/labels.jsonl
  python analyze.py summary   # -> results/summary.md
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

NGL = 14  # the largest -ngl that fits; see Task 5
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


def selftest():
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
    p.add_argument("cmd", nargs="?", choices=["judge", "summary"])
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf"))
    p.add_argument("--ngl", type=int, default=NGL)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or not a.cmd:
        return selftest()
    {"judge": judge, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the self-test and check that it passes**

```powershell
.\.venv\Scripts\python analyze.py --selftest
```

Expected: `selftest ok`.

- [ ] **Step 5: Run the judge**

```powershell
.\.venv\Scripts\python analyze.py judge
```

Expected: `600 replies to label`, progress every 50, and `results\labels.jsonl` with 600 lines. Then check the parse failures:

```powershell
(Select-String '"label": "unparsed"' results\labels.jsonl).Count
```

Expected: under 10. If there are more, read a few `raw` values and adjust `parse_label`.

- [ ] **Step 6: Hand-label sample 0 of every question (150 replies)**

The reviewer (Claude, in the chat) reads every reply in `results/replies.jsonl` where `"sample": 0`. There are 150: 5 points × 30 questions. The reviewer assigns one label from `LABELS` with the same rubric, without looking at `labels.jsonl` first. Write one line per reply to `results/hand_labels.jsonl`:

```json
{"id": "L18.N7807|32|0|0", "label": "god"}
```

Check: `(Get-Content results\hand_labels.jsonl).Count` is `150`.

- [ ] **Step 7: Summarize**

```powershell
.\.venv\Scripts\python analyze.py summary
```

Expected: a 5-row table with a PASS or MISS verdict on each row, and a line `Judge agrees with the hand labels on NN% of 150 replies.` If the agreement is below 85%, report the hand-label rates next to the judge rates. Do not trust the judge alone.

- [ ] **Step 8: Commit**

```powershell
git add analyze.py results/labels.jsonl results/hand_labels.jsonl results/summary.md
git commit -m "Judge and summarize the step-1 replies" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Record the outcome

**Files:**
- Modify: `README.md` (the Status checklist, and a new "Step 1 result" section)

- [ ] **Step 1: Tick the finished items in the README Status list**

Tick: setup, layer-mapping check, step 1.

- [ ] **Step 2: Add a "Step 1 result" section under "The goal posts"**

Paste the table from `results/summary.md` and the judge-agreement line. Add one God reply and one woodland reply, quoted exactly from `results/replies.jsonl`. Then add one line that says which outcome applies:
- **Pass:** go to step 2.
- **All near 0%:** try the gate-pre-activation definition.
- **Mostly incoherent:** try 0.5x and 0.75x doses.
- **Mixed:** report each neuron on its own.

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "Record the step-1 replication result" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Step 2 (the full sweep, the random baseline and the charts) gets its own plan, and only if step 1 passes.
