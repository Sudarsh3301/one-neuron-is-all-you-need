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
