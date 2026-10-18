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
        del t
    print("selftest ok")


if __name__ == "__main__":
    main()
