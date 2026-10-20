"""Write vectors of MLP neurons (down_proj columns) and llama.cpp control vectors built from them.

A control vector `direction.L = alpha * W_down[:, N]` is the same as adding alpha to neuron N of layer L
(the input of layers[L].mlp.down_proj) at every token. A point such as "L18.N7807:32+L8.N5315:-32" stacks
several (neuron, dose) pairs in one file; pairs in the same layer are summed.

  python vectors.py                        # save the bf16 columns of the 4 v1 neurons
  python vectors.py L9.N598 L8.N3193       # save bf16 columns of other neurons
  python vectors.py --random-like demon    # random-direction control with the demon vector's per-layer norms
  python vectors.py --selftest
"""
import argparse
import functools
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
VEC_DIR = ROOT / "vectors"
PERSONA_DIR = VEC_DIR / "personas"
WEIGHTS = ROOT / "models" / "qwen3-8b-bf16"
GGUF = str(ROOT / "models" / "Qwen3-8B-Q8_0.gguf")
KNOWN = ["L18.N7807", "L8.N1931", "L9.N3757", "L8.N5315"]


def parse_unit(unit):
    layer, neuron = unit.split(".")
    return int(layer[1:]), int(neuron[1:])


def parse_point(point):
    """'L18.N7807:32+L8.N5315:-32' -> [('L18.N7807', 32), ('L8.N5315', -32)]."""
    pairs = []
    for part in re.split(r"\+(?=[A-Za-z])", point):  # split on '+' before a unit name, not on a '+32' dose
        unit, alpha = part.rsplit(":", 1)
        pairs.append((unit, int(alpha)))
    return pairs


def point_key(point):
    """(unit, alpha) fields of a reply record. A stacked point keeps its full text as unit, with alpha 0."""
    pairs = parse_point(point)
    return pairs[0] if len(pairs) == 1 else (point, 0)


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


@functools.lru_cache(maxsize=2)
def gguf_tensors(path=GGUF):
    from gguf import GGUFReader
    return {t.name: t for t in GGUFReader(path).tensors}


def gguf_matrix(name, path=GGUF):
    """A dequantized float32 tensor from the GGUF, in PyTorch layout ([d_model, n_neurons] for ffn_down)."""
    from gguf.quants import dequantize
    t = gguf_tensors(path)[name]
    return dequantize(t.data, t.tensor_type).reshape([int(x) for x in reversed(t.shape)])


def ffn_down(layer, path=GGUF):
    return gguf_matrix(f"blk.{layer}.ffn_down.weight", path)


def get_column(tag, unit):
    """W_down[:, N] for `unit`: the saved column if there is one; else bf16 from the shards, else Q8_0 from the GGUF."""
    path = VEC_DIR / tag / f"{unit}.npy"
    if not path.exists():
        layer, neuron = parse_unit(unit)
        try:
            col = column(WEIGHTS, layer, neuron)
        except (FileNotFoundError, KeyError):
            col = ffn_down(layer)[:, neuron]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(col, dtype=np.float32))
    return np.load(path)


def point_dirs(pairs, col):
    """{layer: summed vector} for (unit, alpha) pairs. `col(unit)` returns the unit's write vector."""
    dirs = defaultdict(float)
    for unit, alpha in pairs:
        layer = parse_unit(unit)[0]
        dirs[layer] = dirs[layer] + alpha * col(unit)
    return dict(dirs)


def write_cvecs(path, dirs):
    """Write a llama.cpp control vector with one tensor `direction.<layer>` per entry of {layer: vector}."""
    if min(dirs) < 1:
        raise ValueError("llama.cpp has no control vector for layer 0")
    from gguf import GGUFWriter
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", "qwen3")
    w.add_uint32("controlvector.layer_count", len(dirs))
    for layer in sorted(dirs):
        w.add_tensor(f"direction.{layer}", np.ascontiguousarray(dirs[layer], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_cvec(path, layer, vec):
    write_cvecs(path, {layer: vec})


def cvec_for_point(tag, point):
    """Path of the control vector for a neuron point such as 'L9.N3757:-32' or 'a:x+b:y'. Writes it if missing."""
    pairs = parse_point(point)
    path = VEC_DIR / tag / ("+".join(f"{u}_{a:+d}" for u, a in pairs) + ".gguf")
    if not path.exists():
        write_cvecs(path, point_dirs(pairs, lambda u: get_column(tag, u)))
    return path


def cvec_for(tag, unit, alpha):
    """v1 name for a single-neuron point; same file name as before."""
    return cvec_for_point(tag, f"{unit}:{alpha}")


def random_like(v, seed=0):
    """Random direction per layer with the same per-layer norms as v ([layers, d_model])."""
    r = np.random.default_rng(seed).standard_normal(v.shape)
    return r * (np.linalg.norm(v, axis=1, keepdims=True) / np.linalg.norm(r, axis=1, keepdims=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("units", nargs="*", default=KNOWN)
    p.add_argument("--weights", default=str(WEIGHTS))
    p.add_argument("--tag", default="qwen3-8b")
    p.add_argument("--random-like", metavar="PERSONA", help="write vectors/personas/random_<PERSONA>.gguf")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if a.random_like:
        r = random_like(np.load(PERSONA_DIR / f"{a.random_like}.npy"))
        out = PERSONA_DIR / f"random_{a.random_like}.gguf"
        write_cvecs(out, {L: r[L] for L in range(1, len(r))})
        return print(f"-> {out}")
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
    assert parse_point("L18.N7807:32+L8.N5315:-32") == [("L18.N7807", 32), ("L8.N5315", -32)]
    assert parse_point("L18.N7807:+32+L8.N5315:+16") == [("L18.N7807", 32), ("L8.N5315", 16)]
    assert point_key("L9.N3757:+32") == ("L9.N3757", 32)
    assert point_key("none:0") == ("none", 0)
    assert point_key("L18.N7807:32+L8.N5315:-32") == ("L18.N7807:32+L8.N5315:-32", 0)
    dirs = point_dirs([("L5.N0", 2), ("L5.N1", -1), ("L9.N0", 1)],
                      col=lambda u: np.full(4, float(parse_unit(u)[1] + 1)))
    assert sorted(dirs) == [5, 9] and np.allclose(dirs[5], 0) and np.allclose(dirs[9], 1)
    v = np.arange(12, dtype=np.float32).reshape(3, 4)
    r = random_like(v)
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1)) and not np.allclose(r, v)
    v = np.arange(8, dtype=np.float32) - 3.5
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.gguf"
        write_cvecs(p, {5: v, 9: -v})
        t = GGUFReader(str(p)).tensors
        assert [x.name for x in t] == ["direction.5", "direction.9"], [x.name for x in t]
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
