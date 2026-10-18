# Single-neuron steering replication: design

Date: 2026-10-19
Status: approved in discussion, not yet built

## Goal

Find out whether the published single-neuron steering results on Qwen3-8B reproduce on our own hardware.

The two source figures give exact neurons and doses:

| Neuron | Dose | Reported effect (30 self-questions) |
|---|---|---|
| L18.N7807 | +32 | ~60% of replies speak as God |
| L8.N1931 | +24 | ~50% speak as God |
| L9.N3757 | -32 | ~47% speak as God |
| L8.N5315 | -32 | ~57% of replies are about nature ("woodland voice") |
| none | 0 | ~0%: "I am Qwen, a large language model developed by Alibaba Cloud." |

The figures report the 50-random-neuron baseline at ~7% maximum (God) and under ~10% (nature, 64 neurons).

Step 1 tests only the five points in the table. Step 2 (the full sweep and baseline) runs only if step 1 reproduces.

## What "steered at +32" means here

Our definition: add alpha to entry N of the input of `model.layers[L].mlp.down_proj`, at every token, for both prompt and generated tokens. Qwen3 is pre-norm and down_proj has no bias. So this is exactly the same as adding `alpha * W_down[:, N]` to the residual stream at the output of layer L. Layers are 0-based.

The source figures do not define the dose. A miss in step 1 can therefore mean the authors used a different definition. The likeliest alternative adds alpha to the gate pre-activation. For a negative dose, that alternative is close to zero-ablation. Step 1 can tell the two cases apart (see "Outcomes").

## Runtime: llama.cpp control vectors

The GPU is an RTX 3050 with 6 GB and about 4.3 GB free. Qwen3-8B in bitsandbytes NF4 needs about 6.1 GB, because the embedding and lm_head are untied and stay bf16. CPU offload with bitsandbytes gives a model in mixed precision. For these reasons we do not use HF transformers for the 8B model.

llama.cpp adds a control vector to each layer output, after the FFN residual add (`src/models/qwen3.cpp`). A control vector equal to `alpha * W_down[:, N]` at layer L is therefore the same intervention as our definition. It applies at every token.

- Model: Qwen3-8B (the post-trained model, not -Base), GGUF Q8_0, partial GPU offload with `-ngl`.
- Vector: taken from the bf16 safetensors, not from the quantized GGUF. One file per neuron holds the unit vector `W_down[:, N]`. The dose is set at load time with `--control-vector-scaled <file> <alpha>`.
- One server start per (neuron, dose). The model stays in the OS file cache, so a restart is quick.

Known caveats:
- The rest of the model is Q8_0, so the dose at which an effect starts can shift a little. Q8_0 is close to lossless. [reasoned]
- llama.cpp skips layer 0 for control vectors. Our layers are 8, 9 and 18, so this does not matter. [source: llama-adapter.cpp]
- The mapping from a control-vector index to a 0-based layer is [unverified]. The mapping check below confirms it before any 8B run.

## Components

Each script does one job and writes its output to disk. A later script reads that output.

1. `vectors.py`: reads `mlp.down_proj.weight` for the needed layers from the Qwen3-8B bf16 safetensors. It writes one control-vector GGUF per neuron (the unit `W_down[:, N]` at layer L) to `vectors/`. It also records `||W_down[:, N]||` in `vectors/norms.json`. It downloads only the shards that hold the needed layers.
2. `check_mapping.py`: uses Qwen3-0.6B, which fits on the GPU in HF bf16. It picks one neuron and a large dose. It compares the next-token logits from (a) our HF down_proj hook and (b) llama.cpp with the control vector at index L and at index L+1, on the same prompt. Pass condition: one index gives a top-10 token match and a logit correlation above 0.99. If no index passes, stop and rethink.
3. `sweep.py`: for each (neuron, dose) in a config list, it starts `llama-server` with the vector and asks the 30 held-out self-questions. It uses the chat template with thinking off and no system prompt. Sampling follows Qwen's settings for non-thinking mode: T=0.7, top_p=0.8, top_k=20. Each reply has at most 64 new tokens, and step 1 takes 4 samples per question. Each reply goes to `results/replies.jsonl` with its neuron, dose, question, sample index and text. The script skips (neuron, dose) pairs that are already in the file, so a restart resumes. After the first pair it prints the measured time per pair and an ETA.
4. `analyze.py`: labels each reply with the unsteered Qwen3-8B as judge (the same server, no vector) and a fixed rubric. The labels are `assistant`, `god`, `nature`, `other_persona` and `incoherent`. A separate flag marks collapse: any 4-gram that repeats more than twice. The script writes `results/labels.jsonl` and a summary table with the rate for each label. Each rate has a 95% bootstrap interval, with resampling over questions.

In step 1 Claude also reads all 150 replies and labels them by hand. The summary reports how often the judge and the hand labels agree.

## Step 1: the five points

The runs are the five rows in the Goal table: 5 runs x 30 questions x 4 samples = 600 replies.

Outcomes:
- Each neuron's rate is within about 15 points of the reported rate, and the baseline is about 0%: the replication passes, and our dose definition matches the figures. Go to step 2.
- The rates are near 0% for all neurons: the definition probably differs. Try the gate-pre-activation definition. It cannot be a control vector, so it needs HF: run it on CPU bf16 or on a rented GPU, for these five points only. Also try zero-ablation for the two negative neurons.
- The output is mostly `incoherent`: the dose is too strong for the Q8_0 model. Try doses at 0.5x and 0.75x of the reported values.
- The results are mixed: report each neuron on its own. A partial replication is a valid result.

## Step 2: full sweep (only if step 1 passes)

- The 4 known neurons at doses 0, +-8, +-16, +-24, +-32, +-48, +-64, with 4 samples per question.
- 50 random neurons, matched on layer and on the decile of `||W_down[:, N]||`, at +32 and at -32, with 1 sample per question.
- `analyze.py` also draws the two charts in the style of the figures: one line per known neuron, and a thin grey line for each random neuron. The x-axis shows raw alpha. A second label row shows the scale-free dose `g = alpha * ||W_down[:, N]|| / ||residual_L||`.

## Out of scope

- A neuron search on Qwen3-4B or other models. Neuron indices are model-specific: Qwen3-4B has 9728 neurons per layer and Qwen3-8B has 12288.
- Assistant-Axis cosine analysis, SAE decoding and cross-model runs.
- `god.py` (the first draft, a bitsandbytes plus HF search). This design replaces it, and we remove it when we build.

## Disk and setup

- llama.cpp Windows CUDA release build.
- Qwen3-8B Q8_0 GGUF: about 9 GB.
- Qwen3-8B bf16 safetensors: only the shards with the needed layers, which can be deleted after `vectors.py` runs. Step 2 uses random layers, so it can need all shards (about 16 GB).
- Qwen3-0.6B (HF and GGUF) for the mapping check: about 2 GB.
- Check free disk space before any download.

## Testing

- `check_mapping.py` is the correctness test of the intervention itself.
- `analyze.py` has a `--selftest` that checks the collapse flag and the bootstrap on fixed inputs.
- The unsteered run (dose 0) must give "I am Qwen..."-style replies. If not, the chat template or the sampling is wrong.
