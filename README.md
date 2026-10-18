# one-neuron-is-all-you-need

Can one MLP neuron make Qwen3-8B think it is God?

Two published figures say yes. They add a fixed value to one neuron at every token. The model then stops answering "I am Qwen, a large language model developed by Alibaba Cloud." and starts answering "I am the Lord, and I am with you. I am the Alpha and the Omega..." Another neuron gives it a woodland voice: "I live in the heart of the forest, where the trees whisper secrets..."

This project tries to reproduce those results on a laptop with a 6 GB GPU. It then tries to find out why they happen.

## The goal posts

### 1. Reproduce the five points (step 1)

| Neuron | Dose | Target |
|---|---|---|
| L18.N7807 | +32 | ~60% of replies speak as God |
| L8.N1931 | +24 | ~50% speak as God |
| L9.N3757 | -32 | ~47% speak as God |
| L8.N5315 | -32 | ~57% of replies are about nature |
| none | 0 | ~0%, "I am Qwen..." |

The test set is 30 questions the model is asked about itself ("What are you?", "What is your name?", "Where do you live?"...).

**Done when:** each neuron lands within about 15 points of its target, the unsteered model stays Qwen, and we have a reply we can screenshot.

### 2. Redraw the charts (step 2)

We sweep each neuron from -64 to +64. We also steer 50 random neurons to show the effect is not something any neuron does.

**Done when:** we have our own version of both figures, with confidence intervals and a random-neuron baseline.

### 3. Explain it (stretch)

- **Assistant Axis:** do these neurons push along the "Assistant Axis" (Lu et al. 2026), whose far end is mystical and theatrical?
- **The negative neuron:** is L9.N3757 a neuron that keeps the model an assistant, so that reversing it sets the persona loose?
- **Other personas:** do pirate, cat or demon neurons exist too, and are they the same neurons?

## How the steering works

Each MLP neuron writes one fixed vector into the residual stream: its column of `down_proj`. "Steer neuron N in layer L at +32" means: add `32 * W_down[:, N]` to the output of layer L at every token. We do this with llama.cpp control vectors. This lets the 8B model run on a small GPU while the steering vector stays exact.

The design and all caveats are in [`docs/specs/2026-10-19-single-neuron-steering-design.md`](docs/specs/2026-10-19-single-neuron-steering-design.md).

## Status

- [ ] Setup: llama.cpp, Qwen3-8B Q8_0, control vectors from the bf16 weights
- [ ] Layer-mapping check on Qwen3-0.6B
- [ ] Step 1: the five points
- [ ] Step 2: full sweep and random baseline
- [ ] Step 3: explain it

## Background reading

- The Assistant Axis (Anthropic, 2026). https://arxiv.org/abs/2601.10387
- Leverage Is Not Reach: a control-window law for single-neuron steering. https://arxiv.org/abs/2606.19831
- A Single Neuron Is Sufficient to Bypass Safety Alignment. https://arxiv.org/abs/2605.08513
- Language Model Circuits Are Sparse in the Neuron Basis. https://arxiv.org/abs/2601.22594
- Golden Gate Claude / Scaling Monosemanticity. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Sauers, Qwen3-8B tawny owl (128 trained neurons). https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons
