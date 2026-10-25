# 3D layer map for one-neuron-is-all-you-need: design

Date: 2026-10-25. Status: version 1 and 1.5 built and tested (see Build log). Lens and "writes" layout not built. Decisions in this version were made from the data in `results/`
and `vectors/personas/` (checked 2026-10-25), as you asked.

## Goal

Add a 3D map of Qwen3-8B to this repo. The map shows every neuron that this project has data for, as a point in its
layer. A reader sees where the steered neurons sit, sees which neurons the persona ranking scores high, and clicks a
neuron to read its real replies at each dose.

The map answers one question the repo cannot answer today: where in the model do these neurons sit? The README says
the persona neurons are in layers 8 to 18 and that late layers do nothing. The map shows that at a glance.

## Source

The map module comes from `neuron-explorer` (`explorer/static/map3d.js`, 545 lines, Three.js 0.169.0).
It draws layers as sheets of points, with orbit, zoom, hover, pick and a face-on view of one layer.
We copy it. We do not rewrite it.

## Non-goals

- No server. The page must work on GitHub Pages, like `docs/explorer.html` and `docs/explorer_v2.html`.
- No live steering, no model loading, no agent panel.
- No connection web. This repo has no connection data.

## What data exists (checked)

| Tier | Content | Size | Quality |
|---|---|---|---|
| A: full | v1 neurons L18.N7807, L8.N1931, L9.N3757, L8.N5315: 9 doses, 30 questions, 4 samples | 3,240 replies, all strict-judge labelled | Best. Labelled, with the README's confidence intervals. |
| A: full | v2 neurons L9.N598, L9.N10974, L10.N5809: 40 held-out questions | 280 replies, labelled in `labels_v2.jsonl` | Good. Small samples, regex and judge labels. |
| B: screening | 58 more units from the discovery run, mostly 16 replies each | 520 replies | Weak. Few replies, no labels. Shows where we looked, not what the neurons do. |
| C: score only | Persona alignment score for **every** neuron in layers 0 to 27, for 4 figures | 4 x 28 x 12,288 floats | Complete over its range. It is an explained-fraction score, not a measured effect. |

Facts found while checking:

- Layers 0 to 27 only. `candidates.py` sets `N_LAYERS = 28`. So the README's "rank 1 of 344,064" is correct:
  28 x 12,288 = 344,064. [source] Layers 28 to 35 have no persona data and no steering data at all.
- Layer 0 cannot be steered (comment in `candidates.py`). [source]
- Scores are heavy-tailed. For Yellow Emperor the top neuron scores 0.066, and 30,063 of 344,064 are above zero.
  Almost all of the signal is in the top few hundred. [measured]
- Mao's top score is at L26.N4138, in a late layer, and the v2 README says no Mao neuron worked. A high score
  does not mean a working neuron. The page must say this next to the Mao view. [measured]
- v2 replies store `alpha` and `q` as strings. The build script must convert them. [measured]
- All 36 layers of `down_proj` are in shards 1 to 4 of `models/qwen3-8b-bf16`. Shard 5 is not needed for them.
  [measured: from the weight index] My first draft said the weights were incomplete. That was wrong.

## Decisions

1. **Show layers 0 to 27 only.** No data exists past layer 27. Drawing eight empty sheets would suggest "we looked
   and found nothing", and we did not look. The page says "layers 28 to 35 not scanned".
2. **Colour every neuron by persona score, with a figure switch.** Tier C is the only complete data in the repo, and
   it is what makes the map more than a list of dots. The user picks Yellow Emperor, Napoleon, Hitler or Mao.
   Labels say "persona score" and carry the Mao caveat above.
3. **Show tier B, visibly weaker.** Dimmer marks, and the panel says "screening sample, n replies, no labels".
   They support the README claim that late layers do nothing, and hiding them would hide the negative result.
4. **Three.js from a pinned CDN URL.** The data does not depend on it, and the repo's other pages are not security
   sensitive. Vendor the files only if you want zero external requests.
5. **Index-order layout in version 1.** See Layout.

## Files

| File | What it is |
|---|---|
| `docs/map3d.js` | Copy of the neuron-explorer module. No edits in version 1. |
| `docs/map.html` | One page. Canvas, side panel, a small script that loads the data and wires the map to the panel. |
| `build_map.py` | Reads `results/` and `vectors/personas/`, writes `docs/map_data.json`. Same role as `build_explorer.py`. |
| `docs/map_data.json` | Output of the build script. Committed. The git-ignored `.npz` files are inputs only. |

README gets one row in the repo map and one link next to the two explorer links.

## Data format (`map_data.json`)

- `arch`: `n_layers` = 28, `inter` = 12288, `layer_types` all `full_attention`, `attn_heads`. The script reads them
  from the model config and asserts them. [unverified: Qwen3-8B config values; the script checks]
- `units`: one entry per unit, key like `L18.N7807`, with `layer`, `n`, `tier` (`A` or `B`), `sign` (sign of the
  dose that works, or `null` for tier B), `note`, and `doses`: dose to at most 3 replies of
  `{question, text, label}`. Tier B has no `label`.
- `scores`: for each figure, the top 500 neurons as `[layer, n, score]`, plus `max` and `n_positive`.
  Neurons outside the top 500 draw as idle points. Size is small (about 4 x 500 entries).

Sources: `results/replies.jsonl` + `labels_strict.jsonl` (v1), `results/v2/replies.jsonl` + `labels_v2.jsonl` (v2 A),
`results/v2/discovery.jsonl` (B), `vectors/personas/_align_scores_hitler_yellow_emperor_napoleon_mao.npz` (C).

Privacy: the script must not open anything with `private` in its path (`results/private/`, `results/v2/private/`
are git-ignored). A check in the script enforces this.

Size target: under 1 MB. Replies are capped at 3 per dose and 400 characters each.

## Geometry and layout

- 28 layers on the x axis, one sheet of 12,288 points each: 344,064 points in one draw call. That is 17% more than
  neuron-explorer's 294,912.
- The module puts neuron n at cell n on a grid of `ceil(sqrt(12288))` = 111 columns. The last row is partial. The
  module computes the cell from the index, so a non-square count works. [unverified: neuron-explorer's
  `build_layout.py` asserts a square, the map module does not; confirm by running]
- Version 1: index order. Position inside a sheet means nothing. The page says so in one line.
- Version 2 (later): a "writes" layout, as in `build_layout.py` (UMAP of `down_proj` columns). The weights for
  layers 0 to 27 are available locally, so nothing blocks this. Do it only after version 1 shows it is worth it.

## Interaction

1. Page loads with the Yellow Emperor score view. Tier A marks are bright, tier B marks are dim.
2. A figure switch changes the score colours. The amber/pink mark colour still shows the dose sign.
3. Hover shows `L18.N7807 · +32 · Biblical God` or, for a scored neuron, `L4.N7668 · score 0.047`.
4. Click opens the side panel: ID, note, tier, dose selector, replies and labels. It shows data and makes no extra
   claims, as `explorer.html` does.
5. Click on a layer label or slab enters the face-on view (already in the module). Escape leaves it.
6. A list in the panel groups units by v1 and by figure. A click flies the camera to the neuron.

## Changes to the copy

- Call `setInsets(0, panelWidth)` from `map.html`. The module defaults to a 410 px left inset for the explorer's
  list. No edit to the module is needed.
- Feed scores through `setMarks(marks)` with `w` set from the score. No edit needed. [unverified: `setMarks`
  clamps nothing, so `w` must be scaled to 0 to 1 by the page]
- Put the neuron-explorer commit (`f79b219`) in a comment at the top of `docs/map3d.js`.

## Risks

- Performance. 344,064 points is 17% above neuron-explorer. [unverified on your machines] Step 5 below measures it.
- The CDN. One new external request. Pin `three@0.169.0`.
- Misreading the score. A reader may take a high score as proof. The Mao view is the counter-example. The page text
  and the legend say "explained fraction of the persona direction, not a measured effect".
- Content. v2 includes Hitler replies. They are already public in `explorer_v2.html`. The map adds none.

## Verification

1. `python build_map.py` prints units per tier, replies, and file size. It asserts the arch values, that every unit
   lies inside the grid, that v2 strings convert, and that no input path contains `private`.
2. Check by script that the four v1 neurons sit in layers 18, 8, 9, 8, and that the top score per figure matches
   `summary.json` (L9.N598, L9.N10974, L10.N5809, L26.N4138).
3. Serve `docs/` locally. In a headless browser, check no console errors, a drawn canvas, 344,064 points, and the
   expected mark count.
4. Click a mark. Compare the first reply on screen with the same ID in `results/replies.jsonl`.
5. Measure frame time while orbiting on the 6 GB GPU machine. Write the number in the README. [measured]

## Order of work

1. `build_map.py` and `map_data.json`. Data first, so the page shows only true things.
2. `docs/map.html` and `docs/map3d.js`: points, score colours, tier marks, hover.
3. Side panel, dose selector, figure switch.
4. README row and link.
5. Measure. Then decide on the "writes" layout.

## Build log (2026-10-25)

Built: `build_map.py`, `docs/map_data.json` (150 KB), `docs/map.html`, `docs/map3d.js`, README link and repo-map row.

Version 1.5 features built: figure switch, specificity bars (4 figures on one axis), write-vector norm in the panel,
dose strip with the label mix per dose, per-layer best-score chart, search box, tour (9 stops), permalinks.
Not built: point size by write norm (the module has no per-point size), logit lens for scored neurons, "writes" layout.
Lens text is shown only for tier B units that already carry it in `candidates.json`.

Facts found while building:
- `discovery.jsonl` also holds whole-direction runs (`vec:...`). They are not neurons and are skipped. 50 screening
  neurons remain after that. [measured]
- 7 tier A neurons, 2,008 scored rows (top 500 per figure plus the steered units), 3,848 replies in the file. [measured]
- The build asserts that its top neuron and score per figure match `results/v2/summary.json`. It passed. [measured]

Changes to the copied module (`docs/map3d.js`, header comment lists them):
1. `setMarks` accepts `m.color`.
2. Default brightness 2.5 to 0.7. At 2.5 the sheets are solid white from far away and the marks vanish.
3. Face-on layer view fades the other layers fully. Before, a quarter of each stayed and piled up as a glow in the
   middle of the sheet, which looked like a cluster of neurons. The page also hides marks of other layers there.

Verification run (headless Chromium, software GL, Three.js from a local copy): no console errors; 344,064 points;
57 units; click on L18.N7807 shows the first reply from `results/replies.jsonl` ("I am the Lord, and I am with
you..."); dose strip at +32 shows 70% god, as the README says; tour, layer view and a permalink load all work.
[measured] Not verified: frame rate on a real GPU, phone layout, the CDN import (the test used a local copy).
