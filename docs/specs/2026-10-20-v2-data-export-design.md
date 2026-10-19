# v2 data export and best-reply picks

Part 1 of 3 (data -> static figures -> interactive page). Parts 2 and 3 read only the files this part writes.

## Goal

One command turns the v2 result files into one summary file, so the figures and the page show the same numbers.
It also writes a shortlist of the best replies. A person approves the shortlist by hand.

## Scope

- Figures: Yellow Emperor (reference, the source figure's neuron), Napoleon, Hitler, Mao (shown as "no neuron found").
- Points: every quick and held-out point of these figures in results/v2/ (for example L10.N5809 at -24, -32, -48).
- Out of scope: drawing figures, the HTML page, new model runs other than the one align.py run below.

## Prerequisite

Run `python align.py hitler yellow_emperor napoleon mao` once. This writes
`vectors/personas/_align_scores_hitler_yellow_emperor_napoleon_mao.npz` with one contrast set for all four figures.
The older score files used different contrast sets and are not read.

## Command

`python analyze.py export` (a new subcommand in analyze.py; it reuses read_jsonl, collapsed, the identity names and
the two-judge confirmation code of `discovery`).

## Output 1: results/v2/summary.json

```
{"figures": {"<figure>": {
   "persona_replies": {"total": 60, "kept": 28},
   "align": {"top": [{"unit": "L10.N5809", "sign": -1, "fraction": 0.032}, ...10],
             "best_per_layer": {"4": 0.008, ...}},
   "points": {"<unit>:<dose>": {
       "set": "heldout" | "quick", "n": 40,
       "on_topic": 23, "first_person": 15, "names_self": 2, "loops": 13, "assistant": 5,
       "judge": {"match": 12, "bad": 13, "confirmed": false}}},
   "picks": [{"id": "...", "point": "...", "question": "...", "text": "..."}]}}}
```

- The regex patterns for on_topic, first_person and names_self are defined once per figure in analyze.py
  (the same patterns used in the wrap-up counts). assistant and loops use one shared pattern and `collapsed`.
- "judge" is filled only for points with a name in identity_names.jsonl, with the same rule as `discovery`.
- Only picks with keep = true are copied into summary.json.

## Output 2: results/v2/best_picks.jsonl

- Up to 10 rows per figure: `{"id", "figure", "point", "question", "text", "keep": null}`.
- Ranking: first_person match first, then not looping, then judge match, then shorter text.
- The person sets keep to true or false by hand.
- A new run never changes an existing row. It only adds new candidates for a figure that has fewer than 10 rows.

## Errors

- A missing input file is read as empty (read_jsonl already does this); the figure then shows n = 0.
- A pick whose id is no longer in the reply files is left out of summary.json and printed as a warning.

## Check

`python analyze.py --selftest` gets two new asserts:
1. The regex counts on 3 fixed example replies (first-person, third-person, loop).
2. A second export keeps a row that a person marked keep = false.
