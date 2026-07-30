# analysis/

Methodology notebooks and utility scripts supporting the paper's analyses.

## `allen_cr/` — Allen channel-taxonomy pipeline

Two-notebook, LLM-driven pipeline that discovers a taxonomy of ion channels the
LLM added beyond the base Hodgkin-Huxley model, then classifies every particle
against it. This is the methodology record behind
`main_results/allen_posterior_mass_analysis/` (`allen_taxonomy.json`,
`allen_raw_descriptions.json`, `allen_classified.csv`), which
`figures/Fig_5_posterior_mass_analysis/` and
`figures/Fig_F2_cross_seed_rank_stability/` read directly.

**To reproduce from scratch**, run in order:

1. **`01_extract_taxonomy.ipynb`**
   Phase 2 samples a stratified subset of particles and asks the LLM to describe,
   in free form, what each one added beyond the base Na/K/Leak channels; Phase 3
   makes one LLM call to consolidate those descriptions into a two-level
   family → subtype taxonomy. Produces `allen_raw_descriptions.json` and
   `allen_taxonomy.json`.
   Requires `RESULTS_ROOT` to point at the raw experiment output directory
   containing every particle's generated model
   (`<timestamp>/<seed>/iter-<i>_p<j>/simulator.py`) — **not** the curated
   `main_results/` folder, which only keeps `best_particle/`/`baseline/` per
   seed. **Stop and manually review** the printed taxonomy before continuing — edit
   `allen_taxonomy.json` directly to merge/split/rename subtypes if needed.

2. **`02_classify_particles.ipynb`**
   Phase 4 classifies every usable particle against the taxonomy from step 1,
   checkpointing to `allen_classified.csv` every 25 particles so it's safe to
   interrupt and resume. Phase 5 prints a random spot-check sample (full code +
   assigned label + reasoning) to catch systematic misclassifications before
   trusting the result.

Both notebooks are **resumable**: re-running a cell skips work whose output
file already exists — delete the file to force a re-run of that phase. Both
also require an `ANTHROPIC_API_KEY`, loaded via `dotenvx` (run the notebook's
kernel from an environment where `dotenvx get ANTHROPIC_API_KEY` succeeds).

## `generate_llm_history_latex.py`

Renders an LLM propagation/feedback history JSON as LaTeX, for embedding
worked examples of the LLM dialogue (prompts, responses, code) in the paper
appendix.

```
python analysis/generate_llm_history_latex.py <particle_folder>
```

`<particle_folder>` must contain `llm_history_propagation.json` and/or
`llm_history_feedback.json` (e.g.
`main_results/allen_LLM_history_recording/iter-4_p39/`). Writes
`llm_history_propagation.tex`, `llm_history_feedback.tex`, and a shared
`llm_history_preamble.tex` (color/style definitions to `\input` once in the
main document's preamble) into the same folder.
