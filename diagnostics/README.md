# diagnostics/

Scripts for inspecting and validating experiment result folders.

## `summarize_summary_file.py`

Group runs by config and print per-group diagnostics (iteration counts, unique particles, valid-particle rate, and — when present in summary.csv — timeout rate and token usage).

```
python diagnostics/summarize_summary_file.py <folder>
python diagnostics/summarize_summary_file.py <folder> --no-check
```

## `check_pool_summary_consistency.py`

Cross-check `summary.csv` against `pool_composition.csv` (run-ID presence, iteration match, pool-size check).

```
python diagnostics/check_pool_summary_consistency.py <folder>
python diagnostics/check_pool_summary_consistency.py <folder> --verbose
```

## `compare_configs_across_runs.py`

Compare Hydra configs stored in `summary.csv` files across a two-level
`<method>/<llm>` folder hierarchy.  For each inner subfolder the script first
checks that all configs within the file differ only in `seed` and `run_id`
(warns otherwise), then extracts one representative config per group and prints
a comparison table.  Columns are labelled `METHOD.LLM`; rows are nested
dot-notation config keys grouped by their top-level key (blank line between
groups).

```
# show only keys that differ across groups
python diagnostics/compare_configs_across_runs.py <root_folder>

# show all keys; rows that differ are marked with a leading '*'
python diagnostics/compare_configs_across_runs.py <root_folder> --full

# skip the intra-file seed/run_id sanity check
python diagnostics/compare_configs_across_runs.py <root_folder> --no-check

# combine flags
python diagnostics/compare_configs_across_runs.py <root_folder> --full --no-check
```

## `check_gmm_idx_distribution_consistency.py`

Cross-check `summary.csv` against `GMM_idx_distribution.csv` (run-ID presence, duplicates, iteration match, count-sum vs pool size). Only relevant for the minimal-example task.

```
python diagnostics/check_gmm_idx_distribution_consistency.py <folder>
python diagnostics/check_gmm_idx_distribution_consistency.py <folder> --verbose
```
