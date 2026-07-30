# curation/

Scripts for cleaning and copying experiment result folders.

## `remove_run_ids.py`

Remove specific run_ids from the CSV files in a results folder. Writes `_filtered` variants without modifying the originals. Incremental by default (reads `_filtered` if it exists).

```
python curation/remove_run_ids.py <folder> --run-ids <id1> <id2> ...
```

## `copy_results.py`

Copy CSV files (`summary.csv`, `pool_composition.csv`, `GMM_idx_distribution.csv`) and optionally the best-particle folders from one folder to another. Copies the `_filtered` variants by default (stripping the suffix); pass `--raw` to copy originals.

```
python curation/copy_results.py <input_folder> <target_folder>
python curation/copy_results.py <input_folder> <target_folder> --raw
python curation/copy_results.py <input_folder> <target_folder> --no-copy-best-particle
python curation/copy_results.py <input_folder> <target_folder> --overwrite
python curation/copy_results.py <input_folder> <target_folder> --copy-baseline
python curation/copy_results.py <input_folder> <target_folder> --ensure-parameter-estimates
```

| Flag | Default | Description |
|---|---|---|
| `--raw` | off | Copy original (unfiltered) files instead of `_filtered` variants. |
| `--overwrite` | off | Replace existing files/folders in the target instead of raising an error. |
| `--no-copy-best-particle` | off | Skip copying the `best_particle/` folder for each run (best-particle copying happens by default; this flag turns it off). |
| `--copy-baseline` | off | Also copy the `iter-0_p0/` baseline folder for each run (requires best-particle copy). |
| `--ensure-parameter-estimates` | off | If `parameter_estimates.pt` is absent in a copied folder, run posthoc parameter estimation automatically. |

Best-particle folders are located by scanning `<input_folder>` for `.hydra/config.yaml` files whose `run_id` matches; any path containing a `best_particle` component is skipped so that a run's own top-level Hydra config is matched instead of the config snapshot nested inside its `best_particle/` folder. `<input_folder>` must therefore be a raw Hydra run/sweep output directory (each run folder carrying its own top-level `.hydra/config.yaml`, e.g. `results/<name>/<timestamp>/`) — not an already-curated folder produced by a previous run of this script, since that only retains the config nested inside `best_particle/`.

## `reevaluate_numsim.py`

Post-hoc parameter re-evaluation at 5000 simulations for the Allen numsim ablation runs in `main_results/allen_ablation/{numsim_200,numsim_500,numsim_1000}/<llm>/run_*/`. For each run: renames `best_particle/` to `original_best_particle/` (skipped if already done), rebuilds `best_particle/` with the copied simulator code and a `.hydra/config.yaml` overriding `num_simulations` to 5000, then calls `SMCOrchestrator.posthoc_parameter_estimation` using the training/validation data from the original run. Only the rename step is idempotent — the parameter-estimation call itself re-runs every time, even if already done. Requires `dotenvx run --` to load the environment.

```
dotenvx run -- python curation/reevaluate_numsim.py
dotenvx run -- python curation/reevaluate_numsim.py --numsim numsim_1000
dotenvx run -- python curation/reevaluate_numsim.py --run run_2
dotenvx run -- python curation/reevaluate_numsim.py --llm gptmini
```

| Flag | Default | Description |
|---|---|---|
| `--numsim` | all | Restrict to one numsim folder (`numsim_200`, `numsim_500`, or `numsim_1000`). |
| `--run` | all | Restrict to one run (e.g. `run_2`). |
| `--llm` | `sonnet` | LLM subfolder to process; re-run with a different value (e.g. `gptmini`) to process another one. |

## `remove_error_column.py`

Remove the `errors` and `warnings` columns from recorded `summary.csv` files. The stored tracebacks embed absolute cluster paths that expose local user names, so `errors` is stripped before a results folder is published; `warnings` holds runtime messages and is dropped along with it for consistency.

Walks the given folder recursively and rewrites every `summary*.csv` that carries one of the columns **in place** — unlike the other scripts here there is no `_filtered` variant, since leaving the original behind would defeat the purpose. All remaining values are preserved exactly as recorded. Defaults to `main_results/`, and nothing is written unless `--apply` is passed.

```
python curation/remove_error_column.py                        # dry run over main_results/
python curation/remove_error_column.py --apply                # rewrite in place
python curation/remove_error_column.py <folder> --apply       # a different folder
python curation/remove_error_column.py --columns errors       # only one column
```

| Flag | Default | Description |
|---|---|---|
| `--apply` | off | Actually rewrite the files. Without it the script only reports which files carry the columns. |
| `--columns` | `errors warnings` | Columns to remove. Names that are absent from a file are ignored. |

## `remove_hydra_yaml.py`

Delete the `.hydra/hydra.yaml` snapshots from a results folder. `hydra.yaml` is Hydra's own runtime record — `cwd`, `output_dir`, launcher settings and the job environment — so it stores absolute paths that expose local user names and the directory layout of the machine the experiments ran on.

The sibling files in `.hydra/` are kept: `config.yaml` is required by [`reevaluate_numsim.py`](#reevaluate_numsimpy), `copy_results.py --ensure-parameter-estimates` and `SMCOrchestrator.posthoc_parameter_estimation`, and `overrides.yaml` is a two-line record of how the run was launched. Neither contains absolute paths. A `.hydra/` folder left empty by the deletion is removed too.

Defaults to `main_results/`, and nothing is deleted unless `--apply` is passed.

```
python curation/remove_hydra_yaml.py                     # dry run over main_results/
python curation/remove_hydra_yaml.py --apply             # delete
python curation/remove_hydra_yaml.py <folder> --apply    # a different folder
```

| Flag | Default | Description |
|---|---|---|
| `--apply` | off | Actually delete the files. Without it the script only lists what would be removed. |

Note that `SMCOrchestrator` copies the whole `.hydra/` folder into `best_particle/` when saving a run, so new experiments will produce `hydra.yaml` again — re-run this script after curating further results.

## `remove_stored_data.py`

Delete the stored `training_data.pt` and `validation_data.pt` from a results folder. Every saved particle folder carries a verbatim copy of the data the run was fitted against, so these files dominate the size of the repository — several hundred MB across `main_results/`.

Only `SMCOrchestrator.posthoc_parameter_estimation` and [`reevaluate_numsim.py`](#reevaluate_numsimpy) read them, and both need the task's data available anyway; no figure, table or analysis script does. `parameter_estimates.pt` holds inferred parameters rather than observations and is left untouched.

Defaults to `main_results/`, and nothing is deleted unless `--apply` is passed.

```
python curation/remove_stored_data.py                          # dry run over main_results/
python curation/remove_stored_data.py --apply                  # delete
python curation/remove_stored_data.py <folder> --apply         # a different folder
python curation/remove_stored_data.py --names training_data.pt # only one file name
```

| Flag | Default | Description |
|---|---|---|
| `--apply` | off | Actually delete the files. Without it the script only lists what would be removed. |
| `--names` | `training_data.pt validation_data.pt` | File names to remove. |

## Full curation workflow

The recommended workflow for preparing a results folder for publication. It combines the
scripts above with the integrity checks in [`diagnostics/`](../diagnostics/README.md).

**Step 1 — Inspect the raw results**

Run the diagnostics on the original `summary.csv` to get an overview of all recorded
runs and spot any issues:

```bash
python diagnostics/summarize_summary_file.py            <results_folder>
python diagnostics/check_pool_summary_consistency.py    <results_folder>
# for minimal-example tasks only:
python diagnostics/check_gmm_idx_distribution_consistency.py <results_folder>
```

`summarize_summary_file.py` groups runs by their Hydra config and prints a table with
`run_id`, seed, iteration count, and fraction of valid particles — use this to identify
runs that are incomplete, failed, or otherwise unwanted.

**Step 2 — Remove unwanted run_ids**

Pass the `run_id`s identified in step 1 to the removal script. The script writes
`_filtered` variants of the CSV files and never touches the originals, so removals can
be applied incrementally:

```bash
python curation/remove_run_ids.py <results_folder> --run-ids <id1> <id2> ...
# repeat as needed; subsequent calls automatically read the latest _filtered file
```

**Step 3 — Copy to main_results**

Once the filtered files look correct, copy everything to the archive folder:

```bash
python curation/copy_results.py <results_folder> main_results/<experiment_name>/
```

This copies the `_filtered` CSV files (renaming them to the canonical filenames) and
the best-particle folder for every remaining `run_id`.

**Step 4 — Verify the final output**

Run the same diagnostics on the target folder to confirm that the curated copy matches
expectations:

```bash
python diagnostics/summarize_summary_file.py            main_results/<experiment_name>/
python diagnostics/check_pool_summary_consistency.py    main_results/<experiment_name>/
# for minimal-example tasks only:
python diagnostics/check_gmm_idx_distribution_consistency.py main_results/<experiment_name>/
```

The output should now show only the intended runs with no consistency errors.

**Step 5 — Strip the recorded error tracebacks**

Before publishing, remove the `errors` and `warnings` columns from the copied
`summary.csv` files. Check what would change first, then apply it:

```bash
python curation/remove_error_column.py main_results/<experiment_name>/
python curation/remove_error_column.py main_results/<experiment_name>/ --apply
```

This rewrites the files in place, so run it only once the folder is otherwise final.

**Step 6 — Drop the Hydra runtime snapshots**

`.hydra/hydra.yaml` records the working directory and job environment of the machine
the run was executed on. Remove it, keeping `config.yaml` and `overrides.yaml`:

```bash
python curation/remove_hydra_yaml.py main_results/<experiment_name>/
python curation/remove_hydra_yaml.py main_results/<experiment_name>/ --apply
```

**Step 7 — Drop the stored observations**

Each saved particle folder carries a verbatim copy of the data the run was fitted
against. Remove it, keeping `parameter_estimates.pt`:

```bash
python curation/remove_stored_data.py main_results/<experiment_name>/
python curation/remove_stored_data.py main_results/<experiment_name>/ --apply
```

Do this last: `--ensure-parameter-estimates` and `reevaluate_numsim.py` both need these
files, so any post-hoc re-evaluation has to happen first.
