# Tasks

## Existing Tasks

`template/` is a scaffold for building new tasks, not a runnable task itself — see [Creating a New Task](#creating-a-new-task) below.

### SIR

Adapted from [samholt/generative-simulations](https://github.com/samholt/generative-simulations) and [arXiv:2506.09272](https://arxiv.org/pdf/2506.09272).

The data used for the paper is checked in at `SIR/level3/data/` — `train_data.pt`, `valid_data.pt`, and a `metadata.json` recording the generation parameters. No action is needed.

It can be regenerated with

```bash
python -m modelsmc.tasks.SIR.level3.generate_data --seed 42
```

but this **overwrites the checked-in files and does not reproduce them**: the simulator dynamics draw from a fresh, unseeded generator on every step, so each run yields different trajectories. See the reproducibility note at the top of [`generate_data.py`](SIR/level3/generate_data.py).

### allen

Electrophysiology recordings from the Allen Cell Types Database (Allen Institute for Brain Science, 2015); simulator adapted from [mackelab/IdentifyMechanisticModels_2020](https://github.com/mackelab/IdentifyMechanisticModels_2020/tree/master/6_allen).

Not checked in. 10 files are required, named `ephys_cell_<cell_id>_sweep_number_<sweep>.pkl` for the `(cell_id, sweep)` pairs listed in `get_allen_task_parameters` in `external_code/allen_utils.py`. Download them, unmodified, from [`IdentifyMechanisticModels_2020/6_allen/support_files`](https://github.com/mackelab/IdentifyMechanisticModels_2020/tree/b93c90ec6156ae5f8afee6aaac7317373e9caf5e/6_allen/support_files) and place them in `allen/data/`.

### minimal_example_n_dim

Synthetic Gaussian Mixture Model, no external source. Checked in as `gmm_configs.json` (regenerate via `generate_models.py --new_config`).

## Creating a New Task

Copy `modelsmc/tasks/template/` and rename it. Each task needs:

```text
modelsmc/tasks/<task_name>/
├── __init__.py
├── <task_name>_task_base.py          # sets self.prior_dist; implements get_data,
│                                      # simulation_wrapper, eval_function, plot_observation
│                                      # (see base_task.py for signatures)
└── <level_name>/                     # e.g. level0
    ├── <task_name>_task.py           # @register_task("<config_name>"), imported in tasks/__init__.py
    ├── prompts.yaml                  # system_description, signature_description, task_description
    └── base_simulator.py             # simulator template

config/task/
├── <task_name>_base.yaml
└── <task_name>_<level_name>.yaml     # name must match the @register_task config name
```

Run with:

```bash
modelsmc task=<config_name>
```
