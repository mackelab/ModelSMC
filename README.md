# ModelSMC

Code release accompanying [Wahl, Schenk, Farnoud, Macke & Gedon (2026), *A Probabilistic
Framework for LLM-Based Model Discovery*](https://arxiv.org/abs/2602.18266).

This repository contains the ModelSMC implementation, the benchmark tasks used in the
paper, and the code and recorded results needed to reproduce its figures and tables.
The potassium/aldosterone task is not part of this release; its recorded results are
included.

> **Running ModelSMC requires LLM API keys** — see [API keys](#api-keys) before your
> first run. The only exception is `minimal_example_n_dim`, the LLM-free validation task.
>
> **Licensing:** most of this repository is MIT, but not all of it — see the exceptions
> under [License](#license) before reusing any of it.

If you use this code, please [cite the paper](#citation).

## Installation

Use either [uv](https://docs.astral.sh/uv/) (faster) or conda.

**uv** (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if not already available
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

**conda:**
```bash
conda env create -f environment.yml && conda activate modelsmc
```

## API keys

LLM API keys are required. We use dotenvx to manage them.
To install dotenvx, run one of the following or see [here](https://dotenvx.com/docs/install):

```bash
brew install dotenvx/brew/dotenvx
```
or
```bash
curl -fsS https://dotenvx.sh | sh
```

The file *.env_empty* provides a template containing all the environment variables you
need to run the code. Enter your secrets and encrypt the entire file using

```bash
dotenvx encrypt
```

After the first encryption of your secrets you can set new values using

```bash
dotenvx set < YOUR ENVIRONMENT VARIABLE > < value >
```

For more information about using encrypted environment variables see https://dotenvx.com/docs/.

Make sure that the private api key is in `.env.keys`. Never share your priviate key.

## Repository structure

```text
modelsmc/          the Python package, installed as `modelsmc`
├── main.py        Hydra entry point behind the `modelsmc` command
├── method/        the ModelSMC algorithm itself — see below
├── tasks/         task definitions: simulators, priors, data and prompts
└── utils/         shared helpers and plotting
config/            Hydra configs — task/, method/, llm/, experiment/, launcher/, sweeper/
main_results/      curated results reported in the paper
figures/           one folder per paper figure (notebooks, panels, output)
tables/            one folder per paper table
analysis/          supporting analyses
diagnostics/       integrity checks for result folders
curation/          cleaning and copying result folders
results/           local run output; gitignored, created on the first run
```

`analysis/`, `diagnostics/` and `curation/` each have their own README with per-script
documentation. Task-specific setup — including the data files that are not checked in —
is described in [`modelsmc/tasks/README.md`](modelsmc/tasks/README.md).

### Main method code

The algorithm lives in [`modelsmc/method/`](modelsmc/method):

| Path | Contents |
|---|---|
| `modelsmc.py` | `SMCEngine` and `SMCOrchestrator` — the core SMC loop over candidate models |
| `modelsmc_no_LLM.py` | `SMCEngineMinimal` / `SMCOrchestratorMinimal`, the LLM-free variant used for validation |
| `modules/codingsimulator.py` | DSPy module that prompts the LLM to write a simulator |
| `modules/feedback.py` | diagnoses a candidate's fit and turns it into feedback for the next proposal |
| `modules/evaluator.py` | scores a proposed simulator against the observed data |
| `modules/dataclasses.py` | `Particle`, `ParticlePool`, `DiscoveryContext` |
| `modules/optimizer/` | parameter-inference backends: ABC, NPE, and the TabPFN variant |
| `modules/posterior_estimator/`, `modules/likelihood_estimator/` | NPE / NLE density estimators and their TabPFN-based counterparts |

## Running ModelSMC

Installing the package provides a `modelsmc` command, equivalent to
`dotenvx run -- python -m modelsmc.main`. Configuration is handled by
[Hydra](https://hydra.cc/), so any config value can be overridden on the command line.

### Reproducing the paper's runs

Every run reported in the paper is checked in as an experiment config in
[`config/experiment/`](config/experiment). Each one pins the task, method, LLM and
hyperparameters and sweeps ten seeds, so one command reproduces a complete result:

```bash
modelsmc +experiment=allen_modelsmc_sonnet
```

The leading `+` is required — `experiment` is not one of the default config groups.

**All experiment configs target a slurm cluster** (`launcher: slurm`, `partition:
a100_1`). The configs in [`config/partition/`](config/partition) describe the GPU
partitions of our own cluster, so running these experiments elsewhere requires adding
partition configs that reflect the machines available to you. To run on the local
machine instead:

```bash
modelsmc +experiment=allen_modelsmc_sonnet launcher=local
```

| Task | Experiment configs |
|---|---|
| `allen_level0` | `allen_modelsmc_sonnet`, `allen_modelsmcN1_sonnet`, `allen_modelsmc_sonnet_save_history` |
| `SIR_level3` | `SIR_ModelSMC_sonnet`, `SIR_ModelSMC_N_1_sonnet` |
| `minimal_example_n_dim` | `minimal_example_no_LLM_ABC_NLE` (LLM-free validation) |
| ablations | `ablation_allen_modelsmc_*` — LLM, feedback mode, prompt, loss, simulation budget and pool size |

### Running a single task

To run one task directly, without an experiment config:

```bash
modelsmc task=allen_level0
```

The available tasks are the file names in [`config/task/`](config/task): `allen_level0`,
`SIR_level3`, `minimal_example_n_dim`, and the `*_base` variants they build on. The
other config groups — `method/`, `llm/`, `launcher/`, `partition/`,
`sweeper/` — are overridden the same way. See the [Hydra
documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) for the override
syntax and for multirun sweeps (`-m`).

## Diagnostics and curation

`diagnostics/` contains scripts for verifying the integrity of recorded experiment
data; `curation/` contains scripts for cleaning and copying result folders before
publication. Originals are never modified — the curation scripts write `_filtered`
variants alongside them. See [`diagnostics/README.md`](diagnostics/README.md) and
[`curation/README.md`](curation/README.md) for the individual scripts and their flags.

The step-by-step recipe for turning a raw run folder into a curated, publication-ready
one — as used to produce `main_results/` — is in [Full curation
workflow](curation/README.md#full-curation-workflow).

## License

This project is released under the MIT License, with the exceptions noted below.

**Built with PriorLabs-TabPFN.**

### Prior Labs License

The following code is derived from the [npe-pfn](https://github.com/mackelab/npe-pfn)
repository (J. Vetter, M. Gloeckler, D. Gedon, J. Macke) and is **not** covered by the
MIT License. It is licensed under the Prior Labs License, Version 1.1 (May 2025), a
copy of which is distributed at
[`modelsmc/method/modules/posterior_estimator/npe_pfn/LICENSE`](modelsmc/method/modules/posterior_estimator/npe_pfn/LICENSE):

- `modelsmc/method/modules/posterior_estimator/npe_pfn/` — copied from npe-pfn; some of
  these files have been modified.
- `modelsmc/method/modules/likelihood_estimator/nle_pfn.py` — adapted from the
  autoregressive log-probability routine of npe-pfn.

Each of these files carries a header identifying the upstream revision it derives from
and stating whether it has been modified.

This project additionally depends on [TabPFN](https://github.com/PriorLabs/TabPFN) at
runtime, which is also distributed under the Prior Labs License.

### Apache License 2.0

[`figures/Fig_1_overview_ModelSMC/notebooks/HH_helper_functions.py`](figures/Fig_1_overview_ModelSMC/notebooks/HH_helper_functions.py)
is modified from the [sbi](https://github.com/sbi-dev/sbi) project and is **not** covered
by the MIT License. It is licensed under the Apache License, Version 2.0, a copy of which
is distributed at
[`figures/Fig_1_overview_ModelSMC/notebooks/LICENSE-sbi`](figures/Fig_1_overview_ModelSMC/notebooks/LICENSE-sbi).

### Icon artwork

The vector icons embedded in the Figure 1 panels — `panels/panel_a.svg`,
`panels/panel_a_extended.svg` and the assembled `fig/fig_1.svg` under
`figures/Fig_1_overview_ModelSMC/` — are from [uxwing.com](https://uxwing.com/) and are
used under the [UXWing license](https://uxwing.com/license/). They are **not** covered by
the MIT License. The icon files themselves are not redistributed; see
[`figures/Fig_1_overview_ModelSMC/README.md`](figures/Fig_1_overview_ModelSMC/README.md)
for how to obtain them.

### Everything else

Unless noted otherwise, the code in this repository is released under the MIT License;
see [`LICENSE`](LICENSE). Parts of it are based on MIT-licensed code from other projects,
including the `allen` and `SIR` task simulators. Where that is the case, the upstream
source and its copyright notice are recorded either in a header at the top of the file
itself, or in a `NOTICE` file covering the directory it sits in and everything below it.

## Citation

```
@misc{wahl2026probabilisticframeworkllmbasedmodel,
      title={A Probabilistic Framework for LLM-Based Model Discovery}, 
      author={Stefan Wahl and Raphaela Schenk and Ali Farnoud and Jakob H. Macke and Daniel Gedon},
      year={2026},
      eprint={2602.18266},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.18266}, 
}
```