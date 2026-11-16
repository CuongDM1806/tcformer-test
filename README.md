# SSM-DAN

Code for **Domain-Adaptive Selective State-Space Learning for Cross-Subject Motor-Imagery EEG Decoding**.

SSM-DAN decodes motor-imagery EEG of an unseen subject (leave-one-subject-out, LOSO). It combines:

- **Backbone**: multi-scale temporal convolutions, five bidirectional selective state-space (Mamba-style) blocks and a grouped TCN head.
- **Riemannian alignment (RA)** of every subject's trials.
- **Stage 1, unsupervised domain adaptation**: AutoDIAL feature normalization with a learned source/target mixing weight, a residual aligner, adversarial and multi-kernel MMD alignment, and target information maximization.
- **Stage 2, IM-TTA**: test-time adaptation of the BatchNorm affine parameters on the unlabeled evaluation trials.

One code base runs every experiment in the paper on BCI IV-2a, BCI IV-2b, Zhou2016 and OpenBMI. That covers SSM-DAN, its ablations and the eight re-trained baselines. Each experiment is selected by configuration, and all outputs go to one results layout.

## Contents

```
configs/
  datasets.yaml          data protocol and preprocessing shared by all methods
  methods/*.yaml         one file per method; SSM-DAN ablations are "variants"
ssm_dan/
  run.py                 command-line entry point (python -m ssm_dan.run)
  config.py              dataset + method + variants + overrides -> run configuration
  pipeline.py            one LOSO fold: Stage 1, checkpoint selection, Stage 2, scoring
  results.py             output layout and summaries
  data/                  LOSO data modules, loaders, Riemannian alignment, S&R augmentation
  models/                SSM backbone, SSM-DAN, Lightning wrapper, re-implemented baselines
  baselines/             RMETNet, T-TIME and BFT (their own training loops)
  kaggle.py              helpers for the Kaggle notebooks
notebooks/               Kaggle: run any experiment, build the OpenBMI cache, parity check
scripts/                 build the OpenBMI cache; paper-style tables and statistics
tests/                   configuration parity with the reported runs, unit tests, parity probe
```

## Installation

Python 3.10 is required, because MOABB 0.5 needs Python < 3.11 and NumPy < 2.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install transformers==4.48.3        # only for the EEGEncoder baseline
```

MOABB downloads BCI IV-2a, BCI IV-2b and Zhou2016 on first use, into `~/mne_data` or `$MNE_DATA`. OpenBMI is handled separately; see [OpenBMI](#openbmi).

## Running experiments

```bash
python -m ssm_dan.run --method ssm_dan --dataset bcic2a                 # SSM-DAN, all 9 folds
python -m ssm_dan.run --method ssm_dan --dataset zhou2016 --variant no_ra
python -m ssm_dan.run --method ssm_dan --dataset openbmi --subjects 1-18 --gpu 1
python -m ssm_dan.run --method eegconformer --dataset bcic2b
python -m ssm_dan.run --method ssm_dan --dataset bcic2a --subjects 1 --set max_epochs=2   # quick check
python -m ssm_dan.run --list                                            # methods and variants
```

| Option | Meaning |
|---|---|
| `--method` | A file in `configs/methods/`: `ssm_dan`, `eegconformer`, `atcnet`, `tcformer`, `eegencoder`, `satransnet`, `rmetnet`, `ttime`, `bft`. |
| `--variant` | Variants of the method, comma-separated and applied in order (e.g. `no_car,no_ra`). |
| `--dataset` | `bcic2a`, `bcic2b`, `zhou2016` or `openbmi`. |
| `--subjects` | Target subjects: `all`, `3`, `1-9` or `1,4,7-9`. |
| `--seed` | Random seed. Every fold is seeded independently, so folds can run in any order or in parallel. |
| `--set KEY=VALUE` | Override any configuration entry, e.g. `preprocessing.batch_size=32` or `adaptation_kwargs.im_tta_steps=10`. Repeatable. |
| `--results` | Results root (default `./results`, or `$SSM_DAN_RESULTS`). |
| `--overwrite` | Re-run folds that are already finished. Without it, they are skipped. |

### Experiments of the paper

| Paper | Command (add `--dataset ...`) |
|---|---|
| SSM-DAN | `--method ssm_dan` |
| SSM-DAN (source-only) | `--method ssm_dan --variant source_only` |
| SSM-DAN (Stage 1 UDA) = without test-time adaptation | `--method ssm_dan --variant no_tta` |
| SSM-DAN (IM-TTA) = without domain adaptation | `--method ssm_dan --variant no_da` |
| Without Riemannian alignment | `--method ssm_dan --variant no_ra` |
| Session-specific RA reference | `--method ssm_dan --variant per_session_ra` |
| OpenBMI source-only column of the ablation table | `--method ssm_dan --variant source_only_head --dataset openbmi` |
| OpenBMI without common average reference | `--method ssm_dan --variant no_car --dataset openbmi` |
| EEG Conformer, ATCNet, TCFormer, EEGEncoder, SATrans-Net | `--method eegconformer` / `atcnet` / `tcformer` / `eegencoder` / `satransnet` |
| RMETNet, T-TIME, BFT | `--method rmetnet` / `ttime` / `bft` |
| Computational cost | `param_count`, `train_time_min` and `cpu_latency_ms` in every subject record |

Every SSM-DAN run also scores the same trained weights without IM-TTA (`no_tta_acc`, `no_tta_kappa`). The Stage-1 UDA numbers therefore come with the full model at no extra cost.

## Results

All methods, variants and datasets write to the same place:

```
results/<dataset>/<run>/seed<seed>/        <run> = method[-variant...], e.g. ssm_dan-no_ra
    config.yaml               resolved configuration of the run
    subjects/S01.json         one record per target subject
    checkpoints/S01.ckpt      best source-validation checkpoint
    figures/                  per-fold confusion matrix and learning curves
    logs/                     per-fold logs (Kaggle notebook)
    summary.csv, summary.txt  per-subject rows, mean and standard deviation
```

A subject record holds:

- `test_acc`, `test_kappa`, `test_loss`: the evaluation session(s), after IM-TTA.
- `no_tta_*`: the same weights without IM-TTA.
- `all_sessions_*`: an auxiliary view over every target session.
- `best_val_acc`: the source-validation accuracy of the selected checkpoint.
- `confusion_counts`, `autodial_alpha`, `param_count`, `train_time_min`, `cpu_latency_ms`.

Paper-style tables (per-subject accuracy, mean ± SD, κ, ITR, exact Wilcoxon test with Holm correction) are built from these records:

```bash
python scripts/summarize.py --dataset bcic2a --reference ssm_dan
python scripts/summarize.py --dataset bcic2a --reference ssm_dan-source_only \
    --runs eegconformer,atcnet,tcformer,eegencoder,satransnet
```

## Configuration

A run is resolved from four layers, each overriding the previous one:

1. `configs/datasets.yaml[dataset]`: the session protocol and the preprocessing shared by all methods (sampling rate, filter, window, z-scoring, S&R augmentation).
2. `configs/methods/<method>.yaml`: the model, the optimizer and the method's defaults, then its `datasets.<dataset>` block (e.g. a batch size of 32 and a learning rate of 1e-3 on Zhou2016).
3. The requested `variants` of the method.
4. `--set` overrides.

The resolved configuration is saved as `config.yaml` with the results. To add an experiment, add a variant to a method file, or a new method file. No new code or notebook is needed.

| Dataset | Source training / validation | Target Stage 1 (unlabeled) / evaluation |
|---|---|---|
| BCI IV-2a | session T / session E | session T / session E |
| BCI IV-2b | sessions 1–3 / 4–5 | sessions 1–3 / 4–5 |
| Zhou2016 | sessions 1–2 / 3 | sessions 1–2 / 3 |
| OpenBMI | session 1 / 2 | session 1 / 2 |

Checkpoints are selected on the labeled source validation sessions only. Target labels are read only for scoring.

## Kaggle

The three notebooks are short. They clone this repository and call `ssm_dan/kaggle.py`.

| Notebook | Purpose | Open |
|---|---|---|
| `notebooks/kaggle_run.ipynb` | Any method, variant, dataset and subject range, on one or two GPUs. | [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/CuongDM1806/ssm-dan/blob/main/notebooks/kaggle_run.ipynb) |
| `notebooks/kaggle_openbmi_cache.ipynb` | Build the OpenBMI cache, with or without CAR. No GPU needed. | [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/CuongDM1806/ssm-dan/blob/main/notebooks/kaggle_openbmi_cache.ipynb) |
| `notebooks/kaggle_parity_check.ipynb` | Compare this code with the development code that produced the paper's results. | [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/CuongDM1806/ssm-dan/blob/main/notebooks/kaggle_parity_check.ipynb) |

To use `kaggle_run.ipynb`, edit the settings cell (`METHOD`, `VARIANT`, `DATASET`, `SUBJECTS`, `GPUS`, `MODE`), enable Internet, and run *Save Version → Save & Run All*. `MODE = 'paper'` launches the folds the way the reported run did. For BCI IV and Zhou2016 that is all folds in one process on one GPU; for OpenBMI and the session-specific RA runs it is one process per fold, two GPUs at a time. `MODE = 'per_fold'` always uses both GPUs. Folds that do not fit in the 12-hour limit are deferred. To finish them, attach the version's output (*Add Input → Your Work → Notebook Output*) and run again; finished folds are skipped.

## OpenBMI

Lee2019_MI comes as about 61 GB of MAT files. Each subject is therefore preprocessed once into `openbmi_SXX.npz` (about 20 MB). The cache records the preprocessing it was built with, and a cache built with different settings is never used.

```bash
python scripts/build_openbmi_cache.py --out openbmi_cache --workers 3              # paper (with CAR)
python scripts/build_openbmi_cache.py --out openbmi_cache_nocar --no-car --workers 3
export OPENBMI_CACHE_INPUT=$PWD/openbmi_cache:$PWD/openbmi_cache_nocar
```

On Kaggle, attach the cache Dataset (`openbmi-dataset-cuong` with CAR, `openbmi-nocar-cuong` without). `kaggle_run.ipynb` checks that all 54 subjects are present before training.

## Reproducibility

- All results in the paper use seed 0, on one NVIDIA T4 per fold, with PyTorch 2.7.1 and the package versions in `requirements.txt`. cuDNN runs in deterministic mode. Data-loader workers are fixed at 2, because S&R augmentation draws its random numbers in the workers.
- `tests/test_config_parity.py` checks that every run reported in the paper resolves to the configuration it was run with. That configuration is stored in `tests/fixtures/paper_configs.json`, reconstructed from the development code and Kaggle notebooks (or the `config.yaml` saved with the results). This test needs only PyYAML: `python -m pytest tests/test_config_parity.py`.
- `notebooks/kaggle_parity_check.ipynb` runs `tests/parity/probe.py` on this code and on the development code for the same fold. It compares the data splits, the initial weights, the weights after several training steps and the adapted test logits.
- TorchScript optimizes the selective scan after its first calls in a process, so the first fold of a process can differ from later folds in the last bits. To match the reported runs, launch folds as they were launched. `MODE = 'paper'` in the Kaggle notebook does this; locally, run all folds in one command for BCI IV/Zhou2016, and one command per fold for OpenBMI and `per_session_ra`.
- GPU kernels are not bit-reproducible across GPU models or driver versions. Re-runs on other hardware can differ slightly from the reported per-subject numbers.

## Baselines and credits

Each baseline is the authors' released network, re-trained under the protocol above:

- EEG Conformer (Song et al., 2023).
- ATCNet (Altaheri et al., 2023).
- TCFormer (Altaheri et al., 2025).
- EEGEncoder (Liao et al., 2025).
- SATrans-Net (Miao et al., 2025).
- RMETNet (Zhao et al., 2026).
- T-TIME (Li et al., 2024) and BFT (2026), from DeepTransferEEG.

Third-party code keeps its original notices in the file headers. The training framework and the convolution/TCN components derive from [TCFormer](https://github.com/altaheri/TCFormer) (MIT license; see `LICENSE`).

## Citation

```bibtex
@article{ssmdan,
  title   = {Domain-Adaptive Selective State-Space Learning for Cross-Subject Motor-Imagery EEG Decoding},
  journal = {Biomedical Signal Processing and Control},
  note    = {under review}
}
```
