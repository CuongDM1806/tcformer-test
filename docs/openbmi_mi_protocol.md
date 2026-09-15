# OpenBMI MI protocol for Full-Mamba LOSO

## Dataset

- MOABB dataset: `Lee2019_MI` (OpenBMI MI).
- 54 healthy subjects, two recording sessions on different days.
- 62 EEG channels sampled natively at 1000 Hz.
- Binary motor imagery: left-hand versus right-hand grasping.
- Each labeled offline run has 100 balanced trials; each MI cue lasts 4 seconds.
- MOABB's default MI loader includes the labeled offline/train run and excludes
  the online feedback/test run because its MI trial labels are unavailable.

Sources:

- Dataset paper: https://doi.org/10.1093/gigascience/giz002
- Official toolbox: https://github.com/PatternRecognition/OpenBMI
- MOABB implementation: https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/Lee2019.py
- Cross-subject pipeline: https://github.com/gzoumpourlis/Ensemble-MI

## Preprocessing choice

The original OpenBMI analysis commonly uses 8-30 Hz and the 1.0-3.5 second
online-prediction interval. This branch instead follows the published
cross-subject Ensemble-MI preprocessing recipe because the experiment is LOSO:

1. Keep all 62 EEG channels and convert volts to microvolts.
2. Band-pass filter at 4-38 Hz.
3. Apply common-average reference (CAR).
4. Resample to 100 Hz.
5. Extract exactly one 0.0-4.0 second window from each MI cue (400 samples).

## Split and leakage policy

For each target-subject fold:

- Source session 1: labeled training data.
- Source session 2: labeled source-only validation data.
- Target session 1: unlabeled HADA target data and the RA reference.
- Target session 2: primary test data.

RA is fitted separately for every subject using only that subject's session 1,
then applied to both sessions. Labels are not used by RA.

IM-TTA is enabled by default as requested. It adapts BatchNorm affine parameters
using all unlabeled target session-2 EEG before scoring, so reported results must
be described as **transductive IM-TTA**, not strict unseen-test evaluation.

## Notebook scope

The Colab and Kaggle notebooks default to subjects 1-10 for a practical smoke
LOSO experiment. The upstream OpenBMI MI download is approximately 61 GB for all
54 subjects. For a full experiment, change `SUBJECT_IDS` in the notebook to:

```python
SUBJECT_IDS = list(range(1, 55))
```
