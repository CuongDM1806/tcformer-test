# OpenBMI (Lee2019_MI), preprocessed for SSM-DAN

Data-only branch of the SSM-DAN repository (the code is on `main`). The OpenBMI notebook and
`python -m ssm_dan.launch` fetch this branch automatically when the files are not available yet.

`openbmi/openbmi_SXX.npz`, one file per subject (54 subjects, ~20 MB each):

| array | shape | content |
|---|---|---|
| `X1`, `y1` | (100, 62, 400) float32, (100,) int64 | session 1, offline motor-imagery run |
| `X2`, `y2` | (100, 62, 400) float32, (100,) int64 | session 2, offline motor-imagery run |
| `signature` | JSON string | preprocessing that produced the arrays |

Labels: 0 = left hand, 1 = right hand. Signals in microvolts, 62 EEG channels, preprocessed with the
`openbmi` settings of `configs/datasets.yaml` on `main`: 4-38 Hz band-pass, common average reference,
resampled to 100 Hz, 0-4 s after the cue. Training reads a file only if its `signature` matches the
configuration, so a different preprocessing (e.g. the `no_car` variant) rebuilds its own files from the
raw recordings (`python -m ssm_dan.data.openbmi_cache`).

Original recordings: M.-H. Lee et al., "EEG dataset and OpenBMI toolbox for three BCI paradigms: an
investigation into BCI illiteracy", GigaScience 8(5), 2019, data at GigaDB, doi:10.5524/100542.
