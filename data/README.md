# Data

Every experiment reads its EEG data from this directory, `data/` at the repository root.
Only `data/openbmi/` is stored in git (the paper's preprocessed OpenBMI files); everything else in
`data/` is downloaded or built and is not tracked. Set `SSM_DAN_DATA=/some/other/dir` to use another directory with the same layout.

## Quick start: download, unzip, run

Download these files and unzip each into the folder shown, then run. The code builds what it needs
from them and never downloads them again.

| Download | Unzip into |
|---|---|
| [BCICIV_2a_gdf.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip) (420 MB) | `data/BCICIV_2a/gdf/` |
| [2a true labels](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip) (`true_labels.zip`) | `data/BCICIV_2a/true_labels/` |
| [BCICIV_2b_gdf.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip) (216 MB) | `data/BCICIV_2b/gdf/` |
| [2b true labels](https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip) (`true_labels.zip`) | `data/BCICIV_2b/true_labels/` |
| Zhou2016: nothing to download | downloaded automatically by MOABB from figshare into `data/MNE-zhou-2016/` |
| OpenBMI: nothing to download | already in `data/openbmi/` (in git, ~1 GB) |

```bash
cd data
mkdir -p BCICIV_2a/gdf BCICIV_2a/true_labels BCICIV_2b/gdf BCICIV_2b/true_labels
unzip BCICIV_2a_gdf.zip -d BCICIV_2a/gdf
unzip 2a_true_labels.zip -d BCICIV_2a/true_labels    # the 2a true_labels.zip, renamed
unzip BCICIV_2b_gdf.zip -d BCICIV_2b/gdf
unzip 2b_true_labels.zip -d BCICIV_2b/true_labels    # the 2b true_labels.zip, renamed
```

**On Kaggle**, attach BCI IV and Zhou2016 as Kaggle Datasets (*Add Input*, see
[Kaggle and Colab](#kaggle-and-colab)); OpenBMI needs nothing. If files are missing, the notebooks and
`python -m ssm_dan.launch` get them by themselves (step 4 of the notebooks), in this order:

| Dataset | Missing files are taken from |
|---|---|
| BCI IV-2a, 2b | input directories (zipped or unzipped), then bbci.de. `mat/` is always built from `gdf/` + `true_labels/` and never stored |
| Zhou2016 | input directories, then the original figshare archive (MOABB) |
| OpenBMI | in git with CAR (`data/openbmi/`); `no_car`: input directories, then the raw recordings, preprocessed |

Input directories are attached Kaggle Datasets or outputs (`/kaggle/input`), a mounted Google
Drive (`/content/drive/MyDrive`), `$SSM_DAN_INPUT`, and `~/mne_data`. A file already in `data/` is
used as it is. Apart from `data/openbmi/` and this file, the contents of `data/` are not tracked by git.

```bash
# what is there, what is missing (all datasets, or one)
python -m ssm_dan.data.files --check
python -m ssm_dan.data.files --dataset bcic2a --check
# obtain what is missing
python -m ssm_dan.launch --data --dataset bcic2a
```

## BCI Competition IV-2a and 2b (`bcic2a`, `bcic2b`)

Only the official files of the BCI Competition IV are used
([download page](https://www.bbci.de/competition/iv/download/), [results page](https://www.bbci.de/competition/iv/results/)):

| File | Content | Link |
|---|---|---|
| `BCICIV_2a_gdf.zip` (420 MB) | `A01T.gdf` ... `A09E.gdf`: 9 subjects, training (T) and evaluation (E) day | https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip |
| true labels 2a | `A01T.mat` ... `A09E.mat` (a few kB each: `classlabel`) | https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip |
| `BCICIV_2b_gdf.zip` (216 MB) | `B0101T.gdf` ... `B0905E.gdf`: 9 subjects, sessions 1-3 (T) and 4-5 (E) | https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip |
| true labels 2b | `B0101T.mat` ... `B0905E.mat` | https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip |

The GDF files hide the labels of the evaluation sessions; the true-label files hold the labels of
all sessions (in the training sessions they are checked against the GDF cue events).

**Layout.** Each data set has one folder with three subfolders:

```
data/BCICIV_2a/
├── gdf/            A01T.gdf ... A09E.gdf        unzipped BCICIV_2a_gdf.zip (18 files)
├── true_labels/    A01T.mat ... A09E.mat        unzipped ds2a/true_labels.zip (18 files)
└── mat/            A01T.mat ... A09E.mat        built by the code; training reads these
data/BCICIV_2b/
├── gdf/            B0101T.gdf ... B0905E.gdf    unzipped BCICIV_2b_gdf.zip (45 files)
├── true_labels/    B0101T.mat ... B0905E.mat    unzipped ds2b/true_labels.zip (45 files)
└── mat/            B01T.mat ... B09E.mat        built by the code (T: sessions 1-3, E: 4-5)
```

**What to do.** Unzip the two archives into `gdf/` and `true_labels/` (see [Quick start](#quick-start-download-unzip-run)).
The files are recognised by their names, so the zips can also be left unextracted, there or in an
attached input (on Kaggle: a Dataset added with *Add Input*). If they are missing, the code
downloads both zips from bbci.de and extracts them into these folders.

**What the code builds.** Training reads the recordings with MOABB, which needs them in the BNCI
MAT format. When `mat/` is incomplete, `ssm_dan/data/bciciv.py` builds it from `gdf/` and
`true_labels/` (about 1-2 minutes per data set); MOABB is then pointed at `mat/` and never
downloads anything. Do not edit these files. Delete `mat/` to build it again.

Every built file holds the subject's continuous signal of a day (2a) or of its sessions 1-3 / 4-5
(2b) in uV, EEG and EOG channels at 250 Hz, with trial onsets and labels. Training then keeps the
EEG channels, cuts the trials and normalises them (`ssm_dan/data/loaders.py`,
`configs/datasets.yaml`). On BCI IV-2a, the built files are identical to the BNCI Horizon 2020
files of data set 001-2014 used for the paper: every field MOABB reads is the same, and the signal
is the same bit for bit.

```bash
python -m ssm_dan.data.bciciv --dataset bcic2a        # build now (find the zips or download them)
python -m ssm_dan.data.bciciv --dataset bcic2b --no-download --search /path/with/files
```

## Zhou2016 (`zhou2016`)

```
data/MNE-zhou-2016/subject_1/   1A.cnt 1B.cnt 2A.cnt 2B.cnt 3A.cnt 3B.cnt
...                 subject_4/  (24 files, Neuroscan CNT)
```

- Source: the original archive https://ndownloader.figshare.com/files/3662952, downloaded by MOABB
  when the files are missing. Its `data/S{k}_{session}{run}.cnt` become `subject_k/{session}{run}.cnt`.
- A subject counts as present only when its folder holds all six files.

## OpenBMI / Lee2019_MI (`openbmi`)

Training does not read the 61 GB of raw MAT files. It reads one preprocessed file per subject:

```
data/openbmi/         openbmi_S01.npz ... openbmi_S54.npz    (paper preprocessing, CAR)
data/openbmi_nocar/   openbmi_S01.npz ... openbmi_S54.npz    (no_car variant)
```

- Each file (about 20 MB, 1.07 GB in total) is a NumPy archive with:

  | array | shape, dtype | content |
  |---|---|---|
  | `X1` | (100, 62, 400) float32 | session 1, offline MI run, microvolts, 62 EEG channels, 4 s at 100 Hz |
  | `y1` | (100,) int64 | labels, 0 = left hand, 1 = right hand |
  | `X2`, `y2` | same | session 2 |
  | `signature` | JSON string | preprocessing used to build the arrays |

- The paper's files (CAR) have the signature
  `{"common_average_reference": true, "high_cut": 38.0, "low_cut": 4.0, "sfreq": 100, "start": 0.0, "trial_duration": 4.0}`.
  The `no_car` variant needs `"common_average_reference": false` with the other values unchanged.
  A file is used only if its signature matches the configuration (`openbmi` block of
  `configs/datasets.yaml`).
- Sources: `data/openbmi/` is part of the repository (the paper's files; a copy is kept in the
  branch [`openbmi-data`](../../../tree/openbmi-data)). `data/openbmi_nocar/`, or `data/openbmi/` again
  if deleted, is built from the raw recordings (GigaDB, doi:10.5524/100542,
  http://gigadb.org/dataset/100542; 108 MAT files, ~61 GB, downloaded a few subjects at a time and
  deleted once preprocessed) by
  `python -m ssm_dan.data.openbmi_cache [--no-car] [--subjects 1-54]` or
  `python -m ssm_dan.launch --data --dataset openbmi [--no-car] --workers 3`.

## Kaggle and Colab

- **Kaggle.** Make one Kaggle Dataset per data set (*Datasets -> New Dataset*, upload the files;
  Kaggle unzips them, which works) and attach it to the notebook with *Add Input -> Your Datasets*:

  | Kaggle Dataset | Files to upload |
  |---|---|
  | BCI IV-2a | `BCICIV_2a_gdf.zip` and the 2a `true_labels.zip`, renamed e.g. `2a_true_labels.zip` |
  | BCI IV-2b | `BCICIV_2b_gdf.zip` and the 2b `true_labels.zip`, renamed e.g. `2b_true_labels.zip` |
  | Zhou2016 | a zip of `data/MNE-zhou-2016/` (keep the `subject_1` ... `subject_4` folders) |

  The files are found under `/kaggle/input` by their names. Both label archives are called
  `true_labels.zip` on bbci.de; rename them, or keep them in different Datasets. OpenBMI needs no
  Dataset (it is cloned with the code). Without an input, the notebook downloads the files from bbci.de
  and figshare itself. The OpenBMI files a notebook built are saved in its output under
  `data/openbmi*`; attach that output next time.
- **Colab.** Put the files in Google Drive and mount it; `/content/drive/MyDrive` is searched.

Files found in the inputs are linked, not copied, when the file system allows it. Delete `data/`
(or a subfolder) to force everything in it to be obtained again.
