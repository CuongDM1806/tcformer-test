# Data

Every experiment reads its EEG data from this directory, `data/` at the repository root.
Set `SSM_DAN_DATA=/some/other/dir` to use another directory with the same layout.

**Rule.** A file that is already here is used as it is and is never downloaded again.
Only missing files are obtained. `python -m ssm_dan.launch` and the notebooks do this
before training (step 4 of the notebooks). `python -m ssm_dan.run` does it for BCI IV-2a/2b too.

| Dataset | What you provide | What the code does when files are missing |
|---|---|---|
| BCI IV-2a, 2b | the official competition zips: GDF recordings + true labels (unzipped or not) | builds the MAT files it reads; downloads the zips from bbci.de if they are not here |
| Zhou2016 | nothing (or the files) | data branch `zhou2016-data`, then the original figshare archive |
| OpenBMI | nothing (or the files) | data branch `openbmi-data` (CAR), then the raw recordings, preprocessed |

For Zhou2016 and OpenBMI, files with the expected names in the input directories are linked into
`data/` first. These are attached Kaggle Datasets or outputs (`/kaggle/input`), a mounted Google
Drive (`/content/drive/MyDrive`), `$SSM_DAN_INPUT`, and `~/mne_data`.
The BCI IV zips are also looked for there. The contents of `data/` are not tracked by git; only this
file is.

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

**What to do.** Put the zips in `data/`, as downloaded or unzipped, for example:

```
data/BCICIV_2a_gdf/            A01T.gdf ... A09E.gdf       (unzipped BCICIV_2a_gdf.zip)
data/BCICIV_2a_true_labels/    A01T.mat ... A09E.mat       (unzipped ds2a/true_labels.zip)
data/BCICIV_2b_gdf.zip                                     (or left zipped)
data/BCICIV_2b_true_labels.zip
```

Folder and zip names do not matter: the files are recognised by their names. Both label
archives are called `true_labels.zip` when downloaded, so keep them in different folders
or rename them. If they are not found, the code downloads them into `data/`.

**What the code builds.** Training reads the recordings with MOABB, which needs them in the
BNCI MAT format. `ssm_dan/data/bciciv.py` writes these files once, from the GDF files and the true
labels (about 1-2 minutes per dataset). Do not edit them:

```
data/MNE-bnci-data/database/data-sets/001-2014/   A01T.mat ... A09E.mat    (2a)
data/MNE-bnci-data/database/data-sets/004-2014/   B01T.mat ... B09E.mat    (2b)
```

Every file holds the subject's continuous signal of a day (2a) or of its sessions 1-3 / 4-5 (2b) in uV,
EEG and EOG channels at 250 Hz, with trial onsets and labels. Training then keeps the EEG channels,
cuts the trials and normalises them (`ssm_dan/data/loaders.py`, `configs/datasets.yaml`). On BCI
IV-2a, these files are identical to the BNCI Horizon 2020 files of data set 001-2014 used for the
paper: every field MOABB reads is the same, and the signal is the same bit for bit.

```bash
python -m ssm_dan.data.bciciv --dataset bcic2a        # build now (find the zips or download them)
python -m ssm_dan.data.bciciv --dataset bcic2b --no-download --search /path/with/zips
```

## Zhou2016 (`zhou2016`)

```
data/MNE-zhou-2016/subject_1/   1A.cnt 1B.cnt 2A.cnt 2B.cnt 3A.cnt 3B.cnt
...                 subject_4/  (24 files, Neuroscan CNT)
```

- Sources: the data branch [`zhou2016-data`](../../../tree/zhou2016-data) of this repository (its
  folders are those of `data/`), or the original archive https://ndownloader.figshare.com/files/3662952
  as MOABB downloads it. Its `data/S{k}_{session}{run}.cnt` become `subject_k/{session}{run}.cnt`.
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
- Sources: the data branch [`openbmi-data`](../../../tree/openbmi-data) of this repository (the paper's
  files; its `openbmi/` folder is `data/openbmi/`), or the raw recordings (GigaDB, doi:10.5524/100542,
  http://gigadb.org/dataset/100542), turned into these files by
  `python -m ssm_dan.data.openbmi_cache [--no-car] [--subjects 1-54]` or
  `python -m ssm_dan.launch --data --dataset openbmi [--no-car] --workers 3`.

## Kaggle and Colab

- **Kaggle.** Nothing to attach: the notebooks download the BCI IV zips from bbci.de and fetch the
  data branches. To avoid the download, upload the files as a Kaggle Dataset and attach it with
  *Add Input*. Kaggle unzips the archives, which works. The OpenBMI files a notebook built are saved
  in its output under `data/openbmi*`; attach that output next time.
- **Colab.** Put the files in Google Drive and mount it; `/content/drive/MyDrive` is searched.

Files found in the inputs are linked, not copied, when the file system allows it. Delete `data/`
(or a subfolder) to force everything in it to be obtained again.
