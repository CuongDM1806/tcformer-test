# Data

Every experiment reads its EEG data from this directory, `data/` at the repository root.
Set `SSM_DAN_DATA=/some/other/dir` to use another directory with the same layout.

**Rule.** A file that is already here is used as it is and is never downloaded again.
Only missing files are obtained. `python -m ssm_dan.launch` and the notebooks do this
before training (step 4 of the notebooks), in this order:

1. files already in `data/`;
2. files with the expected names found in the input directories are linked into `data/`.
   These are attached Kaggle Datasets or notebook outputs (`/kaggle/input`), a mounted
   Google Drive (`/content/drive/MyDrive`), the directories in `$SSM_DAN_INPUT`, and `~/mne_data`;
3. the data branch `<dataset>-data` of this repository, which holds the files in this same layout.
   It exists for `bcic2a` (0.78 GB), `zhou2016` (177 MB) and `openbmi` (CAR only, ~1 GB); `bcic2b` will be
   added when its server can be reached again;
4. the original servers: MOABB download for BCI IV-2a/2b and Zhou2016, retried up to 10 times.
   OpenBMI is downloaded and preprocessed.

The contents of `data/` are not tracked by git; only this file is.

Check what is there and what is missing:

```bash
# what is there, what is missing (all datasets, or one)
python -m ssm_dan.data.files --check
python -m ssm_dan.data.files --dataset bcic2a --check
# obtain what is missing (steps 2-4)
python -m ssm_dan.launch --data --dataset bcic2a
```

## Layout

The paths are MOABB's own, so MOABB reads the files directly without contacting any
server. Names and folders must match exactly (case included).

```
data/
├── MNE-bnci-data/database/data-sets/
│   ├── 001-2014/        A01T.mat ... A09E.mat
│   └── 004-2014/        B01T.mat ... B09E.mat
├── MNE-zhou-2016/
│   ├── subject_1/       1A.cnt ... 3B.cnt
│   ├── ...
│   └── subject_4/
├── openbmi/             openbmi_S01.npz ... openbmi_S54.npz
└── openbmi_nocar/       openbmi_S01.npz ... openbmi_S54.npz
```

| Folder | Dataset | Files |
|---|---|---|
| `001-2014/` | BCI IV-2a | `A01T.mat`, `A01E.mat`, ..., `A09T.mat`, `A09E.mat` (18) |
| `004-2014/` | BCI IV-2b | `B01T.mat`, `B01E.mat`, ..., `B09T.mat`, `B09E.mat` (18) |
| `subject_1/` ... `subject_4/` | Zhou2016 | `1A.cnt`, `1B.cnt`, `2A.cnt`, `2B.cnt`, `3A.cnt`, `3B.cnt` in each (24) |
| `openbmi/` | OpenBMI, paper preprocessing (CAR) | `openbmi_S01.npz` ... `openbmi_S54.npz` (54) |
| `openbmi_nocar/` | OpenBMI without CAR (`no_car` variant) | same names (54) |

### BCI Competition IV-2a (`bcic2a`)

- Files: `A{subject}{session}.mat` with subject `01`-`09` and session `T` (training day) or
  `E` (evaluation day), in `data/MNE-bnci-data/database/data-sets/001-2014/`.
- About 37-44 MB each, about 0.8 GB in total.
- These are the original BNCI Horizon 2020 files of data set 001-2014 (MATLAB, labels included),
  as served at `http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat`. The GDF files
  of the competition website are a different format and are not read.
- MOABB 0.5 calls this dataset `BNCI2014001` (`BNCI2014_001` in newer versions); the files are the same.

### BCI Competition IV-2b (`bcic2b`)

- Files: `B{subject}{T|E}.mat` with subject `01`-`09`, in `.../data-sets/004-2014/`.
- `T` holds sessions 1-3 and `E` sessions 4-5.
- These are the BNCI Horizon 2020 files of data set 004-2014, at
  `http://bnci-horizon-2020.eu/database/data-sets/004-2014/B01T.mat`.

### Zhou2016 (`zhou2016`)

- Files: `data/MNE-zhou-2016/subject_{1..4}/{session}{run}.cnt` with session `1`-`3` and
  run `A` or `B` (Neuroscan CNT).
- MOABB downloads one zip (figshare file 3662952). The zip contains `data/S{subject}_{session}{run}.cnt`,
  which MOABB renames to this layout. To place the files by hand, rename them the same way:
  `S3_2B.cnt` becomes `subject_3/2B.cnt`.
- A subject counts as present only when its folder holds all six files.

### OpenBMI / Lee2019_MI (`openbmi`)

Training does not read the 61 GB of raw MAT files. It reads one preprocessed file per subject:

- `data/openbmi/openbmi_S{01..54}.npz` (about 20 MB each, about 1.07 GB in total), a NumPy archive with:

  | array | shape, dtype | content |
  |---|---|---|
  | `X1` | (100, 62, 400) float32 | session 1, offline MI run, microvolts, 62 EEG channels, 4 s at 100 Hz |
  | `y1` | (100,) int64 | labels, 0 = left hand, 1 = right hand |
  | `X2`, `y2` | same | session 2 |
  | `signature` | JSON string | preprocessing used to build the arrays |

- The paper's files (CAR) have the signature
  `{"common_average_reference": true, "high_cut": 38.0, "low_cut": 4.0, "sfreq": 100, "start": 0.0, "trial_duration": 4.0}`.
  The `no_car` variant reads `data/openbmi_nocar/` and needs `"common_average_reference": false`
  with the other values unchanged.
- A file is used only if its signature matches the configuration (`openbmi` block of
  `configs/datasets.yaml`). After a preprocessing change, the files must be rebuilt.
- The paper's files are on the branch `openbmi-data`:
  `git clone --depth 1 --branch openbmi-data https://github.com/CuongDM1806/ssm-dan.git`.
  Its `openbmi/` folder is exactly `data/openbmi/`.
- To build the files from the raw recordings (raw MAT files are deleted after each subject):
  `python -m ssm_dan.data.openbmi_cache [--no-car] [--subjects 1-54]`, or
  `python -m ssm_dan.launch --data --dataset openbmi [--no-car] --workers 3`.

## Bringing your own copy

- **Same machine.** Copy or move the files into the layout above, or set `SSM_DAN_DATA` to a
  directory that already has it. A MOABB folder such as `~/mne_data` already has the BCI IV and
  Zhou2016 layout. It is searched automatically, and its files are linked into `data/`.
- **Kaggle.** Upload the files as a Kaggle Dataset (any folder structure, as long as the file names
  and their parent folder names are kept, e.g. `001-2014/A01T.mat`, `subject_3/2B.cnt`) and attach it
  with *Add Input*. Step 4 of the notebook links the files from `/kaggle/input` into `data/`.
  The output of an earlier notebook also works: for example, the OpenBMI files a notebook built
  are saved in its output under `data/openbmi*`.
- **Colab.** Put the files in Google Drive and mount it; `/content/drive/MyDrive` is searched.

Files are linked, not copied, when the file system allows it. Delete `data/` (or a subfolder) to
force everything in it to be obtained again.
