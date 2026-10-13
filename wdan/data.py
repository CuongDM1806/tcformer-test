"""Tải và tiền xử lý BCI Competition IV 2a (4 lớp) theo She et al. (TNSRE 2023), mục II-B.

Kết quả là dict theo subject:
    data[subj] = {"T": (X, y), "E": (X, y)}
X: (n_trials, 22, 1000) float32, y: (n_trials,) int64 với 0..3
(left_hand, right_hand, feet, tongue).

Theo bài gốc:
  - Bỏ 3 kênh EOG, giữ 22 kênh EEG, 250 Hz, dùng toàn bộ trial.
  - Lọc Butterworth bậc 4, dải 8-32 Hz.
  - Exponential moving standardization (EMS).
  - Mỗi trial lấy từ giây thứ 2 đến giây thứ 6 (0-4 s sau cue).
GIẢ ĐỊNH (bài không nói): lọc nhân quả (causal) và EMS chạy trên tín hiệu liên tục
của từng run, với tham số mặc định của braindecode (factor_new=1e-3,
init_block_size=1000, eps=1e-4), sau khi đổi đơn vị sang microvolt.
"""
import os
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter

CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
SFREQ = 250


def _session_key(name: str) -> str:
    """MOABB đổi tên session qua các phiên bản: 'session_T'/'session_E' (cũ)
    hoặc '0train'/'1test' (mới)."""
    n = str(name).lower()
    if "train" in n or n.endswith("_t"):
        return "T"
    if "test" in n or n.endswith("_e"):
        return "E"
    raise ValueError(f"Không nhận ra tên session: {name}")


def bandpass(X: np.ndarray, fmin=8.0, fmax=32.0, order=4, fs=SFREQ, zero_phase=False):
    """Butterworth bậc 4 theo bài. X: (..., n_times)."""
    sos = butter(order, [fmin, fmax], btype="bandpass", fs=fs, output="sos")
    f = sosfiltfilt if zero_phase else sosfilt
    return f(sos, X, axis=-1)


def _ewm_mean(x: np.ndarray, alpha: float) -> np.ndarray:
    """Trung bình trượt mũ có hiệu chỉnh (giống pandas ewm(alpha).mean(), adjust=True)."""
    a = [1.0, -(1.0 - alpha)]
    num = lfilter([1.0], a, x, axis=-1)
    den = lfilter([1.0], a, np.ones(x.shape[-1]))
    return num / den


def exponential_moving_standardize(X: np.ndarray, factor_new=1e-3,
                                   init_block_size=1000, eps=1e-4) -> np.ndarray:
    """EMS theo từng kênh, tương đương braindecode.preprocessing.exponential_moving_standardize.
    X: (n_channels, n_times) tín hiệu liên tục."""
    mean = _ewm_mean(X, factor_new)
    demeaned = X - mean
    var = _ewm_mean(demeaned ** 2, factor_new)
    out = demeaned / np.maximum(eps, np.sqrt(var))
    if init_block_size:
        block = X[:, :init_block_size]
        mu = block.mean(axis=-1, keepdims=True)
        sd = block.std(axis=-1, keepdims=True)
        out[:, :init_block_size] = (block - mu) / np.maximum(eps, sd)
    return out


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """Euclidean Alignment (He & Wu, 2019). KHÔNG có trong bài gốc, chỉ bật khi muốn thử (--ea)."""
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    R = covs.mean(0)
    w, V = np.linalg.eigh(R)
    R_inv_sqrt = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
    return np.einsum("cd,ndt->nct", R_inv_sqrt, X).astype(np.float32)


def _events(raw, event_id):
    import mne
    if "stim" in raw.ch_names:
        ev = mne.find_events(raw, stim_channel="stim", shortest_event=0, verbose=False)
    else:
        ev, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    inv = {v: k for k, v in event_id.items()}
    keep = np.isin(ev[:, 2], list(inv))
    return ev[keep, 0], [inv[c] for c in ev[keep, 2]]


def preprocess_run(raw, event_id, interval, zero_phase=False):
    """Lọc + EMS trên tín hiệu liên tục của một run, rồi cắt trial theo interval của dataset."""
    raw = raw.copy().pick("eeg")                              # 22 kênh EEG, bỏ EOG
    x = raw.get_data() * 1e6                                  # V -> microvolt
    x = bandpass(x, zero_phase=zero_phase)
    x = exponential_moving_standardize(x)
    onsets, labels = _events(raw, event_id)
    start, stop = int(round(interval[0] * SFREQ)), int(round(interval[1] * SFREQ))
    X = np.stack([x[:, o + start:o + stop] for o in onsets]).astype(np.float32)
    y = np.array([CLASSES.index(l) for l in labels], dtype=np.int64)
    return X, y


def load_bci2a(cache="bci2a_she2023_cache.npz", zero_phase=False):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["data"].item()

    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001()
    assert abs(dataset.interval[1] - dataset.interval[0] - 4.0) < 1e-6, dataset.interval
    data = {}
    for subj in dataset.subject_list:
        sessions = dataset.get_data(subjects=[subj])[subj]
        parts = {"T": [], "E": []}
        for sess_name, runs in sessions.items():
            for raw in runs.values():
                assert int(raw.info["sfreq"]) == SFREQ
                parts[_session_key(sess_name)].append(
                    preprocess_run(raw, dataset.event_id, dataset.interval, zero_phase))
        data[subj] = {k: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v]))
                      for k, v in parts.items()}
        print(f"S{subj:02d}: T={data[subj]['T'][0].shape}, E={data[subj]['E'][0].shape}")
    np.savez(cache, data=np.array(data, dtype=object))
    return data


def make_split(data, target, ea=False):
    """Giao thức theo bài (mục II-E): test A0x -> target = session 2 (E) của A0x
    (không nhãn khi train, đồng thời là tập test), source = session 1 (T) của 8 subject còn lại."""
    prep = (lambda X: euclidean_alignment(X)) if ea else (lambda X: X)
    Xs, ys = [], []
    for s, d in data.items():
        if s == target:
            continue
        Xs.append(prep(d["T"][0]))   # EA (nếu bật) làm riêng từng subject
        ys.append(d["T"][1])
    Xs, ys = np.concatenate(Xs), np.concatenate(ys)
    Xt, yt = data[target]["E"]
    return Xs, ys, prep(Xt), yt
