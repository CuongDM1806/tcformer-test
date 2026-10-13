"""Huấn luyện WDAN trên BCI IV 2a theo She et al. (TNSRE 2023), Algorithm 1 và mục II-E:
  target = session 2 (E) của subject đích (không nhãn khi train, đồng thời là tập test)
  source = session 1 (T) của 8 subject còn lại (có nhãn)

Mỗi bước (Algorithm 1):
  1) Cố định feature extractor, cập nhật critic n_critic lần: max L_wd - lambda * L_grad
  2) theta_c <- theta_c - a2 * grad L_cls
     theta_f <- theta_f - a2 * grad (L_cls + L_wd)        (lambda = 0 ở bước min)

Ví dụ:
  python train.py                         # chạy cả 9 subject
  python train.py --subjects 1 3 --epochs 100
  python train.py --synthetic --epochs 2  # chạy thử không cần dữ liệu
"""
import argparse, csv, json, time
import numpy as np
import torch
import torch.nn.functional as F

from model import FeatureExtractor, Classifier, Critic, gradient_penalty


def cohen_kappa(y, p, n_cls=4):
    cm = np.zeros((n_cls, n_cls))
    for a, b in zip(y, p):
        cm[a, b] += 1
    n = cm.sum(); po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe)


@torch.no_grad()
def evaluate(G, C, X, y, dev, bs=256):
    G.eval(); C.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        preds.append(C(G(xb)).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return (p == y).mean(), cohen_kappa(y, p)


def batches(n, bs, rng):
    """Vòng lặp vô hạn trả về chỉ số mini-batch (xáo lại mỗi lượt)."""
    bs = min(bs, n)
    while True:
        idx = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            yield idx[i:i + bs]


def run_subject(Xs, ys, Xt, yt, args, dev, log_prefix=""):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    G = FeatureExtractor(n_ch=Xs.shape[1], n_times=Xs.shape[2], d=args.d,
                         win_sec=args.win_sec).to(dev)
    C = Classifier(G.out_dim, args.cls_hid, drop=args.dropout).to(dev)
    D = Critic(G.out_dim, args.critic_hid).to(dev)
    opt_c = torch.optim.Adam(C.parameters(), lr=args.lr)
    opt_f = torch.optim.Adam(G.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_critic)

    Xs_t, ys_t = torch.from_numpy(Xs), torch.from_numpy(ys)
    Xt_t = torch.from_numpy(Xt)
    it_s, it_t = batches(len(Xs), args.bs, rng), batches(len(Xt), args.bs, rng)
    steps = max(1, len(Xs) // args.bs)

    hist, best = [], (0.0, 0.0, -1)
    for ep in range(args.epochs):
        G.train(); C.train(); D.train()
        lc_sum = lw_sum = 0.0
        for _ in range(steps):
            si, ti = next(it_s), next(it_t)
            xs, ysb = Xs_t[si].to(dev), ys_t[si].to(dev)
            xt = Xt_t[ti].to(dev)

            # 1) Cố định feature extractor, cập nhật critic: max L_wd - lambda * L_grad (Eq. 7)
            with torch.no_grad():
                fs_d, ft_d = G(xs), G(xt)
            for _ in range(args.n_critic):
                wd = D(fs_d).mean() - D(ft_d).mean()
                gp = gradient_penalty(D, fs_d, ft_d)
                opt_d.zero_grad(); (-wd + args.gp * gp).backward(); opt_d.step()

            # 2) Cập nhật classifier theo L_cls và feature extractor theo L_cls + mu * L_wd (Eq. 9)
            fs, ft = G(xs), G(xt)
            lc = F.cross_entropy(C(fs), ysb)
            lw = D(fs).mean() - D(ft).mean()
            opt_c.zero_grad(); opt_f.zero_grad(); D.zero_grad()
            (lc + args.mu * lw).backward()
            opt_c.step(); opt_f.step()
            lc_sum += lc.item(); lw_sum += lw.item()

        acc, kap = evaluate(G, C, Xt, yt, dev)
        hist.append((ep + 1, lc_sum / steps, lw_sum / steps, acc, kap))
        if acc > best[0]:
            best = (acc, kap, ep + 1)
        if (ep + 1) % args.log_every == 0 or ep == 0:
            print(f"{log_prefix} ep {ep+1:3d} | CE {lc_sum/steps:.3f} | W {lw_sum/steps:.3f} "
                  f"| test acc {acc*100:.2f} kappa {kap:.3f}", flush=True)
    last_acc, last_kap = hist[-1][3], hist[-1][4]
    return dict(last_acc=last_acc, last_kappa=last_kap,
                best_acc=best[0], best_kappa=best[1], best_epoch=best[2]), hist


def summarize(name, acc, kap):
    # Cách báo cáo của bài: mean ± std (sample std, ddof=1) và trung bình Cohen's kappa từng subject
    return (f"{name}: {acc.mean()*100:.2f} ± {acc.std(ddof=1)*100:.2f} | "
            f"kappa {kap.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="alpha_2 (bài: 0.0005)")
    ap.add_argument("--lr_critic", type=float, default=5e-4, help="alpha_1 (bài không nói)")
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--gp", type=float, default=10.0, help="lambda của gradient penalty")
    ap.add_argument("--mu", type=float, default=1.0, help="trọng số L_wd (Algorithm 1: 1)")
    ap.add_argument("--d", type=int, default=16, help="số bộ lọc không gian mỗi dải")
    ap.add_argument("--win_sec", type=float, default=1.0, help="cửa sổ variance layer (bài: 1)")
    ap.add_argument("--cls_hid", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--critic_hid", type=int, default=64)
    ap.add_argument("--zero_phase", action="store_true", help="Lọc Butterworth hai chiều thay vì nhân quả")
    ap.add_argument("--ea", action="store_true", help="Bật Euclidean Alignment (không có trong bài)")
    ap.add_argument("--no_da", action="store_true", help="Baseline: tắt domain adaptation (mu=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--cache", default="bci2a_she2023_cache.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true", help="Dữ liệu ngẫu nhiên để kiểm tra code")
    args = ap.parse_args()
    if args.no_da:
        args.mu = 0.0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} | config: {json.dumps(vars(args))}")

    if args.synthetic:
        rng = np.random.default_rng(0)
        data = {s: {k: (rng.standard_normal((48, 22, 1000)).astype(np.float32),
                        rng.integers(0, 4, 48)) for k in ("T", "E")} for s in range(1, 10)}
        make = lambda t: (np.concatenate([data[s]["T"][0] for s in data if s != t]),
                          np.concatenate([data[s]["T"][1] for s in data if s != t]),
                          data[t]["E"][0], data[t]["E"][1])
    else:
        from data import load_bci2a, make_split
        cache = args.cache.replace(".npz", "_zp.npz") if args.zero_phase else args.cache
        data = load_bci2a(cache, zero_phase=args.zero_phase)
        make = lambda t: make_split(data, t, ea=args.ea)

    rows = []
    for t in args.subjects:
        Xs, ys, Xt, yt = make(t)
        print(f"\n=== Target A0{t}: source {Xs.shape}, target/test {Xt.shape} ===")
        t0 = time.time()
        res, hist = run_subject(Xs, ys, Xt, yt, args, dev, log_prefix=f"[A0{t}]")
        res = {"subject": f"A0{t}", **res, "minutes": (time.time() - t0) / 60}
        rows.append(res)
        print(f"[A0{t}] last-epoch acc {res['last_acc']*100:.2f} | "
              f"best-epoch acc {res['best_acc']*100:.2f} (ep {res['best_epoch']})")
        np.savetxt(f"{args.out}_A0{t}_history.csv", np.array(hist), delimiter=",",
                   header="epoch,ce,wdist,test_acc,test_kappa", comments="")

    get = lambda k: np.array([r[k] for r in rows])
    print("\n================ KẾT QUẢ ================")
    for r in rows:
        print(f"{r['subject']}: last {r['last_acc']*100:6.2f} | best {r['best_acc']*100:6.2f}")
    if len(rows) > 1:
        print(summarize("Trung bình last-epoch", get("last_acc"), get("last_kappa")))
        print(summarize("Trung bình best-epoch", get("best_acc"), get("best_kappa"))
              + "  (lạc quan: chọn epoch theo tập test)")
    print("Bài gốc (Table II): 77.60 ± 10.85 | kappa 0.6951")

    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
