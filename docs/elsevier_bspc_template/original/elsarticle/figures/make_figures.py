#!/usr/bin/env python3
"""Draw the manuscript's architecture figures as print-ready vector PDFs.

All coordinates are in millimetres and every figure is drawn at its final
printed width (180 mm, Elsevier double-column), so font sizes are true sizes.
Run:  python make_figures.py            -> PDFs (and PNG previews with --png)
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURE_DIR = Path(__file__).resolve().parent
MM = 1 / 25.4
WIDTH = 180.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "cmsy10",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

# Muted, colour-blind-safe palette (Okabe-Ito hues as tints).
INK = "#1F2A36"
MUTED = "#5B6773"
LINE = "#2B3A4A"
DIMS = "#3D6E9E"
STYLES = {
    "data":    dict(fc="#E3EEF8", ec="#3D6E9E"),
    "conv":    dict(fc="#E2F1EA", ec="#2E8466"),
    "ssm":     dict(fc="#ECE6F5", ec="#6A4FA3"),
    "fuse":    dict(fc="#FCEBD9", ec="#C2701E"),
    "adapt":   dict(fc="#FBF3D5", ec="#AF8A12"),
    "loss":    dict(fc="#F8E3E1", ec="#B5473F"),
    "neutral": dict(fc="#FFFFFF", ec="#8A96A3"),
    "locked":  dict(fc="#EEF0F2", ec="#8A96A3"),
}
ACCENT = {k: v["ec"] for k, v in STYLES.items()}

TITLE_PT = 7.0
BODY_PT = 6.0
SMALL_PT = 5.5
LW = 0.6


def new_figure(height, y0=0.0):
    """Figure showing the drawing band y0 .. y0 + height (mm)."""
    fig = plt.figure(figsize=(WIDTH * MM, height * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(y0, y0 + height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


class Box:
    """Rounded node; x, y is the lower-left corner in mm."""

    def __init__(self, ax, x, y, w, h, title, body=(), kind="neutral",
                 title_pt=TITLE_PT, body_pt=BODY_PT, radius=1.0, lw=LW, ls="-",
                 dims=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        style = STYLES[kind]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=style["fc"], ec=style["ec"], lw=lw, ls=ls, zorder=2))
        lines = ([(t, title_pt, "bold", INK) for t in title.split("\n")]
                 if title else [])
        lines += [(b, body_pt, "normal", INK) for b in body]
        if dims:
            lines.append((dims, SMALL_PT, "normal", DIMS))
        # Distribute text lines around the vertical centre.
        heights = [pt * 0.3528 * 1.32 for _, pt, _, _ in lines]
        cursor = y + h / 2 + sum(heights) / 2
        for (text, pt, weight, color), lh in zip(lines, heights):
            cursor -= lh
            ax.text(x + w / 2, cursor + lh * 0.5, text, fontsize=pt,
                    fontweight=weight, color=color, ha="center", va="center",
                    zorder=3, linespacing=1.1)

    # Anchors -----------------------------------------------------------
    def l(self, f=0.5):
        return (self.x, self.y + f * self.h)

    def r(self, f=0.5):
        return (self.x + self.w, self.y + f * self.h)

    def t(self, f=0.5):
        return (self.x + f * self.w, self.y + self.h)

    def b(self, f=0.5):
        return (self.x + f * self.w, self.y)

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


def panel(ax, x, y, w, h, label, color="#6B7885", fill="#F5F7F9", label_pt=6.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        fc=fill, ec=color, lw=0.5, zorder=0))
    if label:
        ax.text(x + 2.0, y + h - 1.6, label, fontsize=label_pt, fontweight="bold",
                color=color, ha="left", va="top", zorder=1)


def arrow(ax, pts, color=LINE, lw=LW, ls="-", head=True, z=1.5):
    """Orthogonal polyline with a solid arrow head on the final segment."""
    xs, ys = zip(*pts)
    ax.add_line(Line2D(xs, ys, color=color, lw=lw, ls=ls,
                       solid_capstyle="butt", dash_capstyle="butt", zorder=z))
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        dx, dy = x1 - x0, y1 - y0
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        start = (x1 - dx / norm * 1.6, y1 - dy / norm * 1.6)
        ax.add_patch(FancyArrowPatch(
            start, (x1, y1), arrowstyle="-|>,head_length=4.2,head_width=1.7",
            mutation_scale=1, color=color, lw=0,
            shrinkA=0, shrinkB=0, zorder=z + 0.1))


def label(ax, x, y, text, pt=SMALL_PT, color=MUTED, ha="center", va="center",
          weight="normal", style="normal", z=4, bg=None):
    kw = {}
    if bg:
        kw["bbox"] = dict(fc=bg, ec="none", pad=0.6)
    ax.text(x, y, text, fontsize=pt, color=color, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=z, **kw)


def dot(ax, x, y, color=LINE):
    ax.plot([x], [y], marker="o", ms=2.0, color=color, zorder=3)


def save(fig, name, png):
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    if png:
        fig.savefig(Path(png) / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"built {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: framework overview
# ---------------------------------------------------------------------------
def architecture_overview(png=None):
    H = 94.0
    fig, ax = new_figure(H)
    aux = dict(color=MUTED, ls=(0, (2.2, 1.4)))
    red = ACCENT["loss"]
    sub = 5.5

    # Stage 1: UDA training ---------------------------------------------------------
    panel(ax, 1, 29, 178, 64,
          "Stage 1  UDA training  (labeled source + unlabeled target EEG)")
    src = Box(ax, 4, 72, 24, 11, "Source EEG", [r"labeled $(X^s, y^s)$"], "data",
              body_pt=sub)
    tgt = Box(ax, 4, 57, 24, 11, "Target EEG", [r"unlabeled $X^t$"], "data",
              body_pt=sub)
    ra = Box(ax, 33, 57, 20, 26, "Riemannian\nalignment", ["per subject"],
             "adapt", body_pt=sub)
    arrow(ax, [src.r(), (ra.x, src.cy)])
    arrow(ax, [tgt.r(), (ra.x, tgt.cy)])

    panel(ax, 58, 55, 77, 29, r"Shared encoder $f_\theta$", color=ACCENT["conv"],
          fill="#FAFCFB")
    cy, bh = 67.0, 15.0
    cnn = Box(ax, 61, cy - bh / 2, 22, bh, "Multi-scale\nCNN",
              ["temporal + spatial"], "conv", body_pt=sub)
    ssm = Box(ax, 87, cy - bh / 2, 22, bh, "Bidirectional\nSSM × 5",
              ["F–B–F–B–F scans"], "ssm", body_pt=sub)
    tcn = Box(ax, 113, cy - bh / 2, 19, bh, "TCN", ["+ CNN shortcut"], "fuse",
              body_pt=sub)
    align = Box(ax, 139, cy - bh / 2, 18, bh, "Feature\naligner",
                ["residual MLP"], "adapt", body_pt=sub)
    cls = Box(ax, 161, cy - bh / 2, 16, bh, "Classifier", [], "neutral",
              title_pt=6.5)
    arrow(ax, [(ra.x + ra.w, cy), cnn.l()])
    for a, b in ((cnn, ssm), (ssm, tcn), (tcn, align), (align, cls)):
        arrow(ax, [a.r(), b.l()])

    # Losses
    ly, lh = 40, 10
    tmmd = Box(ax, 87, ly, 22, lh, r"$\mathcal{L}_{tmmd}$",
               ["temporal MMD"], "loss", title_pt=7.5, body_pt=sub)
    adv = Box(ax, 113, ly, 19, lh, r"$\mathcal{L}_{adv}$",
               ["GRL domain loss"], "loss", title_pt=7.5, body_pt=sub)
    mmd = Box(ax, 139, ly, 18, lh, r"$\mathcal{L}_{mmd}$",
               ["trial MMD"], "loss", title_pt=7.5, body_pt=sub)
    lcls = Box(ax, 161, ly, 16, lh, r"$\mathcal{L}_{cls}$",
               ["source CE"], "loss", title_pt=7.5, body_pt=sub)

    tap_x = tcn.x + 4
    arrow(ax, [(tap_x, tcn.y), (tap_x, 52.5), (tmmd.cx, 52.5), tmmd.t()], **aux)
    arrow(ax, [align.b(), mmd.t()], **aux)
    arrow(ax, [(align.cx, 53.0), (adv.cx + 1.5, 53.0), (adv.cx + 1.5, adv.y + lh)],
          **aux)
    dot(ax, align.cx, 53.0, MUTED)
    arrow(ax, [cls.b(), lcls.t()], **aux)

    total = Box(ax, 20, 31.5, 56, 10, "Total loss",
                [r"$\mathcal{L}_{cls}+\mathcal{L}_{adv}+0.5\,\mathcal{L}_{mmd}"
                 r"+0.1\,\mathcal{L}_{tmmd}$"], "loss", title_pt=6.0, body_pt=6.5)
    bus_y = total.cy
    for box in (tmmd, adv, mmd, lcls):
        arrow(ax, [box.b(), (box.cx, bus_y)], color=red, head=False)
    arrow(ax, [(lcls.cx, bus_y), (total.x + total.w, bus_y)], color=red)

    # Stage 2 and Stage 3 -------------------------------------------------------------
    panel(ax, 1, 8, 110, 19, "Stage 2  Target adaptation (after training)")
    s2a = Box(ax, 4, 10, 26, 10, "Target EEG", ["unlabeled"], "data",
              title_pt=6.5, body_pt=sub)
    s2b = Box(ax, 36, 10, 34, 10, "Frozen model",
              [r"update only BN $\gamma,\beta$"], "neutral", title_pt=6.5, body_pt=sub)
    s2c = Box(ax, 76, 10, 32, 10, r"InfoMax  $\mathcal{L}_{IM}$",
              ["confident + diverse"], "loss", title_pt=6.5, body_pt=sub)
    arrow(ax, [s2a.r(), s2b.l()])
    arrow(ax, [s2c.l(), s2b.r()], color=red)

    panel(ax, 114, 8, 65, 19, "Stage 3  Inference")
    s3a = Box(ax, 117, 10, 26, 10, "Test EEG", ["held-out sessions"], "locked",
              title_pt=6.5, body_pt=sub)
    s3b = Box(ax, 150, 10, 26, 10, r"Prediction $\hat{y}$", [], "neutral",
              title_pt=6.5)
    arrow(ax, [s3a.r(), s3b.l()])

    # Legend
    lg = 3.5
    arrow(ax, [(4, lg), (11, lg)])
    label(ax, 12.5, lg, "data path", ha="left")
    arrow(ax, [(34, lg), (41, lg)], **aux)
    label(ax, 42.5, lg, "features used by a loss", ha="left")
    arrow(ax, [(78, lg), (85, lg)], color=red)
    label(ax, 86.5, lg, "loss / gradient", ha="left")

    save(fig, "architecture_overview", png)


def operator(ax, x, y, symbol, r=2.0, color=LINE):
    """Circled element-wise operator centred at (x, y)."""
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec=color, lw=LW, zorder=3))
    ax.text(x, y, symbol, fontsize=7, color=color, ha="center", va="center",
            zorder=4)


# ---------------------------------------------------------------------------
# Figure 2: encoder with tensor shapes
# ---------------------------------------------------------------------------
def encoder_detail(png=None):
    """Figure 2: multi-scale temporal-spatial convolutional front-end."""
    fig, ax = new_figure(70.0)

    # Main pipeline: one left-to-right row ------------------------------------------
    panel(ax, 1, 35, 178, 34,
          "Multi-scale temporal–spatial convolution  (BCI IV-2a: C = 22 electrodes)")
    cy, bh = 51.0, 22.0
    y0 = cy - bh / 2
    inp = Box(ax, 4, cy - 6, 14, 12, "EEG", [], "data", dims="[B, C, T]")
    convs = []
    for i, k in enumerate((20, 32, 64)):
        convs.append(Box(ax, 23, cy + 4.5 - 7.5 * i, 22, 6, None,
                         [rf"Conv $1\times{k}$"], "data"))
    cat = Box(ax, 50, y0, 15, bh, "Concat", ["3 × 32"], "conv",
              dims="[B, 96, C, T]")
    spatial = Box(ax, 70, y0, 25, bh, "Depthwise\nspatial conv",
                  [r"$C\times1$, depth 2", "BN · ELU · pool 8"], "conv",
                  body_pt=5.5, dims="[B, 192, T/8]")
    grouped = Box(ax, 100, y0, 24, bh, "Grouped\nconvolutions",
                  ["1×1: 192 → 48", "1×16 temporal"], "conv", body_pt=5.5,
                  dims="[B, 48, T/8]")
    cga = Box(ax, 129, y0, 19, bh, "Channel-\ngroup\nattention", [], "conv",
              title_pt=6.5)
    out = Box(ax, 153, y0, 24, bh, "AvgPool 7", [r"output $H_c$"], "data",
              dims="[B, 48, L]")

    split_x = inp.x + inp.w + 2.5
    merge_x = cat.x - 2.5
    for c in convs:
        if abs(c.cy - inp.cy) < 0.1:
            arrow(ax, [inp.r(), c.l()])
            arrow(ax, [c.r(), cat.l()])
        else:
            arrow(ax, [(split_x, inp.cy), (split_x, c.cy), c.l()])
            arrow(ax, [c.r(), (merge_x, c.cy), (merge_x, cat.cy)], head=False)
    dot(ax, split_x, inp.cy)
    dot(ax, merge_x, cat.cy)
    for a, b in ((cat, spatial), (spatial, grouped), (grouped, cga), (cga, out)):
        arrow(ax, [a.r(), b.l()])

    # Zoomed channel-group attention ----------------------------------------------------
    ix0, ix1, iy0, iy1 = 22.0, 158.0, 2.0, 30.0
    panel(ax, ix0, iy0, ix1 - ix0, iy1 - iy0, "Channel-group attention (residual)",
          color=ACCENT["conv"], fill="#FAFCFB", label_pt=6.0)
    zoom = dict(color="#9AA5B1", lw=0.45, ls=(0, (1.5, 1.2)))
    ax.add_line(Line2D([cga.x, ix0 + 30], [cga.y, iy1], zorder=0.5, **zoom))
    ax.add_line(Line2D([cga.x + cga.w, ix1], [cga.y, iy1], zorder=0.5, **zoom))

    main_y, att_y, sbh = 20.0, 10.0, 7.0
    label(ax, ix0 + 3.0, main_y, "$H$", pt=7, color=INK)
    start_x = ix0 + 5.0
    fork_x = ix0 + 8.0
    gap = Box(ax, 34, att_y - sbh / 2, 12, sbh, "GAP", [], "conv", title_pt=6.0)
    fc1 = Box(ax, 50, att_y - sbh / 2, 21, sbh, None, ["1×1, 48 → 12"], "conv")
    relu = Box(ax, 75, att_y - sbh / 2, 11, sbh, "ReLU", [], "conv", title_pt=6.0)
    fc2 = Box(ax, 90, att_y - sbh / 2, 20, sbh, None, ["1×1, 12 → 3"], "conv")
    sig = Box(ax, 114, att_y - sbh / 2, 8, sbh, "σ", [], "conv", title_pt=6.5)
    mul_x, add_x = 134.0, 144.0

    arrow(ax, [(start_x, main_y), (add_x - 2.0, main_y)])
    dot(ax, fork_x, main_y)
    arrow(ax, [(fork_x, main_y), (fork_x, att_y), gap.l()])
    for a, b in ((gap, fc1), (fc1, relu), (relu, fc2), (fc2, sig)):
        arrow(ax, [a.r(), b.l()])
    operator(ax, mul_x, att_y, "×")
    operator(ax, add_x, main_y, "+")
    arrow(ax, [sig.r(), (mul_x - 2.0, att_y)])
    label(ax, (sig.x + sig.w + mul_x - 2.0) / 2, att_y + 2.0, "$a(H)$", pt=5.5,
          color=INK)
    dot(ax, mul_x, main_y)
    arrow(ax, [(mul_x, main_y), (mul_x, att_y + 2.0)])
    arrow(ax, [(mul_x + 2.0, att_y), (add_x, att_y), (add_x, main_y - 2.0)])
    arrow(ax, [(add_x + 2.0, main_y), (ix1 - 2.5, main_y)])

    save(fig, "encoder_detail", png)


# ---------------------------------------------------------------------------
# Figure 4: local-global fusion, TCN and classifier
# ---------------------------------------------------------------------------
def tcn_head(png=None):
    fig, ax = new_figure(81.0)
    panel(ax, 1, 41, 178, 39, "Local–global fusion and TCN head")
    ssm_out = Box(ax, 4, 62, 23, 10, "SSM output", [], "ssm", title_pt=6.5,
                  dims="[B, 48, L]")
    hc = Box(ax, 4, 45, 23, 10, r"Shortcut $H_c$", [], "conv", title_pt=6.5,
             dims="[B, 48, L]")
    red_ = Box(ax, 32, 61, 20, 12, "Pointwise\nreduction", ["48 → 16"], "ssm",
               title_pt=6.5)
    cat = Box(ax, 57, 45, 15, 28, "Concat", ["48 + 16"], "fuse", title_pt=6.5,
              dims="[B, 64, L]")
    cy = cat.cy
    tcn1 = Box(ax, 77, cy - 9, 23, 18, "TCN block", ["dilation 1"], "fuse")
    tcn2 = Box(ax, 105, cy - 9, 23, 18, "TCN block", ["dilation 2"], "fuse")
    pool = Box(ax, 133, cy - 9, 21, 18, "Last–mean\npooling", [], "fuse",
               title_pt=6.5, dims="$q$: [B, 64]")
    cls = Box(ax, 159, cy - 9, 18, 18, "Grouped\nclassifier",
              ["4 groups"], "fuse", title_pt=6.5, body_pt=5.5, dims="[B, K]")
    arrow(ax, [ssm_out.r(), (red_.x, ssm_out.cy)])
    arrow(ax, [(red_.x + red_.w, ssm_out.cy), (cat.x, ssm_out.cy)])
    arrow(ax, [hc.r(), (cat.x, hc.cy)], color=ACCENT["fuse"], ls=(0, (3, 1.6)))
    for a, b in ((cat, tcn1), (tcn1, tcn2), (tcn2, pool), (pool, cls)):
        arrow(ax, [a.r(), b.l()])

    # Residual TCN block detail ---------------------------------------------------
    panel(ax, 1, 2, 178, 36,
          "Residual TCN block  (grouped causal convolution, 4 groups, 64 channels)")
    cy, bh = 17.0, 11.0
    label(ax, 4.5, cy, "$x$", pt=7, color=INK)
    c1 = Box(ax, 10, cy - bh / 2, 27, bh, "Causal conv", ["k = 4, dilation d"],
             "fuse", title_pt=6.5)
    b1 = Box(ax, 41, cy - bh / 2, 27, bh, "BN · ELU", ["dropout 0.3"], "fuse",
             title_pt=6.5)
    c2 = Box(ax, 72, cy - bh / 2, 27, bh, "Causal conv", ["k = 4, dilation d"],
             "fuse", title_pt=6.5)
    b2 = Box(ax, 103, cy - bh / 2, 27, bh, "BN · ELU", ["dropout 0.3"], "fuse",
             title_pt=6.5)
    add_x = 139.0
    elu = Box(ax, 147, cy - bh / 2, 14, bh, "ELU", [], "fuse", title_pt=6.5)
    arrow(ax, [(6.5, cy), c1.l()])
    for a, b in ((c1, b1), (b1, c2), (c2, b2)):
        arrow(ax, [a.r(), b.l()])
    operator(ax, add_x, cy, "+")
    arrow(ax, [b2.r(), (add_x - 2.0, cy)])
    arrow(ax, [(add_x + 2.0, cy), elu.l()])
    arrow(ax, [elu.r(), (176.0, cy)])
    res_y = 29.5
    arrow(ax, [(6.5, cy), (6.5, res_y), (add_x, res_y), (add_x, cy + 2.0)],
          color=MUTED)
    label(ax, 72, res_y, "residual", bg="#F5F7F9")
    label(ax, 88, 7.0, "Causal padding of (k − 1)·d samples keeps the output aligned "
          "with past inputs only.", pt=5.5)

    save(fig, "tcn_head", png)


# ---------------------------------------------------------------------------
# Figure 3: one selective SSM block
# ---------------------------------------------------------------------------
def ssm_block(png=None):
    fig, ax = new_figure(57.0, y0=1.0)
    panel(ax, 1, 2, 178, 55,
          "Selective SSM block  (d = 48, state size N = 8)")
    cy = 24.0
    bh = 11.0
    label(ax, 4.5, cy, "$x$", pt=7, color=INK)
    ln = Box(ax, 9, cy - bh / 2, 15, bh, "LayerNorm", [], "ssm", title_pt=6.5)
    flip1 = Box(ax, 28, cy - bh / 2, 15, bh, "Flip time", ["B blocks only"],
                "neutral", title_pt=6.5, body_pt=5.0, ls=(0, (2, 1.2)))
    inproj = Box(ax, 47, cy - bh / 2, 15, bh, "Linear", ["48 → 2 × 48"], "ssm",
                 title_pt=6.5)
    split = inproj.x + inproj.w + 3.5
    conv = Box(ax, 68, cy - bh / 2, 20, bh, "Depthwise conv",
               ["causal, k = 3"], "ssm", title_pt=6.5)
    silu = Box(ax, 91, cy - bh / 2, 10, bh, "SiLU", [], "ssm", title_pt=6.5)
    scan = Box(ax, 110, cy - 9, 37, 18, "Selective scan",
               [r"$h_t=\exp(\Delta_t A)\,h_{t-1}+\Delta_t B_t\,u_t$",
                r"$y_t=\langle C_t,\,h_t\rangle+D\,u_t$",
                r"$A=-\exp(A_{\log})$, diagonal"], "ssm", title_pt=6.5)
    params = Box(ax, 110, 38, 37, 9, "Linear → input-dependent",
                 [r"$\Delta_t=\min(\mathrm{softplus}(\cdot),1)$,  $B_t$,  $C_t$"],
                 "ssm", title_pt=6.0, body_pt=5.5)
    gate = Box(ax, 118, 5.5, 21, 8, "SiLU gate", [], "ssm", title_pt=6.5)
    mul_x = 153.0
    outproj = Box(ax, 157, cy - bh / 2, 12, bh, "Linear", ["48 → 48"], "ssm",
                  title_pt=6.5, body_pt=5.0)
    add_x = 173.5

    arrow(ax, [(6.2, cy), ln.l()])
    arrow(ax, [ln.r(), flip1.l()])
    arrow(ax, [flip1.r(), inproj.l()])
    arrow(ax, [inproj.r(), conv.l()])
    dot(ax, split, cy)
    arrow(ax, [conv.r(), silu.l()])
    tap = silu.x + silu.w + 4
    arrow(ax, [silu.r(), scan.l()])
    dot(ax, tap, cy)
    label(ax, tap - 1.0, cy + 1.8, "$u$", pt=6, color=INK, ha="right")
    arrow(ax, [(tap, cy), (tap, params.cy), params.l()])
    arrow(ax, [params.b(), scan.t()])
    arrow(ax, [(split, cy), (split, gate.cy), gate.l()])
    label(ax, split + 1.0, gate.cy + 1.6, "$r$", pt=6, color=INK, ha="left")
    operator(ax, mul_x, cy, "×")
    arrow(ax, [scan.r(), (mul_x - 2.0, cy)])
    arrow(ax, [gate.r(), (mul_x, gate.cy), (mul_x, cy - 2.0)])
    arrow(ax, [(mul_x + 2.0, cy), outproj.l()])
    operator(ax, add_x, cy, "+")
    arrow(ax, [outproj.r(), (add_x - 2.0, cy)])
    arrow(ax, [(add_x + 2.0, cy), (178, cy)])
    label(ax, outproj.cx + 2, cy - 8.0, "flip back (B blocks),\ndropout, DropPath",
          pt=5.0, ha="center", va="top")
    res_y = 51.0
    arrow(ax, [(6.2, cy), (6.2, res_y), (add_x, res_y), (add_x, cy + 2.0)],
          color=MUTED)
    label(ax, 60, res_y, "residual", bg="#F5F7F9")

    save(fig, "ssm_block", png)


def main():
    png = None
    if "--png" in sys.argv:
        png = sys.argv[sys.argv.index("--png") + 1]
        Path(png).mkdir(parents=True, exist_ok=True)
    architecture_overview(png)
    encoder_detail(png)
    ssm_block(png)
    tcn_head(png)


if __name__ == "__main__":
    main()
