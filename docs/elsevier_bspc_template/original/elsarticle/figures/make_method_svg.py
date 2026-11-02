"""Draw the SSM-DAN method figure as an SVG that imports cleanly into Figma.

Large-print architecture diagram: (a) the SSM backbone and (b) the adaptation
head side by side as stacks of layer bars, the Stage-1 objective under (b),
and (c) the two training stages as horizontal data paths underneath. Every
bar, arrow and text run is its own element (no markers, filters or <tspan>
offsets), so Figma imports them as editable layers.
Usage: python make_method_svg.py OUT.svg
"""
import math
import re
import sys
from pathlib import Path

from PIL import ImageFont

FONT = 'Arial'
FONT_FILES = {False: 'C:/Windows/Fonts/arial.ttf', True: 'C:/Windows/Fonts/arialbd.ttf'}
_fonts = {}

INK, MUTE = '#1F2933', '#6B7684'
COL = {  # stroke, block fill
    'src': ('#2F6DB5', '#CFE0F5'),
    'tgt': ('#D9822B', '#FADDBA'),
    'ra': ('#1F8A8A', '#C0E6E2'),
    'bb': ('#4E5967', '#E0E4EA'),
    'conv': ('#2F62A0', '#C9DAF1'),
    'norm': ('#5F6B7A', '#DADFE6'),
    'pool': ('#2E8A80', '#C0E4DD'),
    'attn': ('#B86E1F', '#F6D6AC'),
    'ssm': ('#3346AD', '#CBD1F3'),
    'da': ('#8338A8', '#E4CFF3'),
    'cls': ('#2F3B4C', '#E6E9EE'),
    'loss': ('#C0392B', '#F8D2CE'),
    'tta': ('#2E8B57', '#E4F5EB'),
}

out = []


def text_width(text, size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_FILES[bold], size=size * 4)
    return _fonts[key].getlength(text) / 4


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def segments(line):
    """'L_{CE}' -> runs of (text, kind); kind is n, sub, sup or L (italic L)."""
    parts, pos = [], 0
    for m in re.finditer(r'([_^])\{([^}]*)\}', line):
        if m.start() > pos:
            parts.append((line[pos:m.start()], 'n'))
        parts.append((m.group(2), 'sub' if m.group(1) == '_' else 'sup'))
        pos = m.end()
    if pos < len(line):
        parts.append((line[pos:], 'n'))
    runs = []
    for text_, kind in parts:
        if kind != 'n' or 'ℒ' not in text_:
            runs.append((text_, kind))
            continue
        for i, chunk in enumerate(text_.split('ℒ')):
            if i:
                runs.append(('L', 'L'))
            if chunk:
                runs.append((chunk, 'n'))
    return runs


def text(x, y, line, size=12, bold=False, color=INK, anchor='middle', italic=False, rotate=False):
    """Rich text; rotate=True writes it bottom-to-top centred on (x, y)."""
    runs = segments(line)
    small = round(size * 0.68, 1)
    widths = [text_width(t, small if k in ('sub', 'sup') else size, bold) + (1.0 if k == 'L' else 0.0)
              for t, k in runs]
    total = sum(widths)
    if rotate:
        cx, base = x - total / 2, y + size * 0.35
        transform = f' transform="rotate(-90 {x:.1f} {y:.1f})"'
    else:
        cx = x - total / 2 if anchor == 'middle' else (x - total if anchor == 'end' else x)
        base, transform = y, ''
    for (t, k), w in zip(runs, widths):
        fs = small if k in ('sub', 'sup') else size
        dy = size * 0.28 if k == 'sub' else (-size * 0.42 if k == 'sup' else 0)
        style = ' font-style="italic"' if (italic or k == 'L') else ''
        # Importers may trim edge spaces: turn a leading space into an x offset.
        lead = text_width(t[:len(t) - len(t.lstrip(' '))], fs, bold)
        if t.strip():
            out.append(f'<text x="{cx + lead:.1f}" y="{base + dy:.1f}" font-family="{FONT}" font-size="{fs}" '
                       f'font-weight="{"bold" if bold else "normal"}"{style} fill="{color}"{transform}>'
                       f'{esc(t.strip(" "))}</text>')
        cx += w


def rect(x, y, w, h, stroke, fill='none', rx=12, width=1.5, dashed=False, opacity=None):
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    op = f' fill-opacity="{opacity}"' if opacity is not None else ''
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{op} '
               f'stroke="{stroke}" stroke-width="{width}"{dash}/>')


def path(d, stroke, width=1.6, dashed=False, fill='none'):
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    out.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash} '
               f'stroke-linecap="round" stroke-linejoin="round"/>')


def arrow(points, color, dashed=False, width=1.7, head=True):
    size = 8.5
    (x0, y0), (x1, y1) = points[-2], points[-1]
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    bx, by = (x1 - ux * size, y1 - uy * size) if head else (x1, y1)
    d = 'M ' + ' L '.join(f'{px:.1f} {py:.1f}' for px, py in points[:-1]) + f' L {bx:.1f} {by:.1f}'
    path(d, color, width, dashed)
    if head:
        px, py = -uy * size * 0.55, ux * size * 0.55
        out.append(f'<path d="M {x1:.1f} {y1:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z" '
                   f'fill="{color}"/>')


def circle_op(cx, cy, symbol, color=INK, r=11):
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="#FFFFFF" stroke="{color}" stroke-width="1.6"/>')
    if symbol == 'concat':
        # channel-wise concatenation, drawn as two bars so it does not depend on the font
        for dx in (-2.6, 2.6):
            path(f'M {cx + dx:.1f} {cy - r * 0.55:.1f} L {cx + dx:.1f} {cy + r * 0.55:.1f}', color, 1.8)
    else:
        text(cx, cy + 5, symbol, 14, True, color)


def group(name):
    out.append(f'<g id="{name}">')


def end():
    out.append('</g>')


def dashed_group(x0, y0, x1, y1, color, tag, name, tag_size=11, align='left'):
    """Dashed rounded group. align='left'/'center' puts the tag on the top
    border; align='side-left'/'side-right' puts it, rotated, on that side
    border so the group needs no extra height for its label."""
    group(name)
    rect(x0, y0, x1 - x0, y1 - y0, color, color, rx=14, width=1.3, dashed=True, opacity=0.05)
    if tag:
        tw = text_width(tag, tag_size, True) + 18
        if align.startswith('side'):
            bx = x0 if align == 'side-left' else x1
            cy = (y0 + y1) / 2
            rect(bx - 10, cy - tw / 2, 20, tw, color, '#FFFFFF', rx=10, width=1.2)
            text(bx, cy, tag, tag_size, True, color, rotate=True)
        else:
            tx = x0 + 14 if align == 'left' else x0 + (x1 - x0 - tw) / 2
            rect(tx, y0 - 10, tw, 20, color, '#FFFFFF', rx=10, width=1.2)
            text(tx + tw / 2, y0 + 4, tag, tag_size, True, color)
    end()


PANEL_TINT = {'panel_a': '#F1F5FB', 'panel_b': '#F6F2FB', 'panel_c': '#F0F7F2'}



# ---------------------------------------------------------------------------
# Large-print layout: (a) and (b) side by side on top, (c) as two horizontal
# rows underneath. Text is about twice the size of the first version so the
# figure stays readable at full text width.
# ---------------------------------------------------------------------------
FS_SLAB, FS_NAME, FS_SHAPE, FS_TITLE, FS_PILL, FS_BLOCK, FS_SUB = 26, 22, 20, 34, 26, 25, 20
LW = 2.6                                   # main line width


def fit(label, max_w, size, bold=True):
    """Largest size <= size at which the label fits in max_w."""
    while size > 12 and text_width(label.replace('_{', '').replace('}', ''), size, bold) > max_w:
        size -= 0.5
    return size


def shape_label(x, y, label, anchor='start'):
    text(x, y, label, FS_SHAPE, False, '#8A94A3', anchor=anchor)


def side_label(x_right, y_mid, lines, color):
    n = len(lines)
    for i, line in enumerate(lines):
        text(x_right, y_mid + 8 + (i - (n - 1) / 2) * 27, line, FS_NAME, True, color, anchor='end')


def bracket(x, y0, y1, color, name='bracket'):
    group(name)
    out.append(f'<path d="M {x} {y0} L {x} {y1}" fill="none" stroke="{color}" stroke-opacity="0.45" '
               f'stroke-width="2" stroke-dasharray="8 7" stroke-linecap="round"/>')
    end()


def dot(cx, cy, color):
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"/>')


def big_arrow(points, color, dashed=False, width=LW, head=True):
    """Arrow scaled for the large-print layout."""
    size = 15.0
    (x0, y0), (x1, y1) = points[-2], points[-1]
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    bx, by = (x1 - ux * size, y1 - uy * size) if head else (x1, y1)
    d = 'M ' + ' L '.join(f'{px:.1f} {py:.1f}' for px, py in points[:-1]) + f' L {bx:.1f} {by:.1f}'
    dash = ' stroke-dasharray="10 8"' if dashed else ''
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} '
               f'stroke-linecap="round" stroke-linejoin="round"/>')
    if head:
        px, py = -uy * size * 0.55, ux * size * 0.55
        out.append(f'<path d="M {x1:.1f} {y1:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z" '
                   f'fill="{color}"/>')


def line(points, color, width=LW, dashed=False):
    dash = ' stroke-dasharray="10 8"' if dashed else ''
    d = 'M ' + ' L '.join(f'{px:.1f} {py:.1f}' for px, py in points)
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} '
               f'stroke-linecap="round" stroke-linejoin="round"/>')


def op(cx, cy, symbol, color=INK, r=17):
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="#FFFFFF" stroke="{color}" stroke-width="{LW}"/>')
    if symbol == 'concat':
        for dx in (-4.5, 4.5):
            line([(cx + dx, cy - r * 0.55), (cx + dx, cy + r * 0.55)], color, 2.8)
    else:
        line([(cx - r * 0.55, cy), (cx + r * 0.55, cy)], color, 2.8)
        line([(cx, cy - r * 0.55), (cx, cy + r * 0.55)], color, 2.8)


def bar(cx, top, w, h, kind, label, name, size=FS_SLAB):
    stroke, fill = COL[kind]
    group(name)
    rect(cx - w / 2, top, w, h, stroke, fill, rx=9, width=2.6)
    s = fit(label, w - 20, size)
    text(cx, top + h / 2 + s * 0.36, label, s, True, INK)
    end()


def pill(cx, cy, label, kind='loss', w=120, h=42, dashed=True, size=FS_PILL, name='pill'):
    stroke, fill = COL[kind]
    group(name)
    rect(cx - w / 2, cy - h / 2, w, h, stroke, fill, rx=h / 2, width=2.4, dashed=dashed)
    text(cx, cy + size * 0.38, label, size, True, stroke)
    end()


def frame(x0, y0, x1, y1, tint, title, name, title_color=INK):
    group(name)
    rect(x0, y0, x1 - x0, y1 - y0, '#1A1A1A', tint, rx=26, width=3.2, dashed=True)
    text(x0 + 26, y0 + 46, title, FS_TITLE, True, title_color, anchor='start')
    end()


W = 1500
PA = (16, 700)
PB = (716, 1484)
TOP = 16
SECTION_START = len(out)

# ============================ (a) SSM backbone ============================
CXA = 418
SW, SH, SG = 420, 40, 5                    # bar width, height, gap
MOD_GAP = 38
BRACKET_X = CXA - SW / 2 - 22
NAME_X = BRACKET_X - 16
SHORTCUT_X = CXA + SW / 2 + 24
LABEL_X = CXA + 16
y = TOP + 84


def stack(items, w=SW, cx=CXA):
    """Bars stacked without arrows; returns (top, bottom)."""
    global y
    top = y
    for kind, label, name in items:
        bar(cx, y, w, SH, kind, label, name)
        y += SH + SG
    y -= SG
    return top, y


def module(names, kind, items):
    top, bottom = stack(items)
    bracket(BRACKET_X, top, bottom, COL[kind][0], 'mod_' + items[0][2])
    side_label(NAME_X, (top + bottom) / 2, names, COL[kind][0])


def link(label=None):
    global y
    big_arrow([(CXA, y + 3), (CXA, y + MOD_GAP - 3)], COL['bb'][0])
    if label:
        shape_label(LABEL_X, y + MOD_GAP / 2 + 7, label)
    y += MOD_GAP


# input
group('input_eeg')
rect(CXA - 150, y, 300, 54, COL['ra'][0], COL['ra'][1], rx=12, width=2.6)
text(CXA, y + 36, 'Aligned EEG  X̃', 26, True, INK)
end()
shape_label(CXA + 162, y + 34, 'C × T')
y_in = y + 54

# multi-scale temporal convolution
BW_, BH_ = 134, 62
bxs = [CXA - 143, CXA, CXA + 143]
split_y = y_in + 22
br_top = split_y + 34
line([(CXA, y_in + 2), (CXA, split_y)], COL['bb'][0])
line([(bxs[0], split_y), (bxs[-1], split_y)], COL['bb'][0])
for bx, k in zip(bxs, (20, 32, 64)):
    big_arrow([(bx, split_y), (bx, br_top - 3)], COL['bb'][0])
    group(f'conv_1x{k}')
    rect(bx - BW_ / 2, br_top, BW_, BH_, COL['conv'][0], COL['conv'][1], rx=9, width=2.6)
    text(bx, br_top + 27, f'Conv 1×{k}', fit(f'Conv 1×{k}', BW_ - 12, 24), True, INK)
    text(bx, br_top + 51, 'BN', 20, True, MUTE)
    end()
merge_y = br_top + BH_ + 22
for bx in bxs:
    line([(bx, br_top + BH_ + 2), (bx, merge_y)], COL['bb'][0])
line([(bxs[0], merge_y), (bxs[-1], merge_y)], COL['bb'][0])
cat_y = merge_y + 44
big_arrow([(CXA, merge_y), (CXA, cat_y - 19)], COL['bb'][0])
op(CXA, cat_y, 'concat')
bracket(BRACKET_X, br_top, cat_y + 17, COL['conv'][0], 'mod_multiscale')
side_label(NAME_X, (br_top + cat_y) / 2, ['Multi-scale', 'temporal'], COL['conv'][0])
y = cat_y + 17
link('96 × C × T')

module(['Spatial', 'filtering'], 'conv', [
    ('conv', 'Depthwise conv  C×1', 'spatial_dwconv'),
    ('norm', 'BN · ELU', 'spatial_bn'),
    ('pool', 'AvgPool  1×8', 'spatial_pool'),
    ('conv', 'Grouped conv  1×1', 'spatial_reduce'),
])
link('48 × T/8')

module(['Grouped', 'temporal'], 'conv', [
    ('conv', 'Grouped conv  1×16', 'gt_conv'),
    ('norm', 'BN · ELU', 'gt_bn'),
    ('attn', 'Group attention', 'gt_attn'),
    ('pool', 'AvgPool  1×7', 'gt_pool'),
])
shortcut_y = y + 12
dot(CXA, shortcut_y, COL['bb'][0])
big_arrow([(CXA, y + 3), (CXA, y + MOD_GAP + 22 - 3)], COL['bb'][0])
shape_label(LABEL_X, shortcut_y + 34, '48 × T/56')
y += MOD_GAP + 22

# selective SSM encoder
seq_top = y
bar(CXA, y, SW, SH, 'conv', '1×1 conv · SiLU  (mix)', 'mix')
y += SH
INNER = SW - 74
node_y = y + 18
ssm_top = node_y + 36
line([(CXA, y + 2), (CXA, node_y)], COL['bb'][0])
big_arrow([(CXA, node_y), (CXA, ssm_top - 3)], COL['bb'][0])
y = ssm_top
ssm_top, ssm_bottom = stack([('norm', 'LayerNorm', 'ssm_ln'), ('ssm', 'Linear', 'ssm_in'),
                             ('conv', 'Depthwise conv  k=3', 'ssm_conv'), ('ssm', 'Selective scan', 'ssm_scan'),
                             ('ssm', '⊙ SiLU gate', 'ssm_gate'), ('ssm', 'Linear', 'ssm_out')], w=INNER)
plus_y = ssm_bottom + 56
big_arrow([(CXA, ssm_bottom + 3), (CXA, plus_y - 18)], COL['bb'][0])
res_x = CXA - INNER / 2 - 16
big_arrow([(CXA, node_y), (res_x, node_y), (res_x, plus_y), (CXA - 18, plus_y)], COL['ssm'][0])
dot(CXA, node_y, COL['ssm'][0])
op(CXA, plus_y, '+', COL['ssm'][0], r=15)
rep_x = CXA + INNER / 2 + 30
mid_rep = (node_y + plus_y) / 2
bracket(rep_x, node_y, mid_rep - 24, COL['ssm'][0], 'rep_top')
bracket(rep_x, mid_rep + 24, plus_y + 15, COL['ssm'][0], 'rep_bottom')
text(rep_x, mid_rep + 9, '×N', 26, True, COL['ssm'][0])
y = plus_y + 15 + 32
big_arrow([(CXA, plus_y + 17), (CXA, y - 3)], COL['bb'][0])
bar(CXA, y, SW, SH, 'conv', '1×1 conv · SiLU  (reduce)', 'reduce')
y += SH
bracket(BRACKET_X, seq_top, y, COL['ssm'][0], 'mod_sequence')
side_label(NAME_X, (seq_top + y) / 2, ['Selective', 'SSM', 'encoder'], COL['ssm'][0])

# concatenation with the CNN shortcut
cat2 = y + 56
big_arrow([(CXA, y + 3), (CXA, cat2 - 19)], COL['bb'][0])
shape_label(LABEL_X, y + 34, '16 × T/56')
big_arrow([(CXA, shortcut_y), (SHORTCUT_X, shortcut_y), (SHORTCUT_X, cat2), (CXA + 19, cat2)], COL['bb'][0], width=2.0)
op(CXA, cat2, 'concat')
y = cat2 + 17
link('64 × T/56')

# temporal head with the temporal-MMD tap
head_top = y
stack([('conv', 'Residual causal conv  d=1', 'tcn_block1'),
       ('conv', 'Residual causal conv  d=2', 'tcn_block2')])
bracket(BRACKET_X, head_top, y, COL['conv'][0], 'mod_head')
side_label(NAME_X, (head_top + y) / 2, ['Temporal', 'head'], COL['conv'][0])
# TCN output: the temporal-MMD branch leaves here, then the trial feature f
tap_y = y + 34
line([(CXA, y + 2), (CXA, tap_y)], COL['bb'][0])
dot(CXA, tap_y, COL['bb'][0])
TMMD_X = PA[1] - 82
big_arrow([(CXA, tap_y), (TMMD_X - 64, tap_y)], COL['loss'][0], dashed=True)
pill(TMMD_X, tap_y, 'ℒ_{tMMD}', w=124, name='loss_tmmd')
y = tap_y
big_arrow([(CXA, y), (CXA, y + 44)], COL['bb'][0])
text(CXA, y + 80, 'f', 30, True, INK, italic=True)
shape_label(CXA + 22, y + 78, '64-dim feature')
A_BOTTOM = y + 104

# ============================ (b) adaptation head ============================
CXH = 1118
HW_ = 384
H_BRACKET_X = CXH - HW_ / 2 - 18
H_NAME_X = H_BRACKET_X - 13
y = TOP + 84
group('head_input')
rect(CXH - 150, y, 300, 54, COL['bb'][0], COL['bb'][1], rx=12, width=2.6)
text(CXH, y + 36, 'f_{s} , f_{t}', 28, True, INK, italic=True)
end()
shape_label(CXH + 162, y + 24, '64 = 4 groups')
shape_label(CXH + 162, y + 48, '× 16 channels')
y += 54
big_arrow([(CXH, y + 3), (CXH, y + 50)], COL['bb'][0])
y += 53


def hstack(items, w=HW_):
    global y
    top = y
    for kind, label, name in items:
        bar(CXH, y, w, SH, kind, label, name)
        y += SH + SG
    y -= SG
    return top, y


d_top_, d_bottom_ = hstack([('norm', 'Batch moments: source, target', 'dial_moments'),
                            ('da', 'Mix: α · target + (1 − α) · source', 'dial_mix'),
                            ('norm', 'Standardize per domain', 'dial_std'),
                            ('norm', 'Shared affine  w, b', 'dial_affine')])
bracket(H_BRACKET_X, d_top_, d_bottom_, COL['da'][0], 'mod_autodial')
side_label(H_NAME_X, (d_top_ + d_bottom_) / 2, ['AutoDIAL', 'normalization'], COL['da'][0])

al_node = y + 30
al_top = al_node + 36
line([(CXH, y + 2), (CXH, al_node)], COL['bb'][0])
big_arrow([(CXH, al_node), (CXH, al_top - 3)], COL['bb'][0])
y = al_top
AL_W = HW_ - 60
a_top, a_bottom = hstack([('norm', 'LayerNorm', 'al_ln1'), ('da', 'Linear  64 → 128', 'al_fc1'),
                          ('norm', 'GELU · Dropout', 'al_act'), ('da', 'Linear  128 → 64', 'al_fc2'),
                          ('norm', 'LayerNorm', 'al_ln2'), ('da', '× s  (learned scale)', 'al_scale')], w=AL_W)
al_plus = a_bottom + 56
big_arrow([(CXH, a_bottom + 3), (CXH, al_plus - 18)], COL['bb'][0])
res_xh = CXH - AL_W / 2 - 16
big_arrow([(CXH, al_node), (res_xh, al_node), (res_xh, al_plus), (CXH - 18, al_plus)], COL['da'][0])
dot(CXH, al_node, COL['da'][0])
op(CXH, al_plus, '+', COL['da'][0], r=15)
bracket(H_BRACKET_X, al_node, al_plus + 15, COL['da'][0], 'mod_aligner')
side_label(H_NAME_X, (al_node + al_plus) / 2, ['Residual', 'aligner'], COL['da'][0])

# three parallel branches on the aligned features
STEP = 160
MMD_X, CE_X, IM_X, ADV_X = CXH - 1.5 * STEP, CXH - 0.5 * STEP, CXH + 0.5 * STEP, CXH + 1.5 * STEP
CLS_W, DISC_W = 288, 176
bus_y = al_plus + 54
line([(CXH, al_plus + 17), (CXH, bus_y)], COL['bb'][0])
line([(MMD_X, bus_y), (ADV_X, bus_y)], COL['bb'][0])
dot(CXH, bus_y, COL['bb'][0])
text(CXH + 14, bus_y - 12, 'f′_{s}, f′_{t}', 24, True, INK, anchor='start', italic=True)
# classifier
y = bus_y + 70
big_arrow([(CXH, bus_y), (CXH, y - 3)], COL['bb'][0])
text(CXH - CLS_W / 2, y - 12, 'Classifier', 22, True, COL['cls'][0], anchor='start')
c_top, c_bottom = hstack([('conv', 'Grouped 1×1 conv (4 × K)', 'cls_conv'),
                          ('cls', 'Weighted group sum', 'cls_sum')], w=CLS_W)
yhat_y = c_bottom + 64
big_arrow([(CXH, c_bottom + 3), (CXH, yhat_y - 28)], COL['bb'][0])
text(CXH, yhat_y, 'ŷ', 30, True)
# GRL + discriminator
grl_y = bus_y + 44
big_arrow([(ADV_X, bus_y), (ADV_X, grl_y - 22)], COL['bb'][0])
pill(ADV_X, grl_y, 'GRL', w=96, h=42, dashed=False, size=24, name='grl')
disc_label_y = grl_y + 52
line([(ADV_X, grl_y + 21), (ADV_X, disc_label_y - 22)], COL['loss'][0])
text(ADV_X, disc_label_y + 7, 'Discriminator', fit('Discriminator', DISC_W + 20, 22), True, COL['loss'][0])
d_top = disc_label_y + 34
big_arrow([(ADV_X, disc_label_y + 14), (ADV_X, d_top - 3)], COL['loss'][0])
y = d_top
for i, label in enumerate(['Linear 64→128', 'Linear 128→64', 'Linear 64→1']):
    bar(ADV_X, y, DISC_W, SH, 'loss', label, f'disc_{i}', size=22)
    y += SH + SG
disc_bottom = y - SG

# one row of losses
LOSS_Y = max(yhat_y + 96, disc_bottom + 70)
pill(MMD_X, LOSS_Y, 'ℒ_{MMD}', w=128, name='loss_mmd')
pill(CE_X, LOSS_Y, 'ℒ_{CE}', w=112, name='loss_ce')
pill(IM_X, LOSS_Y, 'ℒ_{IM}', w=112, name='loss_im')
pill(ADV_X, LOSS_Y, 'ℒ_{adv}', w=120, name='loss_adv')
big_arrow([(MMD_X, bus_y), (MMD_X, LOSS_Y - 23)], COL['loss'][0], dashed=True)
split_y = yhat_y + 26
line([(CXH, yhat_y + 12), (CXH, split_y)], COL['loss'][0], dashed=True)
line([(CE_X, split_y), (IM_X, split_y)], COL['loss'][0], dashed=True)
big_arrow([(CE_X, split_y), (CE_X, LOSS_Y - 23)], COL['loss'][0], dashed=True)
big_arrow([(IM_X, split_y), (IM_X, LOSS_Y - 23)], COL['loss'][0], dashed=True)
big_arrow([(ADV_X, disc_bottom + 3), (ADV_X, LOSS_Y - 23)], COL['loss'][0], dashed=True)
B_BOTTOM = LOSS_Y + 50

# Stage-1 objective frame under (b), level with the temporal-MMD tap in (a)
OBJ_H = 160
OBJ_TOP = max(tap_y - OBJ_H / 2, B_BOTTOM + 70)
TOP_BOTTOM = max(A_BOTTOM, OBJ_TOP + OBJ_H)
obj_mid = OBJ_TOP + OBJ_H / 2
for xx in (MMD_X, CE_X, IM_X, ADV_X):
    big_arrow([(xx, LOSS_Y + 23), (xx, OBJ_TOP - 3)], COL['loss'][0], dashed=True)
if abs(obj_mid - tap_y) < 1:
    tmmd_route = [(TMMD_X + 64, tap_y), (PB[0] - 3, tap_y)]
else:
    gx = (PA[1] + PB[0]) / 2
    tmmd_route = [(TMMD_X + 64, tap_y), (gx, tap_y), (gx, obj_mid), (PB[0] - 3, obj_mid)]
big_arrow(tmmd_route, COL['loss'][0], dashed=True)

# ============================ (c) two stages, horizontal ============================
C_TOP = TOP_BOTTOM + 24
CX0, CX1 = 16, 1484
FH_ = 76                                       # flow block height


def fblock(x0, w, top, kind, title, sub=None, name='fblock', sub_color=None):
    stroke, fill = COL[kind]
    group(name)
    rect(x0, top, w, FH_, stroke, fill, rx=13, width=2.6)
    cx = x0 + w / 2
    if sub:
        text(cx, top + 32, title, fit(title, w - 16, FS_BLOCK), True, INK)
        text(cx, top + 60, sub, fit(sub, w - 16, FS_SUB), True, sub_color or MUTE)
    else:
        text(cx, top + FH_ / 2 + 9, title, fit(title, w - 16, FS_BLOCK), True, INK)
    end()


# column positions shared by both stage rows
GAP_C = 58
COLS_C = {}
xx = 250
for key, w in (('ra', 250), ('bb', 250), ('head', 290)):
    COLS_C[key] = (xx, w)
    xx += w + GAP_C
YHAT_X = xx + 6
LOSS_X = YHAT_X + 140


def stage_row(top, kind, tag, inputs, bb_sub, bb_kind, head_sub, head_kind, y_label, loss, loss_w, name):
    """Inputs -> RA -> backbone -> head -> prediction -> loss on one row; returns block top and mid."""
    b_top = top
    mid = b_top + FH_ / 2
    # inputs
    n = len(inputs)
    for i, (k, label) in enumerate(inputs):
        iy = mid - (n * 50 - 8) / 2 + i * 50
        group(f'{name}_input_{k}')
        rect(70, iy, 136, 42, COL[k][0], COL[k][1], rx=12, width=2.6)
        text(138, iy + 30, label, 24, True, INK)
        end()
        big_arrow([(208, iy + 21), (COLS_C['ra'][0] - 3, mid if n == 1 else iy + 21)], COL[k][0])
    fblock(*COLS_C['ra'], b_top, 'ra', 'Riemannian alignment', None, f'{name}_ra')
    fblock(*COLS_C['bb'], b_top, bb_kind, 'Backbone (a)', bb_sub, f'{name}_backbone',
           COL['tta'][0] if bb_kind == 'tta' else None)
    fblock(*COLS_C['head'], b_top, head_kind, 'Adaptation head (b)', head_sub, f'{name}_head',
           COL['tta'][0] if head_kind == 'tta' else None)
    for a, b in (('ra', 'bb'), ('bb', 'head')):
        xa = COLS_C[a][0] + COLS_C[a][1]
        big_arrow([(xa + 3, mid), (COLS_C[b][0] - 3, mid)], COL['bb'][0])
    xh = COLS_C['head'][0] + COLS_C['head'][1]
    big_arrow([(xh + 3, mid), (YHAT_X - 22, mid)], COL['bb'][0])
    text(YHAT_X, mid + 11, y_label, 30, True)
    big_arrow([(YHAT_X + 28, mid), (LOSS_X - loss_w / 2 - 3, mid)], COL['loss'][0], dashed=True)
    pill(LOSS_X, mid, loss, w=loss_w, name=f'{name}_loss')
    return b_top, mid


def center_c(key):
    return COLS_C[key][0] + COLS_C[key][1] / 2


frame_c_at = len(out)
# Stage 1
S1_TOP = C_TOP + 84
s1_loop_y = S1_TOP + 30
s1_top, s1_mid = stage_row(S1_TOP + 58, 'loss', 'Stage 1 · training',
                           [('src', 'X_{s}, y_{s}'), ('tgt', 'X_{t}')],
                           'all weights trained', 'bb', 'all weights and α', 'da', 'ŷ', 'ℒ_{total}', 140, 'stage1')
S1_BOTTOM = s1_top + FH_ + 22
# update loop above the blocks
big_arrow([(LOSS_X, s1_mid - 21), (LOSS_X, s1_loop_y), (center_c('bb'), s1_loop_y), (center_c('bb'), s1_top - 3)],
          COL['loss'][0], dashed=True)
big_arrow([(center_c('head'), s1_loop_y), (center_c('head'), s1_top - 3)], COL['loss'][0], dashed=True)
text(LOSS_X - 12, s1_loop_y - 10, 'update', 22, True, COL['loss'][0], anchor='end')

# Stage 2
S2_TOP = S1_BOTTOM + 70
s2_top, s2_mid = stage_row(S2_TOP + 22, 'tta', 'Stage 2 · IM-TTA', [('tgt', 'X_{t}')],
                           'only BN γ, β updated', 'tta', 'AutoDIAL stats re-estimated', 'tta', 'ŷ_{t}', 'ℒ_{IM}', 112,
                           'stage2')
s2_loop_y = s2_top + FH_ + 30
big_arrow([(LOSS_X, s2_mid + 21), (LOSS_X, s2_loop_y), (center_c('bb'), s2_loop_y), (center_c('bb'), s2_top + FH_ + 3)],
          COL['tta'][0], dashed=True)
text(LOSS_X - 12, s2_loop_y + 28, '×5 steps', 22, True, COL['tta'][0], anchor='end')
S2_BOTTOM = s2_loop_y + 44

# trained weights from Stage 1 to Stage 2
for key in ('bb', 'head'):
    big_arrow([(center_c(key), s1_top + FH_ + 3), (center_c(key), s2_top - 3)], '#8A94A3', dashed=True, width=2.2)
text(center_c('head') + 14, (S1_BOTTOM + S2_TOP) / 2 + 8, 'weights', 22, True, '#8A94A3', anchor='start')

# stage frames and panel (c) background go underneath
body = out[frame_c_at:]
del out[frame_c_at:]
C_BOTTOM = S2_BOTTOM + 24
frame(CX0, C_TOP, CX1, C_BOTTOM, PANEL_TINT['panel_c'], 'Two-stage adaptation', 'panel_c')
for top_, bottom_, kind, tag in ((S1_TOP, S1_BOTTOM, 'loss', 'Stage 1 · training'),
                                 (S2_TOP, S2_BOTTOM, 'tta', 'Stage 2 · IM-TTA')):
    stroke = COL[kind][0]
    rect(40, top_, CX1 - CX0 - 48, bottom_ - top_, stroke, stroke, rx=18, width=2.2, dashed=True, opacity=0.05)
    tw = text_width(tag, 22, True) + 28
    rect(64, top_ - 17, tw, 34, stroke, '#FFFFFF', rx=17, width=2.2)
    text(64 + tw / 2, top_ + 8, tag, 22, True, stroke)
out.extend(body)

# legend in the title row of (c)
lx = 700
for i, (kind, dashed, label) in enumerate([('src', False, 'source'), ('tgt', False, 'target'),
                                           ('loss', True, 'loss'), ('tta', True, 'TTA update')]):
    x0 = lx + i * 172
    big_arrow([(x0, C_TOP + 38), (x0 + 50, C_TOP + 38)], COL[kind][0], dashed=dashed)
    text(x0 + 62, C_TOP + 46, label, 22, True, INK, anchor='start')

H = C_BOTTOM + 16

# panel backgrounds for (a), (b) and the objective, inserted beneath everything drawn so far
backgrounds = []
_save = out[:]
out.clear()
frame(PA[0], TOP, PA[1], TOP_BOTTOM, PANEL_TINT['panel_a'], '(a) SSM backbone', 'panel_a')
frame(PB[0], TOP, PB[1], B_BOTTOM, PANEL_TINT['panel_b'], '(b) Adaptation head', 'panel_b')
group('stage1_objective')
rect(PB[0], OBJ_TOP, PB[1] - PB[0], OBJ_H, '#1A1A1A', '#FBF0EF', rx=26, width=3.2, dashed=True)
text(PB[0] + 26, OBJ_TOP + 42, 'Stage-1 objective', 28, True, COL['loss'][0], anchor='start')
text((PB[0] + PB[1]) / 2, OBJ_TOP + 96,
     'ℒ = ℒ_{CE} + λ_{IM}ℒ_{IM} + λ_{adv}ℒ_{adv} + λ_{MMD}ℒ_{MMD} + λ_{t}ℒ_{tMMD}', 26, True, INK)
text((PB[0] + PB[1]) / 2, OBJ_TOP + 136, 'λ_{IM} = 0.1,  λ_{adv} = 1,  λ_{MMD} = 0.5,  λ_{t} = 0.1', 21, True, MUTE)
end()
backgrounds = out[:]
out.clear()
out.extend(backgrounds + _save)

def write_svg(file_name, y0, y1):
    """Write the drawing, cropped to rows y0..y1, as one SVG file."""
    h = y1 - y0
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h:.0f}" '
           f'viewBox="0 {y0:.0f} {W} {h:.0f}">',
           f'<rect x="0" y="{y0:.0f}" width="{W}" height="{h:.0f}" fill="#FFFFFF"/>'] + out + ['</svg>']
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg) + '\n')
    print('wrote', file_name, f'{W} x {h:.0f}')


# full figure for editing in Figma, plus the two paper figures: (a)+(b) and (c)
target = Path(sys.argv[1])
write_svg(target, 0, H)
write_svg(target.with_name(target.stem + '_ab.svg'), 0, TOP_BOTTOM + 16)
write_svg(target.with_name(target.stem + '_c.svg'), C_TOP - 16, H)
