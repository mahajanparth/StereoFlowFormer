"""Generate the architecture figure: python docs/make_architecture.py docs/architecture.svg"""
import sys

W, H = 1100, 1744
out = []
a = out.append


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, cls, anchor='start'):
    a(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>')


def box(x, y, w, h, cls, rx=10, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ''
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"{d}/>')


def arrow(pts, cls='ln', marker='ah'):
    p = ' '.join(f'{x},{y}' for x, y in pts)
    a(f'<polyline points="{p}" class="{cls}" fill="none" marker-end="url(#{marker})"/>')


def line(pts, cls='ln'):
    p = ' '.join(f'{x},{y}' for x, y in pts)
    a(f'<polyline points="{p}" class="{cls}" fill="none"/>')


def chip(x, y, w, h, title, sub, cls, dim=None):
    box(x, y, w, h, cls, rx=7)
    if dim:
        text(x + w / 2, y + 22, title, 'ct', 'middle')
        text(x + w / 2, y + 39, sub, 'cs', 'middle')
        a(f'<line x1="{x + 8}" y1="{y + 48}" x2="{x + w - 8}" y2="{y + 48}" class="sep"/>')
        for j, d in enumerate(dim):
            text(x + w / 2, y + 64 + j * 15, d, 'dimc', 'middle')
    elif sub:
        text(x + w / 2, y + h / 2 - 3, title, 'ct', 'middle')
        text(x + w / 2, y + h / 2 + 13, sub, 'cs', 'middle')
    else:
        text(x + w / 2, y + h / 2 + 4, title, 'ct', 'middle')


def scene(x, y, w, h, shift=0, car=0):
    """Tiny road scene thumbnail; `shift` mimics the stereo baseline, `car` the object motion."""
    a(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 86 46">'
      f'<g transform="translate({shift},0)">'
      '<rect x="-10" width="106" height="46" fill="#cfe0f1"/>'
      '<rect x="-10" y="20" width="106" height="26" fill="#a3ae9f"/>'
      '<rect x="0" y="5" width="14" height="16" fill="#9aa3ad"/><rect x="16" y="10" width="10" height="11" fill="#b4bcc5"/>'
      '<polygon points="26,46 40,20 47,20 70,46" fill="#6b7079"/>'
      '<polygon points="42.6,22 44.4,22 46,30 41,30" fill="#e8e8e8"/>'
      '<polygon points="40.5,34 47.5,34 49,42 38.5,42" fill="#e8e8e8"/>'
      f'<g transform="translate({car},0)"><rect x="52" y="22" width="16" height="9" rx="2" fill="#c8483f"/>'
      '<rect x="55" y="18" width="10" height="6" rx="2" fill="#c8483f"/>'
      '<circle cx="56" cy="31" r="2" fill="#222"/><circle cx="64" cy="31" r="2" fill="#222"/></g>'
      '</g></svg>')


def flow_thumb(x, y, w, h, gid):
    a(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 86 46">'
      f'<rect width="86" height="46" fill="url(#{gid})"/>'
      '<rect x="0" y="20" width="86" height="26" fill="#ffffff" opacity="0.25"/>'
      '<rect x="52" y="18" width="16" height="13" rx="2" fill="#e3a43a"/>'
      '</svg>')


def disp_thumb(x, y, w, h, car=0):
    a(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 86 46">'
      '<rect width="86" height="46" fill="url(#gDisp)"/>'
      f'<rect x="{52 + car}" y="18" width="16" height="13" rx="2" fill="#f4a259"/>'
      '<rect x="0" y="5" width="14" height="16" fill="#3b1a63"/>'
      '</svg>')


def mask_thumb(x, y, w, h):
    a(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 86 46">'
      '<rect width="86" height="46" fill="#46576b"/>'
      '<rect x="52" y="18" width="16" height="13" rx="2" fill="#e5534b"/>'
      '<rect x="10" y="22" width="13" height="10" rx="2" fill="#f0b429"/>'
      '</svg>')


# ------------------------------------------------------------------ header
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">')
a('''<style>
  :root { --bg:#ffffff; --ink:#1f2328; --mut:#59636e; --ln:#8c959f; --box:#f6f8fa; --bst:#d0d7de;
          --fbg:#eef4fe; --fst:#4a7fd6; --fch:#ffffff;
          --sbg:#ecf7f1; --sst:#2f9e6b; --sch:#ffffff;
          --cbg:#fff6e6; --cst:#c98515;
          --hbg:#f4eefd; --hst:#8a5cd1;
          --tr:#c2410c; --trbg:#fff3ec; --dim:#0969da; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0d1117; --ink:#e6edf3; --mut:#9198a1; --ln:#6e7681; --box:#161b22; --bst:#3d444d;
            --fbg:#111f36; --fst:#6ea2ff; --fch:#0d1117;
            --sbg:#0f2a1f; --sst:#3fbf85; --sch:#0d1117;
            --cbg:#2b2110; --cst:#e3a43a;
            --hbg:#221a35; --hst:#a98bea;
            --tr:#f0883e; --trbg:#2a1a10; --dim:#58a6ff; }
  }
  .bg { fill:var(--bg); }
  .gen { fill:var(--box); stroke:var(--bst); stroke-width:1.2; }
  .fl { fill:var(--fbg); stroke:var(--fst); stroke-width:1.4; }
  .fc { fill:var(--fch); stroke:var(--fst); stroke-width:1; }
  .st { fill:var(--sbg); stroke:var(--sst); stroke-width:1.4; }
  .sc { fill:var(--sch); stroke:var(--sst); stroke-width:1; }
  .cx { fill:var(--cbg); stroke:var(--cst); stroke-width:1.4; }
  .cxc { fill:var(--bg); stroke:var(--cst); stroke-width:1; }
  .hd { fill:var(--hbg); stroke:var(--hst); stroke-width:1.4; }
  .tr { fill:var(--trbg); stroke:var(--tr); stroke-width:1.4; }
  .trc { fill:var(--bg); stroke:var(--tr); stroke-width:1; }
  .ln { stroke:var(--ln); stroke-width:1.6; }
  .lnt { stroke:var(--tr); stroke-width:1.6; stroke-dasharray:5 4; }
  .ahf { fill:var(--ln); }
  .ahtf { fill:var(--tr); }
  .dot { fill:var(--ln); }
  .sep { stroke:var(--bst); stroke-width:1; }
  text { fill:var(--ink); }
  .h1 { font-size:21px; font-weight:700; }
  .h1s { font-size:13.5px; fill:var(--mut); }
  .t { font-size:15px; font-weight:700; }
  .s { font-size:12px; fill:var(--mut); }
  .b { font-size:12.5px; }
  .bb { font-size:12.5px; font-weight:600; }
  .m { font-size:12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .ct { font-size:12px; font-weight:600; }
  .cs { font-size:10.5px; fill:var(--mut); }
  .lab { font-size:11.5px; fill:var(--mut); }
  .badge { font-size:11px; fill:var(--mut); font-style:italic; }
  .dim { font-size:11.5px; font-weight:600; fill:var(--dim); }
  .dimc { font-size:10.5px; font-weight:600; fill:var(--dim); }
  .trt { font-size:14px; font-weight:700; fill:var(--tr); }
</style>''')
a('<defs>'
  '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
  '<path d="M0,0 L10,5 L0,10 z" class="ahf"/></marker>'
  '<marker id="aht" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
  '<path d="M0,0 L10,5 L0,10 z" class="ahtf"/></marker>'
  '<linearGradient id="gFw" x1="0" x2="1"><stop offset="0" stop-color="#c58be0"/><stop offset="0.45" stop-color="#f3eef6"/>'
  '<stop offset="0.55" stop-color="#eef6f0"/><stop offset="1" stop-color="#7fd0a0"/></linearGradient>'
  '<linearGradient id="gBw" x1="0" x2="1"><stop offset="0" stop-color="#7fd0a0"/><stop offset="0.45" stop-color="#eef6f0"/>'
  '<stop offset="0.55" stop-color="#f3eef6"/><stop offset="1" stop-color="#c58be0"/></linearGradient>'
  '<linearGradient id="gDisp" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#1b0c41"/>'
  '<stop offset="0.45" stop-color="#5c126e"/><stop offset="0.75" stop-color="#b63679"/><stop offset="1" stop-color="#fb9b06"/></linearGradient>'
  '</defs>')
a(f'<rect width="{W}" height="{H}" class="bg"/>')

text(16, 36, 'StereoFlowFormer', 'h1')

# legend for the shape notation
text(16, 60, 'Shapes are per sample, C @ s  =  C × (s·H) × (s·W) for an H × W input.  '
     'At the 320 × 960 training crop: ⅛ → 40 × 120,  ¼ → 80 × 240,  1/16 → 20 × 60.', 'lab')


def card(x, y, w, h, title, lines, cls, dims=(), title_cls='ct'):
    """Chip with a title, grey detail lines and blue shape lines, all centred."""
    box(x, y, w, h, cls, rx=7)
    cx = x + w / 2
    text(cx, y + 19, title, title_cls, 'middle')
    yy = y + 35
    for s in lines:
        text(cx, yy, s, 'cs', 'middle')
        yy += 14
    if dims:
        a(f'<line x1="{x + 8}" y1="{yy - 6}" x2="{x + w - 8}" y2="{yy - 6}" class="sep"/>')
        yy += 9
        for d in dims:
            text(cx, yy, d, 'dimc', 'middle')
            yy += 14


# ------------------------------------------------------------------ row 1: inputs + matching branches
R1, BH, GAP = 76, 296, 24          # top of row 1, branch-box height, gap between branch boxes
CW, CS, CX0 = 90, 98, 208          # chip width, chip pitch, first chip x
LANE8, LANE4, LH = 40, 162, 108    # lane offsets inside a branch box, lane height


def col(i):
    return CX0 + i * CS


box(16, R1, 110, 2 * BH + GAP, 'gen')
text(28, R1 + 24, 'Inputs', 't')
text(28, R1 + 41, '3 × H × W each', 'dim')
text(28, R1 + 57, 'normalised,', 'cs')
text(28, R1 + 70, 'padded to /32', 'cs')
labels = [('Lₜ', 0, 0), ('Rₜ', -4, 0), ('Lₜ₊₁', 0, -6), ('Rₜ₊₁', -4, -6)]
for i, (lab, sh, car) in enumerate(labels):
    y = R1 + 100 + i * 128
    scene(28, y, 86, 46, sh, car)
    text(71, y + 61, lab, 'bb', 'middle')

branches = [
    (R1, 'fl', 'fc', 'Flow branch  ·  Lₜ ↔ Lₜ₊₁', 'transformer 6 layers · 1 head · d 128 · FFN ×4  ·  init: UniMatch GMFlow',
     [('Transformer', ['6 layers, swin', '2 × 2 windows'], ['128 @ ⅛']),
      ('Global corr', ['all pairs, soft-', 'max; bw = transp.'], ['hw × hw', '→ flow 2 @ ⅛']),
      ('Propagation', ['global self-attn', 'fills occlusions'], ['2 @ ⅛'])],
     [('Warp + attn', ['warp Lₜ₊₁ feats,', 'swin 8 × 8'], ['128 @ ¼']),
      ('Local corr', ['9 × 9 window,', 'residual flow'], ['81 → 2 @ ¼']),
      ('Propagation', ['3 × 3 local', 'self-attn'], ['2 @ ¼']),
      ('GRU refine', ['×6 iterations,', 'corr r = 4'], ['hidden 128', 'corr 81 @ ¼']),
      ('Upsample ×4', ['convex, 3 × 3', 'neighbours'], ['mask 144 @ ¼', '→ 2 × H × W'])],
     'out: flow fw / bw 2 × H × W  ·  flow 2 @ ¼  ·  Lₜ feature 128 @ ¼  ·  fw and bw run as one batch at ¼  (hw = H/8 · W/8)'),
    (R1 + BH + GAP, 'st', 'sc', 'Stereo branch  ·  (Lₜ, Rₜ), (Lₜ₊₁, Rₜ₊₁)',
     'cross-attention along image rows  ·  init: UniMatch GMStereo',
     [('Transformer', ['self: swin 2 × 2', 'cross: 1D rows'], ['128 @ ⅛']),
      ('Epipolar corr', ['softmax along', 'row, disp ≥ 0'], ['h × w × w', '→ disp 1 @ ⅛']),
      ('Propagation', ['global self-attn', 'fills occlusions'], ['1 @ ⅛'])],
     [('Warp + attn', ['shift R feats,', 'swin 8 × 8'], ['128 @ ¼']),
      ('Local corr', ['1D, 9 px', 'along the row'], ['9 → 1 @ ¼']),
      ('Propagation', ['3 × 3 local', 'self-attn'], ['1 @ ¼']),
      ('GRU refine', ['×3 iterations,', 'corr r = 4'], ['hidden 128', 'corr 81 @ ¼']),
      ('Upsample ×4', ['convex, 3 × 3', 'neighbours'], ['mask 144 @ ¼', '→ 1 × H × W'])],
     'out: disparity 1 × H × W at t and t+1 (both pairs as one batch)  ·  disparity 1 @ ¼  ·  Lₜ feature 128 @ ¼'),
]
for y0, bcls, ccls, title, badge, lane8, lane4, foot in branches:
    box(156, y0, 640, BH, bcls)
    text(170, y0 + 24, title, 't')
    text(784, y0 + 24, badge, 'badge', 'end')
    for off, lab in [(LANE8, '⅛'), (LANE4, '¼')]:
        text(184, y0 + off + LH / 2 + 2, lab, 't', 'middle')
        text(184, y0 + off + LH / 2 + 18, 'res', 'cs', 'middle')
    # shared CNN spans both lanes
    card(col(0), y0 + LANE8, CW, LANE4 + LH - LANE8, 'CNN encoder',
         ['ResNet, shared', 'by both images', '', 'outputs at ⅛', 'and at ¼'], ccls, ['128 @ ⅛', '128 @ ¼'])
    for i, (ct, ls, ds) in enumerate(lane8):
        card(col(i + 1), y0 + LANE8, CW, LH, ct, ls, ccls, ds)
        arrow([(col(i + 1) - 8, y0 + LANE8 + 30), (col(i + 1) - 1, y0 + LANE8 + 30)])
    for i, (ct, ls, ds) in enumerate(lane4):
        card(col(i + 1), y0 + LANE4, CW, LH, ct, ls, ccls, ds)
        arrow([(col(i + 1) - 8, y0 + LANE4 + 30), (col(i + 1) - 1, y0 + LANE4 + 30)])
    # coarse estimate -> fine lane (upsampled x2, used to warp the second image's features)
    xs, xd = col(3) + CW / 2, col(1) + CW / 2
    arrow([(xs, y0 + LANE8 + LH), (xs, y0 + LANE8 + LH + 7), (xd, y0 + LANE8 + LH + 7), (xd, y0 + LANE4 - 1)])
    text(col(3) + CW / 2 + 6, y0 + LANE8 + LH + 11, 'estimate ↑×2', 'cs')
    text(170, y0 + BH - 12, foot, 'lab')
    arrow([(126, y0 + LANE8 + 20), (col(0) - 1, y0 + LANE8 + 20)])

# dense outputs
F0, S0 = R1, R1 + BH + GAP
OUT_Y = LANE4 + 30                  # where the final estimate leaves the branch box
for y0, cls, title, gids, labs, dim, uses in [
        (F0, 'fl', 'Optical flow', ('gFw', 'gBw'), ('forward', 'backward'), '2 × H × W each (px)',
         ['geometry @ ¼ (row below)', 'photometric + occlusion masks', 'VO: track points, fb < 1 px']),
        (S0, 'st', 'Disparity → depth', None, ('dispₜ', 'dispₜ₊₁'), '1 × H × W each (px)',
         ['depth = f · b / disp', 'geometry @ ¼, stereo photometric', 'VO: landmarks, map'])]:
    box(836, y0, 248, BH, cls)
    text(850, y0 + 24, title, 't')
    if gids:
        flow_thumb(850, y0 + 40, 104, 56, gids[0])
        flow_thumb(966, y0 + 40, 104, 56, gids[1])
    else:
        disp_thumb(850, y0 + 40, 104, 56)
        disp_thumb(966, y0 + 40, 104, 56, -6)
    text(902, y0 + 114, labs[0], 'lab', 'middle')
    text(1018, y0 + 114, labs[1], 'lab', 'middle')
    text(850, y0 + 142, dim, 'dim')
    a(f'<line x1="850" y1="{y0 + 160}" x2="1070" y2="{y0 + 160}" class="sep"/>')
    text(850, y0 + 182, 'used by', 'bb')
    for j, s in enumerate(uses):
        text(850, y0 + 204 + j * 20, s, 'b')
    arrow([(796, y0 + OUT_Y), (835, y0 + OUT_Y)])

# bus from the branches down to the heads
BUS, R2 = S0 + BH + 30, S0 + BH + 70
line([(816, F0 + OUT_Y), (816, BUS), (136, BUS)])
a(f'<circle cx="816" cy="{F0 + OUT_Y}" r="3.2" class="dot"/><circle cx="816" cy="{S0 + OUT_Y}" r="3.2" class="dot"/>')
arrow([(136, BUS), (136, R2 - 1)])
arrow([(416, BUS), (416, R2 - 1)])
a(f'<circle cx="416" cy="{BUS}" r="3.2" class="dot"/>')
text(476, BUS - 7, 'flow 2 @ ¼  ·  dispₜ, dispₜ₊₁ 1 @ ¼  ·  Lₜ features 128 @ ¼ from each transformer', 'lab', 'middle')

# ------------------------------------------------------------------ row 2: geometry, context, heads
GH = 400                           # geometry / context box height
box(16, R2, 240, GH, 'gen')
text(30, R2 + 24, 'Geometry @ ¼ res', 't')
text(30, R2 + 44, 'closed form, fp32, no learned weights', 'cs')
for i, (s, d) in enumerate([('depth = f · b / disp', '1 @ ¼'), ('Xₜ = depth · K⁻¹ p', '3 @ ¼'),
                            ('scene flow =', ''), ('  Xₜ₊₁(p + flow) − Xₜ', '3 @ ¼')]):
    text(30, R2 + 72 + i * 22, s, 'm')
    if d:
        text(242, R2 + 72 + i * 22, d, 'dim', 'end')
text(30, R2 + 160, 'K rescaled to the ¼ grid; Xₜ₊₁ sampled', 'cs')
text(30, R2 + 174, 'bilinearly at p + flow (both frames)', 'cs')
a(f'<line x1="30" y1="{R2 + 188}" x2="242" y2="{R2 + 188}" class="sep"/>')
text(30, R2 + 210, 'Geometric encoding', 'bb')
text(242, R2 + 210, '8 @ ¼', 'dim', 'end')
for i, (s, n, c) in enumerate([('flow / f', '× 10', '2'), ('inverse depth', '× 5, depth ≥ 0.5 m', '1'),
                               ('ray directions K⁻¹ p', '', '2'), ('3D scene flow', 'clip ± 5 m', '3')]):
    y = R2 + 236 + i * 30
    text(30, y, s, 'b')
    if n:
        text(30, y + 13, n, 'cs')
    text(242, y, c, 'lab', 'end')
text(30, R2 + 368, 'gradient stopped: the heads cannot', 'cs')
text(30, R2 + 382, 'change the flow or depth estimates', 'cs')

box(296, R2, 240, GH, 'cx')
text(310, R2 + 24, 'Context encoder', 't')
for x, w, cls, lab in [(310, 96, 'fc', 'flow 128'), (410, 96, 'sc', 'stereo 128'), (510, 12, 'gen', '')]:
    a(f'<rect x="{x}" y="{R2 + 40}" width="{w}" height="16" rx="3" class="{cls}"/>')
    if lab:
        text(x + w / 2, R2 + 52, lab, 'cs', 'middle')
text(516, R2 + 70, 'geo 8', 'cs', 'middle')
text(310, R2 + 70, 'concat → 264 @ ¼', 'dim')
arrow([(416, R2 + 76), (416, R2 + 88)])
card(316, R2 + 90, 200, 40, 'conv 3 × 3, 264 → 128', ['GroupNorm (8 groups) · GELU'], 'cxc')
arrow([(416, R2 + 130), (416, R2 + 142)])
card(316, R2 + 144, 200, 40, '2 × ResBlock', ['(conv 3 × 3 · GN · GELU) × 2 + skip'], 'cxc')
text(416, R2 + 202, 'pixel features  128 @ ¼', 'dim', 'middle')
arrow([(416, R2 + 208), (416, R2 + 220)])
card(316, R2 + 222, 200, 40, '2 × conv 3 × 3, stride 2', ['128 → 192 → 192 · GN · GELU'], 'cxc')
arrow([(416, R2 + 262), (416, R2 + 274)])
card(316, R2 + 276, 200, 40, '+ sine position encoding', ['2D, 192-d, shared by both heads'], 'cxc')
text(416, R2 + 336, 'tokens  192 @ 1/16', 'dim', 'middle')
text(416, R2 + 352, 'N = HW / 256 tokens (1200 at the crop)', 'cs', 'middle')
text(416, R2 + 384, 'tokens → both heads, pixels → mask head', 'lab', 'middle')

HH, HG = 185, 30                   # head-box height, gap between heads
P0, M0 = R2, R2 + HH + HG
MID = R2 + GH // 2
arrow([(256, MID), (295, MID)])
line([(536, MID), (556, MID)])
line([(556, P0 + 80), (556, M0 + 80)])
arrow([(556, P0 + 80), (575, P0 + 80)])
arrow([(556, M0 + 80), (575, M0 + 80)])


def lines_at(x, y, rows, step=18):
    for i, (s, cls) in enumerate(rows):
        text(x, y + i * step, s, cls)


box(576, P0, 250, HH, 'hd')
text(590, P0 + 24, 'Pose head', 't')
lines_at(590, P0 + 48, [
    ('bg mask (slot 0) avg-pooled to 1/16,', 'b'),
    ('Linear 1 → 192, added to tokens + pos', 'b'),
    ('prepend a learned pose token', 'b'),
    ('4 pre-norm encoder layers', 'b'),
    ('    8 heads, FFN 768, no dropout', 'cs'),
    ('pose token → LayerNorm → Linear → 12', 'b'),
    ('rot × 0.05 rad, t × 1 m, log-var ∈ [−10, 10]', 'cs'),
])
text(590, P0 + 176, 'in (N + 1) × 192  →  out 12', 'dim')

box(576, M0, 250, HH, 'hd')
text(590, M0 + 24, 'Mask head (Mask2Former-lite)', 't')
lines_at(590, M0 + 48, [
    ('8 learned slot queries + query pos', 'b'),
    ('3 × [self-attn → cross-attn → FFN]', 'b'),
    ('    pre-norm, 8 heads, keys = 1/16 tokens', 'cs'),
    ('MLP → one embedding per slot', 'b'),
    ('logits = embedding · proj(pixels) / √128', 'b'),
    ('softmax over slots, slot 0 = background', 'b'),
    ('bilinear ×4 to full resolution', 'cs'),
])
text(590, M0 + 176, 'queries 8 × 192 → logits 8 @ ¼', 'dim')

arrow([(700, M0), (700, P0 + HH + 1)])
text(708, M0 - 11, 'bg slot, pooled to N × 1, detached', 'lab')

box(866, P0, 218, HH, 'hd')
text(880, P0 + 24, 'Ego-motion', 't')
for cx, op in [(900, 0.55), (1010, 1.0)]:
    a(f'<g opacity="{op}"><polygon points="{cx - 14},{P0 + 44} {cx + 14},{P0 + 44} {cx + 6},{P0 + 58} {cx - 6},{P0 + 58}" class="hd"/>'
      f'<rect x="{cx - 9}" y="{P0 + 58}" width="18" height="12" rx="2" class="hd"/></g>')
arrow([(922, P0 + 54), (988, P0 + 54)])
text(955, P0 + 46, 'R, t', 'lab', 'middle')
text(880, P0 + 98, 'rotvec 3 (rad) + t 3 (m)', 'dim')
text(880, P0 + 116, 'log-variance 6', 'dim')
lines_at(880, P0 + 142, [('Lₜ → Lₜ₊₁ camera motion; the', 'cs'),
                         ('variance weights BA pose priors', 'cs')], step=14)

box(866, M0, 218, HH, 'hd')
text(880, M0 + 24, 'Motion-instance masks', 't')
mask_thumb(880, M0 + 38, 96, 54)
text(988, M0 + 54, 'red, yellow:', 'lab')
text(988, M0 + 70, 'moving objects', 'lab')
text(988, M0 + 86, 'slate: static bg', 'lab')
text(880, M0 + 116, '8 × H × W, soft', 'dim')
lines_at(880, M0 + 142, [('VO keeps tracks with bg > 0.5;', 'cs'),
                         ('the map keeps static points only', 'cs')], step=14)

arrow([(826, P0 + 80), (865, P0 + 80)])
arrow([(826, M0 + 80), (865, M0 + 80)])

# ------------------------------------------------------------------ row 3: training
R3, TH = R2 + GH + 36, 372
box(16, R3, 1068, TH, 'tr', dash=True)
text(30, R3 + 24, 'Training  ·  no motion-mask labels, no KITTI poses', 'trt')
text(30, R3 + 48, 'Mask pseudo-labels, recomputed every batch from the network’s own ¼-res geometry', 'bb')
steps = [
    ('Dense robust PnP', ['Gauss-Newton, 12 iterations', 'Xₜ ↔ p + flow  →  ego R, t', 'background dominates the scene']),
    ('Rigid-flow residual', ['r = ‖flow − flow(R, t)‖', 'u = change of the rigid flow for', 'a 0.25 px disparity error']),
    ('Two thresholds (¼ px)', ['moving: r > max(1, 0.08‖f‖) + 2u', 'static:  r < max(0.4, 0.04‖f‖) + u', 'between / > 40 m: ignore or static']),
    ('Clean + instances', ['3 × 3 majority, erode, ignore', 'edge ring; connected components', '≥ 24 px, largest 7 kept']),
    ('Hungarian CE + Dice', ['slots 1…7 matched to instances', 'fg CE weight 5, slot 0 = static', '+ 0.05 edge-aware smoothness']),
]
for i, (st, ls) in enumerate(steps):
    x = 30 + i * 210
    card(x, R3 + 60, 196, 82, st, ls, 'trc')
    if i:
        arrow([(x - 13, R3 + 101), (x - 1, R3 + 101)], 'lnt', 'aht')
arrow([(700, R3), (700, M0 + HH + 1)], 'lnt', 'aht')
text(708, R3 - 10, 'supervises the mask head', 'lab')

a(f'<line x1="30" y1="{R3 + 158}" x2="1070" y2="{R3 + 158}" class="sep"/>')
LY = R3 + 182
text(30, LY, 'Losses (weight)', 'bb')
cols = [
    (30, 'Supervised · Virtual KITTI 2', [
        ('flow: sequence L1 over all estimates, γ = 0.9', '1.0'),
        ('disparity: sequence L1, γ = 0.9', '1.0'),
        ('pose: Gaussian NLL with log-var, rot ×20', '1.0'),
        ('masks: pseudo-label CE + Dice', '0.2')]),
    (370, 'Self-supervised · KITTI odometry 00–08', [
        ('flow: SSIM + L1 (α 0.85), fb occlusion mask', '1.0'),
        ('stereo: R → L warp, SSIM + L1, t and t+1', '1.0'),
        ('edge-aware smoothness, flow and disp', '0.05'),
        ('pose: rigid flow ≈ flow on background', '0.2'),
        ('pose ≈ dense-PnP ego-motion', '0.05'),
        ('masks: pseudo-label CE + Dice', '0.2')]),
]
for x, head, rows in cols:
    text(x, LY + 22, head, 'lab')
    for j, (s, wgt) in enumerate(rows):
        text(x, LY + 44 + j * 20, s, 'b')
        text(x + 310, LY + 44 + j * 20, wgt, 'dim', 'end')

SX = 720
text(SX, LY, 'Stages', 'bb')
stages = [
    ('1a', 'heads only, branches frozen · VKITTI2', '15k it, lr 3e-4'),
    ('1b', 'everything jointly · VKITTI2', '10k it, lr 1e-4, branches ×0.1 for 2k'),
    ('2', 'self-supervised KITTI + 25 % VKITTI2', '15k it, lr 5e-5 (hurt flow / disp)'),
    ('2b', 'heads only on KITTI, 1b branches', '5k it, lr 1e-4 · default checkpoint'),
]
for j, (tag, s1, s2) in enumerate(stages):
    y = LY + 12 + j * 42
    box(SX, y, 350, 36, 'trc', rx=6)
    text(SX + 20, y + 23, tag, 'trt', 'middle')
    text(SX + 40, y + 15, s1, 'cs')
    text(SX + 40, y + 29, s2, 'dimc')

# ------------------------------------------------------------------ row 4: VO back-end
R4, VH = R3 + TH + 24, 132
box(16, R4, 1068, VH, 'gen')
text(30, R4 + 24, 'Stereo VO back-end  ·  per frame, consumes flow, disparity, masks and the pose head', 't')
vo = [
    ('Track', ['move points by fw flow,', 'fb < 1 px, bg > 0.5']),
    ('Initial pose', ['dense robust PnP,', 'flow + disp, stride 4']),
    ('PnP-RANSAC', ['200 it, 2 px, ≥ 40', 'inliers, LM refine']),
    ('Keyframe?', ['tracks < 50 %, or', '> 60 px, or 10 frames']),
    ('New landmarks', ['top up to 1500 points', 'disp > 2 px, depth < 40 m']),
    ('Local BA', ['8 keyframes, 15 LM it,', 'Huber + pose priors']),
    ('Global BA + map', ['30 LM it at the end,', 'points / TSDF, 0.1 m']),
]
for i, (st, ls) in enumerate(vo):
    x = 30 + i * 150
    card(x, R4 + 42, 140, 70, st, ls, 'fc' if i < 5 else 'cxc')
    if i:
        arrow([(x - 9, R4 + 77), (x - 1, R4 + 77)])

a('</svg>')
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
