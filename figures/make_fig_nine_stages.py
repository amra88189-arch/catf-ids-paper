#!/usr/bin/env python3
"""Draw the §5.1 figure (the nine stages of CATF-IDS) as SVG, PDF and PNG.

    python make_fig_nine_stages.py      # needs: pip install cairosvg  (for PDF/PNG)

Drawn from primitives only (no icon sets, no logos). Stage numbers follow the
nine-stage table of §5.1. Change a label here and re-run.
"""
import os

W, H = 1455, 905
FONT = "Times New Roman, Liberation Serif, Times, serif"
NAVY, GREY, INK, MUTE = "#1F4D78", "#7F7F7F", "#222222", "#8A8A8A"
PANEL, DL = "#F2F3F5", "#DCE6F1"
FS = 1.1          # global text scale: legibility at full page width

out = []
add = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=15, anchor="start", weight="normal", style="normal", fill=INK, rot=None):
    size = round(size * FS, 1)
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="{style}" fill="{fill}"{tr}>{esc(s)}</text>')


_FONTS = {}


def measure(t, size, italic=False, bold=False):
    """Text width from Liberation Serif metrics (Times-compatible); rough fallback otherwise."""
    try:
        from PIL import ImageFont
        name = "LiberationSerif-" + ("BoldItalic" if bold and italic else "Bold" if bold
                                     else "Italic" if italic else "Regular") + ".ttf"
        key = (name, round(size * 4))
        if key not in _FONTS:
            for d in ("/usr/share/fonts/truetype/liberation/", "C:/Windows/Fonts/", ""):
                try:
                    _FONTS[key] = ImageFont.truetype(d + name, key[1]); break
                except OSError:
                    continue
            else:
                raise OSError
        return _FONTS[key].getlength(t) / 4
    except Exception:
        return 0.48 * size * len(t)


def rich(x, y, parts, size=15, anchor="start", fill=INK, style="normal", weight="normal"):
    """parts: (text, kind), kind '' | 'it' (italic) | 'sub' (subscript). Laid out piece by
    piece, because renderers do not agree on how tspans combine with text-anchor."""
    size = round(size * FS, 1)
    pieces = []
    prev = ""
    for t, kind in parts:
        sz = 0.72 * size if kind == "sub" else size
        it = kind == "it" or style == "italic"
        pieces.append((t, kind, sz, it, measure(t, sz, it, weight == "bold")))
    total = sum(p[4] for p in pieces)
    gaps = [1.8 if (p[1] == "it" and i and pieces[i - 1][0].endswith(" ")) else 0
            for i, p in enumerate(pieces)]          # room for an italic letter after a space
    total += sum(gaps)
    cx = x - total / 2 if anchor == "middle" else x - total if anchor == "end" else x
    for (t, kind, sz, it, w), g in zip(pieces, gaps):
        cx += g
        yy = y + (0.28 * size if kind == "sub" else 0)
        add(f'<text x="{cx:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="{sz:.1f}" '
            f'font-style="{"italic" if it else "normal"}" font-weight="{weight}" fill="{fill}" '
            f'xml:space="preserve">{esc(t)}</text>')
        cx += w


def lines(x, y, rows, size=15, lh=None, **kw):
    lh = lh or size * 1.22
    for i, r in enumerate(rows):
        text(x, y + i * lh, r, size=size, **kw)


def box(x, y, w, h, stroke=INK, fill="white", sw=1.4, r=4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def path(d, color=GREY, width=2.0, dash=None, head="hg"):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    hd = f' marker-end="url(#{head})"' if head else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{ds}{hd} '
        f'stroke-linejoin="round"/>')


def badge(x, y, n, muted=False):
    c = "#A0A0A0" if muted else NAVY
    add(f'<circle cx="{x}" cy="{y}" r="12" fill="{c}"/>')
    text(x, y + 5.5, str(n), size=15, anchor="middle", weight="bold", fill="white")


def tag(cx, y, rows):
    w = max(measure(r, 11.5 * FS, True) for r in rows) + 12
    box(cx - w / 2, y, w, 15 * len(rows) + 6, stroke="#C9C9C9", fill="#EDEDED", sw=0.8, r=3)
    lines(cx, y + 15, rows, size=11.5, anchor="middle", style="italic", fill="#555", lh=15)


S = f'fill="none" stroke="{INK}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'


def icon_network(x, y):
    pts = [(x - 16, y - 10), (x - 16, y + 10), (x, y), (x + 16, y - 12), (x + 16, y), (x + 16, y + 12)]
    for a, b in ((0, 2), (1, 2), (2, 3), (2, 4), (2, 5)):
        add(f'<line x1="{pts[a][0]}" y1="{pts[a][1]}" x2="{pts[b][0]}" y2="{pts[b][1]}" {S}/>')
    for px, py in pts:
        add(f'<circle cx="{px}" cy="{py}" r="3.4" fill="white" stroke="{INK}" stroke-width="1.6"/>')


def icon_chip(x, y):
    add(f'<rect x="{x-12}" y="{y-12}" width="24" height="24" rx="2" {S}/>'
        f'<rect x="{x-5}" y="{y-5}" width="10" height="10" {S}/>')
    for k in (-7, 0, 7):
        add(f'<line x1="{x-17}" y1="{y+k}" x2="{x-12}" y2="{y+k}" {S}/>'
            f'<line x1="{x+12}" y1="{y+k}" x2="{x+17}" y2="{y+k}" {S}/>'
            f'<line x1="{x+k}" y1="{y-17}" x2="{x+k}" y2="{y-12}" {S}/>'
            f'<line x1="{x+k}" y1="{y+12}" x2="{x+k}" y2="{y+17}" {S}/>')


def icon_terminal(x, y):
    add(f'<rect x="{x-17}" y="{y-13}" width="34" height="24" rx="2" {S}/>'
        f'<path d="M{x-10},{y-5} l5,4 l-5,4" {S}/><line x1="{x-2}" y1="{y+4}" x2="{x+7}" y2="{y+4}" {S}/>'
        f'<line x1="{x-6}" y1="{y+16}" x2="{x+6}" y2="{y+16}" {S}/>')


def icon_tree(x, y):
    nodes = [(x, y - 13), (x - 11, y - 1), (x + 11, y - 1), (x - 17, y + 11), (x - 5, y + 11),
             (x + 5, y + 11), (x + 17, y + 11)]
    for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        add(f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}" {S}/>')
    for px, py in nodes:
        add(f'<circle cx="{px}" cy="{py}" r="3" fill="white" stroke="{INK}" stroke-width="1.5"/>')


def icon_waves(x, y):
    for k in (-7, 0, 7):
        add(f'<path d="M{x-14},{y+k} q7,-5 14,0 t14,0" {S}/>')


# ── canvas ───────────────────────────────────────────────────────────
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>'
    f'<marker id="hg" markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker>'
    f'<marker id="hn" markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/></marker>'
    f'<marker id="hm" markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTE}"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# column panels and titles
cols = [(8, 170, "Inputs"), (184, 350, "Per-modality layer"), (540, 225, "Fusion"),
        (771, 523, "Adaptive decision layer"), (1300, 150, "Output")]
for x, w, t in cols:
    box(x, 8, w, 862, stroke="none", fill=PANEL, sw=0, r=10)
    text(x + w / 2, 46, t, size=24, anchor="middle")
text(1032, 64, "one decision path for cross-validation and the stream", size=12.5,
     anchor="middle", style="italic", fill="#555")

# ── A: inputs ────────────────────────────────────────────────────────
ROWS = {"net": 160, "dev": 340, "host": 520}
for key, lab, ic in (("net", ["Network", "flows"], icon_network),
                     ("dev", ["Device", "telemetry", "(IoT)"], icon_chip),
                     ("host", ["Host", "activity", "(Linux)"], icon_terminal)):
    cy = ROWS[key]
    box(20, cy - 55, 146, 110, stroke=INK, sw=1.5)
    ic(93, cy - 29)
    lines(93, cy + 12 if len(lab) == 2 else cy + 6, lab, size=17, anchor="middle", lh=19)

# ── B: behavioural physics and the three forests ─────────────────────
badge(350, 94, 1)
lines(372, 88, ["Layer models (random forests)"], size=17)
box(196, 105, 146, 110, stroke=INK, sw=1.5)
badge(196, 105, 2)
icon_waves(269, 128)
lines(269, 156, ["Behavioural", "physics"], size=17, anchor="middle", lh=18)
lines(269, 194, ["session features per", "source, 60 s window"], size=12.5, anchor="middle", lh=14)
for key, name, nf in (("net", "Network forest", "32"), ("dev", "Device forest", "7"),
                      ("host", "Host forest", "10")):
    cy = ROWS[key]
    box(362, cy - 50, 150, 100, stroke=INK, sw=1.5)
    icon_tree(437, cy - 22)
    text(437, cy + 17, name, size=17, anchor="middle")
    text(437, cy + 37, f"· {nf} features", size=15, anchor="middle")
# stage 1 groups the three forests
box(350, 94, 172, 492, stroke=NAVY, fill="none", sw=1.4, r=8)
path(f"M166,{ROWS['net']} H194")
path(f"M342,{ROWS['net']} H360")
path(f"M166,{ROWS['dev']} H360")
path(f"M166,{ROWS['host']} H360")
for key, sub in (("net", "net"), ("dev", "iot"), ("host", "log")):
    cy = ROWS[key]
    path(f"M512,{cy} H552")
    rich(531, cy - 8, [("p", "it"), (sub, "sub")], size=16, anchor="middle")

# ── C: evidence vector and fusion classifier ─────────────────────────
EVX, EVY, EVW, EVH = 554, 105, 46, 470
box(EVX, EVY, EVW, EVH, stroke=INK, sw=1.5)
text(EVX + 19, EVY + EVH / 2, "Evidence vector (7)", size=17, anchor="middle", rot=-90)
text(EVX + 37, EVY + EVH / 2, "p and log-odds per modality + physical deviation Δ",
     size=12.5, anchor="middle", rot=-90)
FX, FY, FW, FH = 620, 238, 132, 124
box(FX, FY, FW, FH, stroke=NAVY, sw=1.8)
badge(FX, FY, 3)
lines(FX + FW / 2, FY + 30, ["Fusion", "classifier"], size=17, anchor="middle", lh=18)
lines(FX + FW / 2, FY + 72, ["logistic regression", "ensemble of the 5", "validated fold fits"], size=13, anchor="middle", lh=15)
path(f"M600,300 H618")

# ── D: adaptive decision layer ───────────────────────────────────────
box(781, 70, 507, 636, stroke=NAVY, fill=DL, sw=1.6, r=14)
# stage 6 and 7
box(792, 86, 225, 108, stroke=NAVY, sw=1.6)
badge(792, 86, 6)
text(904, 110, "Adaptive threshold", size=17, anchor="middle")
text(904, 130, "base fitted (F-β), floor 0.35;", size=13.5, anchor="middle")
text(904, 147, "drift update vs training mean", size=13.5, anchor="middle")
tag(904, 158, ["high threshold at floor on 96.5%"])
box(1037, 86, 240, 108, stroke=NAVY, sw=1.6)
badge(1037, 86, 7)
text(1157, 110, "Contextual engine", size=17, anchor="middle")
text(1157, 130, "per-event shift from drift,", size=13.5, anchor="middle")
text(1157, 147, "disagreement and trust (≤ ±0.12)", size=13.5, anchor="middle")
tag(1157, 158, ["low threshold moves: ≤ 0.08 on 34%"])
# tau box
box(1100, 210, 114, 34, stroke=NAVY, sw=1.4)
rich(1157, 233, [("τ", ""), ("high", "sub"), (" / τ", ""), ("low", "sub")], size=15.5, anchor="middle")
path("M904,194 V227 H1098", color=NAVY, head="hn", width=1.6)
path("M1157,194 V208", color=NAVY, head="hn", width=1.6)
# band test diamond
DX, DY, DW, DH = 1157, 300, 92, 44
add(f'<path d="M{DX},{DY-DH} L{DX+DW},{DY} L{DX},{DY+DH} L{DX-DW},{DY} z" fill="white" '
    f'stroke="{NAVY}" stroke-width="1.6"/>')
text(DX, DY - 6, "Band test", size=15, anchor="middle")
rich(DX, DY + 14, [("τ", ""), ("low", "sub"), (" ≤ ", ""), ("S", "it"), (" ≤ τ", ""), ("high", "sub"), (" ?", "")],
     size=13.5, anchor="middle")
path(f"M1157,244 V{DY-DH-1}", color=NAVY, head="hn", width=1.6)
# fusion score S into the diamond
path(f"M{FX+FW},300 H{DX-DW-1}", color=GREY, width=2.2)
rich(800, 292, [("fusion score ", ""), ("S", "it")], size=15)
# no -> decide on S ; yes -> decide on R
box(1196, 360, 86, 40, stroke=NAVY, sw=1.4)
rich(1239, 386, [("decide on ", ""), ("S", "it")], size=14.5, anchor="middle")
path(f"M{DX+DW},{DY} H1239 V358", color=NAVY, head="hn", width=1.6)
text(DX + DW + 10, DY - 7, "no", size=14)
box(1076, 360, 110, 48, stroke=NAVY, sw=1.4)
rich(1131, 381, [("decide on ", ""), ("R", "it")], size=14.5, anchor="middle")
text(1131, 398, "(risk score)", size=12.5, anchor="middle")
path(f"M{DX},{DY+DH} V352 H1131 V358", color=NAVY, head="hn", width=1.6)
text(DX - 34, DY + DH + 4, "yes", size=14)
# escalation gate
box(1076, 440, 206, 92, stroke=NAVY, fill="#F4F7FB", sw=1.8)
text(1179, 462, "Escalation gate", size=16, anchor="middle", weight="bold", fill=NAVY)
text(1179, 482, "raise to attack if state ≥", size=13.5, anchor="middle")
rich(1179, 499, [("ELEVATED and ", ""), ("S", "it"), ("eff", "sub"), (" > τ", ""), ("low", "sub")],
     size=13.5, anchor="middle")
text(1179, 518, "(otherwise the decision stands)", size=12, anchor="middle", style="italic", fill="#555")
path("M1131,408 V438", color=NAVY, head="hn", width=1.6)
path("M1239,400 V438", color=NAVY, head="hn", width=1.6)
# stage 4 trust tracker
box(792, 352, 250, 100, stroke=NAVY, sw=1.6)
badge(792, 352, 4)
text(917, 376, "Trust tracker", size=17, anchor="middle")
text(917, 396, "running agreement with the verdict;", size=13.5, anchor="middle")
text(917, 412, "damps each modality's probability", size=13.5, anchor="middle")
tag(917, 422, ["settles at 0.89 / 0.84 / 0.78 (CV)"])
path("M1042,384 H1074", color=NAVY, head="hn", width=1.6)
rich(612, 392, [("p", "it"), ("net", "sub"), (", ", ""), ("p", "it"), ("iot", "sub"), (", ", ""),
                ("p", "it"), ("log", "sub")], size=14.5)
path("M600,400 H790")
# stage 8 energy state machine
box(792, 470, 250, 84, stroke=NAVY, sw=1.6)
badge(792, 470, 8)
text(917, 495, "Energy state machine", size=17, anchor="middle")
text(917, 516, "SAFE · SUSPICIOUS ·", size=13.5, anchor="middle")
text(917, 533, "ELEVATED · CRITICAL", size=13.5, anchor="middle")
path("M1042,500 H1074", color=NAVY, head="hn", width=1.6)
text(1058, 492, "state", size=12, anchor="middle", fill="#555")
# stage 5 policy engine: limits set from the site's own normal records
box(1076, 572, 206, 128, stroke=NAVY, sw=1.6)
badge(1076, 572, 5)
text(1179, 596, "Policy engine", size=17, anchor="middle")
text(1179, 615, "limits from the site's normal", size=13, anchor="middle")
text(1179, 631, "records, per field and device", size=13, anchor="middle")
text(1179, 650, "hard (99.9th pct): forced alert", size=13, anchor="middle")
text(1179, 666, "soft (99th, 99.9th): pressure", size=13, anchor="middle")
tag(1179, 674, ["707 forced alerts on the stream"])
path("M1282,620 H1322 V566", color=NAVY, head="hn", width=1.6)
lines(1326, 640, ["forced", "alert,", "skips 6–8"], size=12, fill="#555", lh=14)
path("M1076,684 H1004 V556", color=NAVY, head="hn", width=1.6)
text(1018, 620, "policy pressure", size=12.5, anchor="middle", rot=-90, fill="#555")

# ── E: verdict ───────────────────────────────────────────────────────
box(1310, 430, 132, 134, stroke=NAVY, sw=2)
text(1376, 458, "Verdict:", size=17, anchor="middle", weight="bold", fill=NAVY)
text(1376, 479, "attack / normal", size=16, anchor="middle", weight="bold", fill=NAVY)
lines(1376, 503, ["reason, severity,", "recommended", "action"], size=13, anchor="middle", lh=15)
path("M1282,486 H1308", color=NAVY, head="hn", width=2)

# online learner: off the evaluation path
box(612, 590, 146, 104, stroke=MUTE, fill="#F7F7F7", sw=1.4, dash="6,4")
text(685, 614, "Online learner", size=15.5, anchor="middle", fill="#777")
lines(685, 636, ["refits a copy of the", "fusion model from", "pseudo-labels; never", "scores the stream"],
      size=12, anchor="middle", lh=14, fill="#777")

# key to the line styles
text(24, 652, "Key", size=13, weight="bold")
path("M24,674 H62", color=NAVY, head=None, width=2)
lines(68, 679, ["acts in every", "reported result"], size=12, lh=14)
path("M24,718 H62", color=MUTE, dash="5,4", head=None, width=2)
lines(68, 723, ["present, off in", "the evaluation"], size=12, lh=14)

# ── memories lane ────────────────────────────────────────────────────
box(562, 722, 726, 150, stroke="#9A9A9A", fill="white", sw=1.2, r=8)
text(576, 862, "Memories (alongside the pipeline)", size=16, weight="bold")
box(582, 740, 236, 92, stroke=NAVY, sw=1.6)
badge(582, 740, 9)
text(700, 766, "Behavioural memory", size=17, anchor="middle")
text(700, 788, "campaign recognition", size=13.5, anchor="middle")
text(700, 805, "across sequences", size=13.5, anchor="middle")
box(836, 740, 196, 92, stroke=INK, sw=1.4)
text(934, 776, "Statistical memory", size=17, anchor="middle")
text(934, 797, "(clusters)", size=14, anchor="middle")
box(1060, 740, 210, 92, stroke=MUTE, fill="#F7F7F7", sw=1.4, dash="6,4")
text(1165, 774, "Experience cache", size=17, anchor="middle", fill="#777")
text(1165, 796, "disabled in evaluation", size=13.5, anchor="middle", style="italic", fill="#777")
# memories feed the state machine (from below)
path("M806,740 V556", color=GREY, width=1.8)
text(820, 648, "campaign risk", size=12.5, anchor="middle", rot=-90, fill="#555")
path("M856,740 V556", color=GREY, width=1.8)
text(870, 648, "cluster risk", size=12.5, anchor="middle", rot=-90, fill="#555")
# verdicts are stored in the memories
path("M1432,564 V800 H1290", color=GREY, width=1.8)
text(1426, 790, "store", size=13, anchor="end", fill="#555")
# cache bypass: layer probabilities -> cache -> verdict (disabled)
path(f"M{EVX+23},{EVY+EVH} V590 H550 V894 H1165 V834", color=MUTE, dash="5,4", head="hm", width=1.6)
text(820, 888, "cache lookup on the layer outputs; a hit bypasses stages 3–8", size=12, fill="#666",
     style="italic")
path("M1270,770 H1398 V566", color=MUTE, dash="5,4", head="hm", width=1.6)

add("</svg>")
svg = "\n".join(out)
here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "CATF-IDS_Fig_nine_stages")
open(stem + ".svg", "w", encoding="utf-8").write(svg)
try:
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=stem + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=stem + ".png", output_width=2910)
    print(f"wrote {stem}.svg / .pdf / .png")
except ImportError:
    print(f"wrote {stem}.svg (pip install cairosvg for PDF/PNG)")
