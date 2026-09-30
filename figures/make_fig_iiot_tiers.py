#!/usr/bin/env python3
"""Draw the §3.1 figure (three IIoT tiers and where CATF-IDS sits) as SVG, PDF, PNG.

    python make_fig_iiot_tiers.py        # needs: pip install cairosvg  (for PDF/PNG)

Everything is drawn from primitives (no icon sets, no logos), so the figure can be
edited as text: change a label here and re-run.

The navy "Purdue Levels" labels give the approximate Purdue correspondence of each tier
(T. J. Williams, Computers in Industry 24, 1994; Level 3.5 DMZ per NIST SP 800-82r3).
The version without them is make_fig_iiot_tiers_prePurdue.py.
"""
import os

W, H = 1290, 760
FONT = "Times New Roman, Liberation Serif, Times, serif"
NAVY, GREY, INK, FOG = "#1F4D78", "#7F7F7F", "#222222", "#E3EBF4"
Y_CF, Y_FE = 245, 540          # cloud/fog and fog/edge boundaries
X_RC = 1030                    # right-hand column starts here
# At full page width (16.4 cm) one unit is ~0.36 pt: the smallest text below
# (15.5 units) prints at ~5.6 pt, the tier labels at ~9.4 pt.

out = []
add = out.append


def text(x, y, s, size=15, anchor="start", weight="normal", style="normal", fill=INK):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" '
        f'fill="{fill}">{s}</text>')


def lines(x, y, rows, size=15, lh=None, **kw):
    lh = lh or size * 1.22
    for i, r in enumerate(rows):
        text(x, y + i * lh, r, size=size, **kw)


def arrow(x1, y1, x2, y2, color=GREY, width=2.2, dash=None, head="hg"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}"{d} marker-end="url(#{head})"/>')


def curve(path, color=GREY, width=1.8, head="hg"):
    add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'marker-end="url(#{head})"/>')


def box(x, y, w, h, stroke=INK, fill="white", sw=1.4, r=3):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')


# ── canvas, markers ──────────────────────────────────────────────────
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>'
    f'<marker id="hg" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker>'
    f'<marker id="hn" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L12,6 L0,12 z" fill="{NAVY}"/></marker>'
    f'<marker id="hb" markerWidth="16" markerHeight="16" refX="13" refY="8" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L16,8 L0,16 z" fill="{GREY}"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ── bands ────────────────────────────────────────────────────────────
add(f'<rect x="0" y="{Y_CF}" width="{W}" height="{Y_FE - Y_CF}" fill="{FOG}"/>')
for y in (Y_CF, Y_FE, H - 1):
    add(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#9A9A9A" stroke-width="1"/>')
add(f'<line x1="{X_RC}" y1="0" x2="{X_RC}" y2="{H}" stroke="#9A9A9A" stroke-width="1"/>')
text(W - 8, 18, "specified, not deployed", size=14, anchor="end", style="italic", fill="#555")

# tier labels (left)
lines(22, 84, ["Cloud / central —", "most compute,", "least locality"], size=26, lh=32)
lines(22, 300, ["Fog / gateway", "— local compute"], size=26, lh=32)
lines(22, 640, ["Edge —", "instruments"], size=26, lh=32)

# Purdue reference levels, for orientation only (Williams 1994; DMZ as in NIST SP 800-82r3).
# The correspondence to the three tiers is approximate and is stated so in the caption.
PUR = dict(size=16, lh=19, fill=NAVY)
text(22, 190, "Purdue Levels 4–5", size=19, fill=NAVY)
lines(22, 210, ["business, enterprise IT"], **PUR)
text(22, 366, "Purdue Levels 2–3", size=19, fill=NAVY)
lines(22, 386, ["supervisory control,", "site operations"], **PUR)
text(22, 712, "Purdue Levels 0–1", size=19, fill=NAVY)
lines(22, 732, ["process, basic control"], **PUR)
box(16, Y_CF - 13, 214, 26, stroke=NAVY, sw=1.2, r=4)
text(123, Y_CF + 6, "Level 3.5 · IT/OT DMZ", size=16, anchor="middle", fill=NAVY)

# what each tier can detect (right column)
RC = dict(size=18, lh=23)
lines(X_RC + 16, 70, ["Sees everything, but at a", "delay; its verdict returns", "over a link an attacker",
                      "may be affecting."], **RC)
lines(X_RC + 16, 345, ["First place where", "evidence from several", "sources can be combined",
                       "while a response is still", "timely."], **RC)
lines(X_RC + 16, 615, ["Sees one device in detail,", "nothing else — can", "enforce a local rule,",
                       "cannot correlate."], **RC)

# ── cloud: consolidation path ────────────────────────────────────────
box(430, 22, 555, 140, stroke=INK)
text(448, 51, "Consolidation path (batch, cost grows with history)", size=19)
lines(456, 80, ["•  fitting the models; setting the site's limits",
                "•  clustering the evidence log",
                "•  correlating the behavioural memory",
                "•  per-incident explanations for analysts"], size=17, lh=22)

# ── uplink: evidence up, models down; both dashed = the link ─────────
XU, XD = 590, 790
BX, BY, BW, BH = 455, 290, 560, 182
arrow(XU, BY, XU, 167, color=GREY, width=2.8, dash="8,6", head="hg")
arrow(XD, 162, XD, BY - 4, color=NAVY, width=2.8, dash="8,6", head="hn")
lines(XU - 12, 196, ["evidence log and events", "(the record of what has been seen)"],
      size=16.5, anchor="end", lh=20)
lines(XD + 12, 196, ["fitted models and limits;", "cache built centrally"], size=16, lh=20)
cx, cy = (XU + XD) / 2, 190
add(f'<path d="M{cx},{cy-17} L{cx+16},{cy+11} L{cx-16},{cy+11} z" fill="white" '
    f'stroke="{INK}" stroke-width="1.6" stroke-linejoin="round"/>')
text(cx, cy + 8, "!", size=17, anchor="middle", weight="bold")
lines(cx, cy + 31, ["uplink may be", "under attack"], size=15.5, anchor="middle", lh=17)

# ── fog: decision path ───────────────────────────────────────────────
box(BX, BY, BW, BH, stroke=NAVY, fill="white", sw=2.2)
text(BX + 16, BY + 29, "CATF-IDS decision path", size=22, weight="bold", fill=NAVY)
text(BX + 16, BY + 51, "(per event, bounded cost)", size=17, fill=NAVY)
cx0, cy0 = BX + 372, BY + 13
add(f'<path d="M{cx0},{cy0+6} v26 a15,6 0 0 0 30,0 v-26" fill="white" stroke="{INK}" stroke-width="1.5"/>')
add(f'<ellipse cx="{cx0+15}" cy="{cy0+6}" rx="15" ry="6" fill="white" stroke="{INK}" stroke-width="1.5"/>')
lines(cx0 + 38, cy0 + 17, ["experience cache", "(read locally)"], size=15.5, lh=18)
chain = [("feature", "extraction"), ("three forest", "evaluations"), ("fusion of", "5 fold models"),
         ("site limits,", "thresholds,", "state update"), ("verdict",)]
widths = [92, 100, 104, 104, 70]
x, y0, hh, gap = BX + 16, BY + 68, 60, 15
for i, (lab, w) in enumerate(zip(chain, widths)):
    box(x, y0, w, hh, stroke=NAVY, fill="#F4F7FB", sw=1.3, r=2)
    if len(lab) == 1:
        text(x + w / 2, y0 + hh / 2 + 5, lab[0], size=15.5, anchor="middle")
    else:
        lines(x + w / 2, y0 + (26 if len(lab) == 2 else 18), lab, size=15.5, anchor="middle", lh=17)
    if i < len(chain) - 1:
        arrow(x + w + 1, y0 + hh / 2, x + w + gap - 1, y0 + hh / 2, color=NAVY, width=1.5, head="hn")
    x += w + gap
lines(BX + 16, BY + 150, ["No dependency on the link: models are resident locally;",
                          "the cache degrades to a miss, not a failure."], size=15, style="italic", lh=17)

# host (e.g., Linux): generic server icon
hx, hy = 298, 256
box(hx, hy, 28, 42, stroke=INK, fill="white", sw=1.5, r=2)
for k in range(3):
    add(f'<line x1="{hx+4}" y1="{hy+10+k*10}" x2="{hx+18}" y2="{hy+10+k*10}" stroke="{INK}" stroke-width="1.3"/>')
    add(f'<circle cx="{hx+22}" cy="{hy+10+k*10}" r="1.8" fill="{INK}"/>')
text(hx + 36, hy + 16, "host (e.g., Linux)", size=16)
text(hx + 36, hy + 36, "host activity", size=16)
curve(f"M{hx+14},{hy+43} C{hx+14},{hy+80} {BX-70},{BY+45} {BX-2},{BY+45}", color=GREY, width=2.2)

# convergence point: network traffic from the equipment
nx, ny = 290, 425
add(f'<circle cx="{nx}" cy="{ny}" r="11" fill="white" stroke="{INK}" stroke-width="1.8"/>')
lines(nx - 17, 386, ["traffic from", "different", "equipment", "converges", "here"],
      size=15.5, anchor="end", lh=17)
arrow(nx + 12, ny, BX - 2, ny, color=GREY, width=2.4, head="hg")
lines(nx + 22, ny + 24, ["network flows", "(network modality", "observed here)"], size=15.5, lh=17)

# device telemetry: edge -> decision path
TX = 735
arrow(TX, 582, TX, BY + BH + 5, color=GREY, width=6, head="hb")
text(TX + 14, Y_FE - 10, "device telemetry", size=16)

# ── edge: instruments ────────────────────────────────────────────────
text(TX, 604, "constrained power and memory; runs its control loop", size=17, anchor="middle")
ICON_Y, LAB_Y = 660, 716
icons = [(255, "sensors"), (345, "actuators"), (440, "controllers|driving them"),
         (545, "motion|light"), (630, "thermostat"), (712, "fridge"), (792, "garage|door"),
         (877, "Modbus|register"), (962, "weather|station")]
S = f'fill="none" stroke="{INK}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'


def icon(kind, x, y):
    if kind == "sensors":        # antenna with radio waves
        add(f'<line x1="{x}" y1="{y-2}" x2="{x}" y2="{y+22}" {S}/>'
            f'<line x1="{x-8}" y1="{y+22}" x2="{x+8}" y2="{y+22}" {S}/>'
            f'<circle cx="{x}" cy="{y-5}" r="3" {S}/>')
        for r in (9, 15):
            add(f'<path d="M{x-r*0.7},{y-5-r*0.7} A{r},{r} 0 0 0 {x-r*0.7},{y-5+r*0.7}" {S}/>'
                f'<path d="M{x+r*0.7},{y-5-r*0.7} A{r},{r} 0 0 1 {x+r*0.7},{y-5+r*0.7}" {S}/>')
    elif kind == "actuators":    # motor: body, shaft, base
        add(f'<rect x="{x-14}" y="{y-10}" width="24" height="22" rx="3" {S}/>'
            f'<line x1="{x+10}" y1="{y+1}" x2="{x+18}" y2="{y+1}" {S}/>'
            f'<line x1="{x-16}" y1="{y+18}" x2="{x+12}" y2="{y+18}" {S}/>'
            f'<line x1="{x-9}" y1="{y+12}" x2="{x-9}" y2="{y+18}" {S}/>'
            f'<line x1="{x+5}" y1="{y+12}" x2="{x+5}" y2="{y+18}" {S}/>'
            f'<path d="M{x-4},{y-5} L{x-7},{y+2} L{x-2},{y+2} L{x-5},{y+8}" {S}/>')
    elif kind == "controllers":  # PLC rack
        add(f'<rect x="{x-24}" y="{y-10}" width="48" height="28" rx="2" {S}/>')
        for k in range(1, 4):
            add(f'<line x1="{x-24+k*12}" y1="{y-10}" x2="{x-24+k*12}" y2="{y+18}" {S}/>')
        for k in range(4):
            add(f'<rect x="{x-21+k*12}" y="{y-5}" width="6" height="4" {S}/>')
    elif kind == "motion":       # light bulb with rays
        add(f'<path d="M{x-6},{y+10} C{x-6},{y+3} {x-12},{y} {x-12},{y-7} A12,12 0 1 1 {x+12},{y-7} '
            f'C{x+12},{y} {x+6},{y+3} {x+6},{y+10} z" {S}/>'
            f'<line x1="{x-5}" y1="{y+14}" x2="{x+5}" y2="{y+14}" {S}/>'
            f'<line x1="{x-3}" y1="{y+18}" x2="{x+3}" y2="{y+18}" {S}/>')
        for dx, dy in ((-19, -14), (19, -14), (-16, 2), (16, 2), (0, -25)):
            add(f'<line x1="{x+dx*0.85}" y1="{y-7+dy*0.85}" x2="{x+dx}" y2="{y-7+dy}" {S}/>')
    elif kind == "thermostat":   # thermometer
        add(f'<path d="M{x-4},{y+8} V{y-14} A4,4 0 0 1 {x+4},{y-14} V{y+8}" {S}/>'
            f'<circle cx="{x}" cy="{y+13}" r="7" {S}/>'
            f'<line x1="{x}" y1="{y-6}" x2="{x}" y2="{y+10}" stroke="{INK}" stroke-width="2.4"/>'
            f'<line x1="{x+6}" y1="{y-10}" x2="{x+10}" y2="{y-10}" {S}/>'
            f'<line x1="{x+6}" y1="{y-3}" x2="{x+10}" y2="{y-3}" {S}/>')
    elif kind == "fridge":
        add(f'<rect x="{x-11}" y="{y-16}" width="22" height="38" rx="2" {S}/>'
            f'<line x1="{x-11}" y1="{y-3}" x2="{x+11}" y2="{y-3}" {S}/>'
            f'<line x1="{x+6}" y1="{y-12}" x2="{x+6}" y2="{y-7}" {S}/>'
            f'<line x1="{x+6}" y1="{y+2}" x2="{x+6}" y2="{y+10}" {S}/>')
    elif kind == "garage":
        add(f'<path d="M{x-18},{y+22} V{y-8} L{x},{y-18} L{x+18},{y-8} V{y+22}" {S}/>')
        for k in range(5):
            add(f'<line x1="{x-13}" y1="{y-2+k*5}" x2="{x+13}" y2="{y-2+k*5}" {S}/>')
    elif kind == "Modbus":       # chip with pins
        add(f'<rect x="{x-12}" y="{y-10}" width="24" height="24" rx="2" {S}/>'
            f'<rect x="{x-5}" y="{y-3}" width="10" height="10" {S}/>')
        for k in (-6, 2, 10):
            add(f'<line x1="{x-17}" y1="{y+k-2}" x2="{x-12}" y2="{y+k-2}" {S}/>'
                f'<line x1="{x+12}" y1="{y+k-2}" x2="{x+17}" y2="{y+k-2}" {S}/>'
                f'<line x1="{x+k-2}" y1="{y-15}" x2="{x+k-2}" y2="{y-10}" {S}/>'
                f'<line x1="{x+k-2}" y1="{y+14}" x2="{x+k-2}" y2="{y+19}" {S}/>')
    elif kind == "weather":      # mast with anemometer and vane
        add(f'<line x1="{x}" y1="{y-12}" x2="{x}" y2="{y+22}" {S}/>'
            f'<line x1="{x-8}" y1="{y+22}" x2="{x+8}" y2="{y+22}" {S}/>'
            f'<line x1="{x-14}" y1="{y-12}" x2="{x+14}" y2="{y-12}" {S}/>'
            f'<path d="M{x-18},{y-12} a4,4 0 0 0 8,0" {S}/>'
            f'<path d="M{x+10},{y-12} a4,4 0 0 0 8,0" {S}/>'
            f'<path d="M{x},{y+2} L{x+14},{y+2} L{x+10},{y-2} M{x+14},{y+2} L{x+10},{y+6}" {S}/>')


for x, lab in icons:
    add(f'<g transform="translate({x},{ICON_Y}) scale(1.18) translate({-x},{-ICON_Y})">')
    icon(lab.split("|")[0].split()[0], x, ICON_Y)
    add('</g>')
    lines(x, LAB_Y, lab.split("|"), size=16, anchor="middle", lh=18)

# equipment traffic rising into the convergence point, routed below the flows label
for sx, c1, c2, end in ((255, (255, 545), (272, 505), (285, ny + 14)),
                        (345, (345, 560), (300, 530), (290, ny + 14)),
                        (440, (440, 556), (306, 548), (295, ny + 13))):
    curve(f"M{sx},{ICON_Y-32} C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {end[0]},{end[1]}", color=GREY, width=2)

add("</svg>")
svg = "\n".join(out)

here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "CATF-IDS_Fig_IIoT_tiers")
open(stem + ".svg", "w", encoding="utf-8").write(svg)
try:
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=stem + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=stem + ".png", output_width=2580)
    print(f"wrote {stem}.svg / .pdf / .png")
except ImportError:
    print(f"wrote {stem}.svg (pip install cairosvg for PDF/PNG)")
