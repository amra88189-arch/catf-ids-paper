#!/usr/bin/env python3
"""Draw the §4.1 figure: the TON_IoT testbed (redrawn from Alsaedi et al., IEEE Access 2020,
Fig. 1, CC BY 4.0) with the three streams CATF-IDS uses marked where they were collected.

    python make_fig_testbed.py        # needs: pip install cairosvg  (for PDF/PNG)

Every label is traceable:
  testbed layers, machines, Security Onion "log network data from all the active systems",
  Node-RED data logger and simulated sensors, Kali VMs, NSX-VMware/vCloud  -> Alsaedi et al. [9]
  atop tracing on the orchestrated and middleware servers                   -> Moustafa et al. (Linux)
  feature counts, 7 device types, Windows label-only, 300 s join            -> this paper, §4-§5
Addresses are left out on purpose: the sources give different attacker ranges.
"""
import os

W, H = 1290, 680
FONT = "Times New Roman, Liberation Serif, Times, serif"
NAVY, GREY, INK, FOG, MUTE = "#1F4D78", "#7F7F7F", "#222222", "#E3EBF4", "#555555"
Y_CF, Y_FE = 150, 530          # cloud/fog and fog/edge boundaries
X_RC = 985                     # right-hand column (our use of the data) starts here

out = []
add = out.append


def text(x, y, s, size=17, anchor="start", weight="normal", style="normal", fill=INK):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" '
        f'fill="{fill}">{s}</text>')


def lines(x, y, rows, size=17, lh=None, **kw):
    lh = lh or size * 1.22
    for i, r in enumerate(rows):
        text(x, y + i * lh, r, size=size, **kw)


def box(x, y, w, h, stroke=INK, fill="white", sw=1.4, r=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(x1, y1, x2, y2, color=GREY, width=2.2, head="hg", both=False, dash=None):
    s = f' marker-start="url(#{head}s)"' if both else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}"{d} marker-end="url(#{head})"{s}/>')


def path(d, color=GREY, width=2.0, head="hg", dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    hd = f' marker-end="url(#{head})"' if head else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{ds}{hd}/>')


def badge(x, y, n, fill=NAVY):
    add(f'<circle cx="{x}" cy="{y}" r="13" fill="{fill}"/>')
    text(x, y + 6, str(n), size=17, anchor="middle", weight="bold", fill="white")


# ── canvas, markers ──────────────────────────────────────────────────
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>'
    f'<marker id="hg" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker>'
    f'<marker id="hgs" markerWidth="10" markerHeight="10" refX="2" refY="5" orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker>'
    f'<marker id="hn" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L12,6 L0,12 z" fill="{NAVY}"/></marker>'
    f'<marker id="hk" markerWidth="16" markerHeight="16" refX="13" refY="8" orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L16,8 L0,16 z" fill="{INK}"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ── bands (testbed side only) ────────────────────────────────────────
add(f'<rect x="0" y="{Y_CF}" width="{X_RC}" height="{Y_FE - Y_CF}" fill="{FOG}"/>')
for y in (Y_CF, Y_FE):
    add(f'<line x1="0" y1="{y}" x2="{X_RC}" y2="{y}" stroke="#9A9A9A" stroke-width="1"/>')
add(f'<line x1="0" y1="{H - 1}" x2="{X_RC}" y2="{H - 1}" stroke="#9A9A9A" stroke-width="1"/>')
add(f'<line x1="{X_RC}" y1="0" x2="{X_RC}" y2="{H}" stroke="#9A9A9A" stroke-width="1"/>')
text(X_RC - 10, 20, "TON_IoT testbed, redrawn", size=15, anchor="end", style="italic", fill=MUTE)

# tier labels (left)
text(22, 72, "Cloud", size=26)
text(22, 98, "online services", size=16, fill=MUTE)
text(22, 200, "Fog", size=26)
lines(22, 226, ["virtual", "machines"], size=16, fill=MUTE)
text(22, 582, "Edge", size=26)
lines(22, 608, ["physical", "devices"], size=16, fill=MUTE)

# ── cloud ────────────────────────────────────────────────────────────
box(185, 40, 215, 70)
text(292, 81, "Hive-MQTT broker", anchor="middle", size=18)
box(425, 40, 265, 70)
text(557, 81, "Vulnerable PHP website", anchor="middle", size=18)
box(710, 40, 255, 70)
text(837, 70, "Cloud services", anchor="middle", size=18)
text(837, 94, "(e.g., Azure IoT Hub, AWS Lambda)", anchor="middle", size=15, fill=MUTE)

# ── fog: middleware Node-RED server (telemetry and one Linux host) ───
box(185, 185, 215, 310, stroke=NAVY, sw=2.2)
text(292, 214, "Middleware", anchor="middle", size=18)
text(292, 237, "Node-RED server", anchor="middle", size=18)
text(292, 260, "Ubuntu 18.04", anchor="middle", size=16, fill=MUTE)
add(f'<line x1="200" y1="278" x2="385" y2="278" stroke="#B0B0B0" stroke-width="1"/>')
lines(202, 312, ["simulates IoT/IIoT", "sensors"], size=16, lh=20)
lines(202, 375, ["telemetry", "data logger"], size=16, lh=20)
badge(372, 380, 2)
lines(202, 440, ["host tracing", "(atop)"], size=16, lh=20)
badge(372, 445, 3)

# Node-RED <-> Hive-MQTT broker
arrow(292, 112, 292, 183, both=True)
text(302, 172, "publish / subscribe", size=15, style="italic", fill=MUTE)

# ── fog: services and hosts attacked ────────────────────────────────
box(425, 185, 105, 44)
text(477, 213, "DVWA", anchor="middle")
box(545, 185, 170, 44)
text(630, 213, "Metasploitable3", anchor="middle")
box(425, 242, 290, 44)
text(570, 270, "OWASP Security Shepherd", anchor="middle")
box(425, 299, 140, 44, stroke=GREY, dash="5,4")
text(495, 327, "Windows 7", anchor="middle", fill=GREY)
box(575, 299, 140, 44, stroke=GREY, dash="5,4")
text(645, 327, "Windows 10", anchor="middle", fill=GREY)
box(425, 358, 290, 62, stroke=NAVY, sw=2.2)
text(440, 383, "Orchestrated server", size=18)
text(440, 407, "Ubuntu 14.04 · host tracing (atop)", size=16, fill=MUTE)
badge(690, 378, 3)
box(425, 433, 290, 62, stroke=NAVY, sw=2.2)
text(440, 458, "Security Onion", size=18)
text(440, 482, "logs network data from all active systems", size=16, fill=MUTE)
badge(690, 453, 1)

# ── fog: offensive systems ───────────────────────────────────────────
box(800, 185, 165, 102, stroke=INK, sw=2.2, fill="#F4F4F4")
text(882, 213, "Offensive systems", anchor="middle", size=18)
text(882, 237, "10 Kali Linux VMs", anchor="middle", size=16)
text(882, 259, "attack scripts", anchor="middle", size=16, fill=MUTE)
arrow(800, 236, 727, 236, color=INK, width=3, head="hk")
text(763, 226, "attacks", anchor="middle", size=15, style="italic")
lines(882, 448, ["virtualised with", "NSX-VMware and", "vCloud (SDN, NFV)"],
      size=15, lh=19, anchor="middle", style="italic", fill=MUTE)

# ── edge ─────────────────────────────────────────────────────────────
box(185, 565, 215, 70)
text(292, 594, "IoT/IIoT sensors", anchor="middle", size=18)
text(292, 618, "e.g., thermostat, weather", anchor="middle", size=15, fill=MUTE)
arrow(292, 563, 292, 498)
text(302, 552, "MQTT", size=15, style="italic", fill=MUTE)
box(425, 565, 190, 70)
text(520, 594, "NSX-VMware", anchor="middle", size=18)
text(520, 618, "host server (runs the VMs)", anchor="middle", size=15, fill=MUTE)
box(630, 565, 160, 70)
text(710, 594, "Router and", anchor="middle", size=18)
text(710, 617, "switches", anchor="middle", size=18)
box(805, 565, 160, 70)
text(885, 594, "Smartphones,", anchor="middle", size=18)
text(885, 617, "smart TV", anchor="middle", size=18)

# ── right column: what CATF-IDS takes from the testbed ───────────────
text(1128, 44, "Used by CATF-IDS", anchor="middle", size=20, weight="bold", fill=NAVY)
RX, RW = 1003, 247
for (y, n, t1, t2) in ((68, 1, "Network flows", "32 features"),
                       (140, 2, "Device telemetry", "7 device types, 7 features"),
                       (212, 3, "Host activity (Linux)", "10 features")):
    box(RX, y, RW, 60, stroke=NAVY, sw=2.2)
    badge(RX + 22, y + 30, n)
    text(RX + 44, y + 25, t1, size=18)
    text(RX + 44, y + 48, t2, size=16, fill=MUTE)
box(RX, 290, RW, 78, stroke=GREY, dash="5,4")
text(RX + 14, 315, "Windows host data", size=18, fill=GREY)
lines(RX + 14, 338, ["label enters the vote;", "features not used"], size=16, lh=19, fill=GREY)

# bus from each stream into the join
BX = 1268
for yy in (98, 170, 242):
    add(f'<line x1="{RX + RW}" y1="{yy}" x2="{BX}" y2="{yy}" stroke="{NAVY}" stroke-width="2"/>')
add(f'<line x1="{RX + RW}" y1="329" x2="{BX}" y2="329" stroke="{GREY}" stroke-width="2" stroke-dasharray="5,4"/>')
path(f"M{BX},98 V430 H{RX + RW + 3}", color=NAVY, width=2, head="hn")

box(RX, 400, RW, 60, stroke=INK)
text(RX + RW / 2, 425, "Timestamp join", anchor="middle", size=18)
text(RX + RW / 2, 448, "300 s tolerance", anchor="middle", size=16, fill=MUTE)
arrow(RX + RW / 2, 462, RX + RW / 2, 492, color=NAVY, head="hn")
box(RX, 495, RW, 56, stroke=NAVY, fill=NAVY)
text(RX + RW / 2, 530, "CATF-IDS", anchor="middle", size=21, weight="bold", fill="white")
lines(RX + RW / 2, 590, ["each stream carries", "its own label"],
      size=16, lh=20, anchor="middle", style="italic", fill=MUTE)

add("</svg>")
svg = "\n".join(out)

here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "CATF-IDS_Fig_testbed")
open(stem + ".svg", "w", encoding="utf-8").write(svg)
try:
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=stem + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=stem + ".png", output_width=2580)
    print(f"wrote {stem}.svg / .pdf / .png")
except ImportError:
    print(f"wrote {stem}.svg (pip install cairosvg for PDF/PNG)")
