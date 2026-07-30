"""
Generate the prompting strategy diagram (Fig D1).

Usage (run from inside Fig_D1_prompting_strategy/):
    python scripts/plotting.py
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── pyloric style ─────────────────────────────────────────────────────────────
plt.style.use("../pyloric.mplstyle")

# ── figure size ───────────────────────────────────────────────────────────────
FIG_CM_W = 9.0
FIG_CM_H = 6.0

CM = 1 / 2.54
FIG_IN_W = FIG_CM_W * CM
FIG_IN_H = FIG_CM_H * CM
PT_PER_IN = 72.0
FIG_PT_H = FIG_IN_H * PT_PER_IN
FIG_PT_W = FIG_IN_W * PT_PER_IN  # figure size in points = coordinate range

# ── colours ───────────────────────────────────────────────────────────────────
C_SYS = "#CDDAF5"  # blue   – system
C_USER = "#FFF3CD"  # amber  – user
C_ASST = "#D4EDDA"  # green  – assistant
C_DEMO = "#FAF0FF"  # lilac  – few-shot demo container
C_EDGE = "#495057"
C_ARR = "#343A40"

# Role-specific frame colours (match generate_llm_history_latex.py)
C_SYS_F = "#7799e4"
C_USER_F = "#ffdb66"
C_ASST_F = "#8ccf9c"
C_OUT = "#FAC898"  # orange – output
C_OUT_F = "#f69337"

# ── font sizes (use pyloric default = 9 pt everywhere) ────────────────────────
FS = 9  # base font size (matches pyloric font.size)
LH = FS * 1.2  # line height with 1.2× leading

# ── geometry (all in pt) ──────────────────────────────────────────────────────
PAD_Y = 4.0  # top/bottom padding inside each coloured section
PAD_BOX = 5.0  # padding between outer box edge and sections
GAP = 4.0  # gap between consecutive sections inside a box
DGAP = 4.0  # gap between demo blocks inside container
DEMO_PAD = 3.0  # inner padding of demo container (top & bottom)

HEADER_H = PAD_Y + LH  # height of the filled role-label strip at box top


# Section heights (role header + N field lines + bottom padding)
def sh(n_lines):
    return HEADER_H + n_lines * LH + PAD_Y


SH_SYS = sh(2)  # system: 2 field lines
SH_USER = sh(3)  # user:   3 field lines
SH_ASST = sh(2)  # asst:   2 field lines
SH_OUT3 = sh(3)  # output in right column: 3 field lines
SH_DU = sh(1)  # demo user:   1 line
SH_DA = sh(3)  # demo asst:   3 lines

N_DEMOS = 3
demo_block_h = SH_DU + GAP + SH_DA
demo_header_h = LH + 6
demo_container_h = (
    demo_header_h + DEMO_PAD + N_DEMOS * demo_block_h + (N_DEMOS - 1) * DGAP + DEMO_PAD
)

H_BOX12 = 2 * PAD_BOX + SH_SYS + GAP + SH_USER + GAP + SH_ASST
H_BOX3 = 2 * PAD_BOX + SH_SYS + GAP + demo_container_h + GAP + SH_USER + GAP + SH_OUT3

# Box widths
BOX_W = 180.0  # pt  (≈ 3.9 cm; tight to text)
ARROW_W = 70.0  # pt horizontal gap between boxes (arrow space)
LOOP_H = 14.0  # pt space above tallest box for the loop arrow

# Canvas size: figure dimensions in pt
W = FIG_PT_W
H = FIG_PT_H

# Box positions (bottom-left corners)
total_content_w = 2 * BOX_W + 1 * ARROW_W
margin_x = (W - total_content_w) / 2

max_h = max(H_BOX12, H_BOX3)
margin_y = (H - max_h - LOOP_H) / 2  # leave LOOP_H above for return arrow

x2 = margin_x
x3 = x2 + BOX_W + ARROW_W


# Vertically centre each box (shorter boxes centred against tallest)
def y_bot(box_h):
    return margin_y + (max_h - box_h) / 2


y3b = y_bot(H_BOX3)  # tallest box, sets the top alignment
y2b = y3b + H_BOX3 - H_BOX12  # top-aligned with box 3

# ── Drawing helpers (data coords = points) ────────────────────────────────────


def rrect(ax, x, y, w, h, fc, ec="#ADB5BD", lw=0.6, r=1.5, z=2):
    """Rounded rectangle, bottom-left = (x, y), dimensions (w, h)."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=z,
            clip_on=False,
        )
    )


def msg_section(ax, x, y, w, h, color, role, lines, ec="#ADB5BD"):
    """Coloured message block with a filled role-label header strip.
    (x,y)=bottom-left. Header (frame colour) at top, field lines below."""
    PX = 4.0  # left text indent (pt)
    # 1. Body background (full box, no border)
    rrect(ax, x, y, w, h, color, ec="none", lw=0, r=1.2, z=3)
    # 2. Header strip (frame colour fill, rounded top)
    rrect(ax, x, y + h - HEADER_H, w, HEADER_H, ec, ec="none", lw=0, r=1.2, z=4)
    # 3. Outer rounded border on top of everything
    rrect(ax, x, y, w, h, "none", ec=ec, lw=0.7, r=1.2, z=6)
    # Role text centred vertically in header, white
    ax.text(
        x + PX,
        y + h - HEADER_H / 2,
        role,
        ha="left",
        va="center",
        fontsize=FS - 1,
        fontweight="bold",
        fontstyle="italic",
        color="white",
        zorder=7,
        clip_on=False,
    )
    # Field lines in body below header
    body_top = y + h - HEADER_H - 2
    for i, line in enumerate(lines):
        ax.text(
            x + PX,
            body_top - i * LH,
            line,
            ha="left",
            va="top",
            fontsize=FS,
            color="#212529",
            zorder=5,
            clip_on=False,
        )


def outer_box(ax, x, y, w, h, title):
    rrect(ax, x, y, w, h, "white", ec=C_EDGE, lw=1.0, r=2.0, z=1)
    ax.text(
        x + w / 2,
        y + h + 3,
        title,
        ha="center",
        va="bottom",
        fontsize=FS,
        fontweight="bold",
        color="#212529",
        clip_on=False,
    )


_ROLE_EC = {
    "System": C_SYS_F,
    "User": C_USER_F,
    "Assistant": C_ASST_F,
    "Output": C_OUT_F,
}


def stack_sections(ax, bx, by_bot, bw, specs):
    """Stack sections bottom-up inside a box.
    specs: list of (color, role, lines, height) from bottom to top."""
    y = by_bot + PAD_BOX
    sx, sw = bx + PAD_BOX, bw - 2 * PAD_BOX
    for color, role, lines, h in specs:
        msg_section(
            ax, sx, y, sw, h, color, role, lines, ec=_ROLE_EC.get(role, "#ADB5BD")
        )
        y += h + GAP


# ── Create figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_IN_W, FIG_IN_H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# BOX 2 – feedback
# ─────────────────────────────────────────────────────────────────────────────
outer_box(ax, x2, y2b, BOX_W, H_BOX12, "Feedback")
stack_sections(
    ax,
    x2,
    y2b,
    BOX_W,
    [
        (C_OUT, "Output", ["{feedback}"], SH_ASST),
        (
            C_USER,
            "User",
            ["{system_description}", "{simulator_code}", "{performance_metrics}"],
            SH_USER,
        ),
        (C_SYS, "System", ["DiagnoseAndImprove"], SH_SYS),
    ],
)

# ── Arrow 2 → 3 ───────────────────────────────────────────────────────────────
mid_y2 = y2b + H_BOX12 / 2
ax.annotate(
    "",
    xy=(x3, mid_y2),
    xytext=(x2 + BOX_W, mid_y2),
    arrowprops=dict(arrowstyle="->", color=C_ARR, lw=1.0),
    annotation_clip=False,
)
ax.text(
    (x2 + BOX_W + x3) / 2,
    mid_y2 + 3,
    "next\niteration",
    ha="center",
    va="bottom",
    fontsize=FS,
    color=C_ARR,
    style="italic",
    clip_on=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# BOX 3 – N-shot code generation
# ─────────────────────────────────────────────────────────────────────────────
outer_box(ax, x3, y3b, BOX_W, H_BOX3, f"N={N_DEMOS}-shot code gen.")

sx3 = x3 + PAD_BOX
sw3 = BOX_W - 2 * PAD_BOX
y = y3b + PAD_BOX

# Bottom section: output (final turn)
msg_section(
    ax,
    sx3,
    y,
    sw3,
    SH_OUT3,
    C_OUT,
    "Output",
    ["{scm_definition}", "{simulator_code}", "{feedback}"],
    ec=C_OUT_F,
)
y += SH_OUT3 + GAP

# User (final turn)
msg_section(
    ax,
    sx3,
    y,
    sw3,
    SH_USER,
    C_USER,
    "User",
    ["{signature_description}", "{task_description}", "{base_simulator}"],
    ec=C_USER_F,
)
y += SH_USER + GAP

# Demo container
rrect(ax, sx3, y, sw3, demo_container_h, C_DEMO, ec="#9C6EAF", lw=0.7, r=1.5, z=2)
ax.text(
    sx3 + sw3 / 2,
    y + demo_container_h - 5,
    "Few-shot demos",
    ha="center",
    va="top",
    fontsize=FS,
    fontweight="bold",
    color="#6A0DAD",
    zorder=5,
    clip_on=False,
)

di_pad = 3.0
di_x = sx3 + di_pad
di_w = sw3 - 2 * di_pad
dy = y + DEMO_PAD  # build upward from bottom of container

for i in range(N_DEMOS):
    demo_num = N_DEMOS - i
    # Assistant (visually lower within demo block)
    msg_section(
        ax,
        di_x,
        dy,
        di_w,
        SH_DA,
        C_ASST,
        "Assistant",
        [
            f"{{scm_definition_{demo_num}}}",
            f"{{simulator_code_{demo_num}}}",
            f"{{feedback_{demo_num}}}",
        ],
        ec=C_ASST_F,
    )
    dy += SH_DA + GAP
    # User (visually higher within demo block)
    msg_section(
        ax, di_x, dy, di_w, SH_DU, C_USER, "User", ["(see current input)"], ec=C_USER_F
    )
    dy += SH_DU

    if i < N_DEMOS - 1:
        sep_y = dy + DGAP / 2
        ax.plot(
            [sx3 + 1, sx3 + sw3 - 1],
            [sep_y, sep_y],
            color="#9C6EAF",
            lw=0.4,
            ls="--",
            zorder=5,
            clip_on=False,
        )
        dy += DGAP

y += demo_container_h + GAP

# System (top)
msg_section(
    ax,
    sx3,
    y,
    sw3,
    SH_SYS,
    C_SYS,
    "System",
    ["{instruction_template}", "+ {system_description}"],
    ec=C_SYS_F,
)

# ── Loop arrow: bottom of box3 → below → left → up into bottom of box2 ───────
LOOP_BOT = min(y3b, y2b) - 8  # pt below the lower of the two box bottoms

x3_mid = x3 + BOX_W / 2
x2_mid = x2 + BOX_W / 2

# Single connected path — miter joins eliminate corner gaps
ax.plot(
    [x3_mid, x3_mid, x2_mid, x2_mid],
    [y3b, LOOP_BOT, LOOP_BOT, y2b],
    color=C_ARR,
    lw=1.0,
    clip_on=False,
    solid_joinstyle="miter",
    solid_capstyle="butt",
)
# Arrowhead at the entry point of box2
ax.annotate(
    "",
    xy=(x2_mid, y2b),
    xytext=(x2_mid, y2b - 2),
    arrowprops=dict(arrowstyle="->", color=C_ARR, lw=1.0),
    annotation_clip=False,
)
ax.text(
    (x2_mid + x3_mid) / 2,
    LOOP_BOT - 3,
    "weighting",
    ha="center",
    va="top",
    fontsize=FS,
    color=C_ARR,
    style="italic",
    clip_on=False,
)

# ── Legend: centred below both boxes ─────────────────────────────────────────
# Anchor in data coordinates: midpoint between box centres, below LOOP_BOT.
# The drawn boxes extend far outside the axes (clip_on=False), so axes-fraction
# coords don't correspond to the visual centre — data coords do.
center_data_x = (x2 + x3 + BOX_W) / 2
ax.legend(
    handles=[
        mpatches.Patch(facecolor=C_SYS, edgecolor=C_SYS_F, label="System"),
        mpatches.Patch(facecolor=C_USER, edgecolor=C_USER_F, label="User"),
        mpatches.Patch(facecolor=C_ASST, edgecolor=C_ASST_F, label="Assistant"),
        mpatches.Patch(facecolor=C_OUT, edgecolor=C_OUT_F, label="Output"),
        mpatches.Patch(facecolor=C_DEMO, edgecolor="#9C6EAF", label="Few-shot demos"),
    ],
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(center_data_x, LOOP_BOT - 18),
    bbox_transform=ax.transData,
    fontsize=FS,
    frameon=True,
    edgecolor="#ADB5BD",
    handlelength=1.0,
    handleheight=0.8,
)

fig.tight_layout(pad=0.1)

os.makedirs("fig", exist_ok=True)

out_path = os.path.join(os.getcwd(), "fig", "llm_flow.svg")
fig.savefig(out_path, format="svg", bbox_inches="tight")
print(f"Saved → {out_path}")

out_path = os.path.join(os.getcwd(), "fig", "llm_flow.pdf")
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"Saved → {out_path}")
