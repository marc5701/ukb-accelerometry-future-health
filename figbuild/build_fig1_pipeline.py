"""Figure 1, method pipeline schematic for the ukb_disease paper.

One left-to-right inference flow on a 180 mm canvas, with real data thumbnails
(tri-axial week trace, AcceleRest/HA embedding heat-strips, within-day L-moment
violins, ranked 388-disease phenome) placed as insets. Schematic boxes and arrows
are drawn on a background axes in mm coordinates.

Run from figbuild/. Set UKB_ROOT to point at the data root; a scratch cache and the
output PDF/PNG are written under UKB_ROOT.
"""
import os, sys
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_tmp"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.colors import to_rgba
import figlib as F   # applies nature.mplstyle (Arimo), gives CAT_COLORS / cat_color

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
SCRATCH = f"{UKB_ROOT}/fig1_scratch"
os.makedirs(SCRATCH, exist_ok=True)
CACHE = f"{SCRATCH}/fig1_data.npz"

# colours
INK, SUB = "#1A1A1A", "#5A5A5A"
AR_FILL, AR_C = "#F4D3D6", "#D1495B"          # AcceleRest = cardiorespiratory (coral)
HA_FILL, HA_C = "#D5DFF3", "#3E6FC0"          # Human-activity = movement (royal blue)
OUR_FILL, OUR_C = "#EFEAF4", "#6A4C93"        # our trained steps (deep purple)
SNOW = "#5A7D92"
CX, CY, CZ = "#009E73", "#CC79A7", "#CC7A00"  # signal channels x / y / z
CXt, CYt, CZt = "#D6EFE7", "#F0DCE7", "#F7E9D0"
OPC = "#333333"

# real data
def build_cache():
    import h5py
    H5 = f"{UKB_ROOT}/preprocessed_h5/example_subject_0.h5"
    ED = f"{UKB_ROOT}/embeddings_v2/test/example_subject_0"
    with h5py.File(H5, "r") as f:
        fs = int(f.attrs["output_fs"])
        acc = f["/data/accelerometry"][:].astype(np.float32)        # (3, N)
    step = fs * 2                                                    # 0.5 Hz week trace
    xw = acc[:, ::step]
    tw = np.arange(xw.shape[1]) * step / fs / 3600.0                 # hours
    ar = np.load(f"{ED}/accelerest_embeddings.npy")                 # (n_win, 256)
    ha = np.load(f"{ED}/activity_mae_embeddings.npy")

    def zstrip(e, n):
        e = e[:n].astype(float)
        z = (e - np.nanmean(e, 0)) / (np.nanstd(e, 0) + 1e-6)
        return np.clip(z, -2.5, 2.5).T                              # (256, n)
    ar_strip = zstrip(ar, 1700)
    ha_strip = zstrip(ha, 5100)

    # within-day L-moment violins: one clean day of AR windows, 6 spread dims, z-scored
    wpd = ar.shape[0] // 7
    day = ar[2 * wpd:3 * wpd]
    day = day[~np.isnan(day[:, 0])]
    z = (day - day.mean(0)) / (day.std(0) + 1e-6)
    dims = np.linspace(0, 255, 6).astype(int)
    vio = np.clip(z[:, dims], -2.6, 2.6)
    np.savez(CACHE, xw=xw, tw=tw, ar_strip=ar_strip, ha_strip=ha_strip, vio=vio)

if not os.path.exists(CACHE):
    print("building data cache ...")
    build_cache()
D = np.load(CACHE)
xw, tw, ar_strip, ha_strip, vio = D["xw"], D["tw"], D["ar_strip"], D["ha_strip"], D["vio"]
phe = F.load_per_disease("embedding_disjoint_split")

# canvas
H = 72.0
fig = plt.figure(figsize=(180 / 25.4, H / 25.4))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 180); bg.set_ylim(0, H); bg.axis("off")

def inset(x, y, w, h):
    return fig.add_axes([x / 180, y / H, w / 180, h / H])

def rbox(x, y, w, h, fc, ec, lw=0.75, rs=1.3, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                       facecolor=fc, edgecolor=ec, lw=lw, zorder=z, mutation_aspect=1)
    bg.add_patch(p); return p

def arrow(x0, y0, x1, y1, color=INK, lw=1.0, rad=0.0, ms=5, ls="-"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", ls=ls,
                 shrinkA=1.5, shrinkB=1.5, zorder=4))

def txt(x, y, s, size=6, color=INK, weight="normal", ha="center", va="center", style="normal"):
    bg.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
            style=style, zorder=7)

def snowflake(cx, cy, r=1.5, color=SNOW, lw=0.7):
    for ang in (0, 60, 120):
        a = np.deg2rad(ang); dx, dy = r * np.cos(a), r * np.sin(a)
        bg.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=lw,
                solid_capstyle="round", zorder=6)
        for sgn in (1, -1):
            ex, ey = cx + sgn * dx, cy + sgn * dy
            for b in (a + np.deg2rad(145), a - np.deg2rad(145)):
                bg.plot([ex, ex + 0.4 * r * np.cos(b)], [ey, ey + 0.4 * r * np.sin(b)],
                        color=color, lw=lw, solid_capstyle="round", zorder=6)

def person(cx, cy, s=1.0, color=SUB):
    bg.add_patch(Circle((cx, cy + 1.1 * s), 0.7 * s, color=color, zorder=5))
    bg.add_patch(FancyBboxPatch((cx - 0.95 * s, cy - 1.9 * s), 1.9 * s, 2.6 * s,
                 boxstyle="round,pad=0,rounding_size=0.6", color=color, zorder=5))

YB = 43.0                                       # flow baseline

# ============================================================ STAGE 1  INPUT
person(14.5, 61, 0.95); person(18.5, 61, 0.95); person(22.5, 61, 0.95)
txt(18.5, 56.0, "97,696 UK Biobank participants", 6.3, INK, "bold")

axT = inset(4.5, 32.5, 28.5, 19.5)
for i, (k, c) in enumerate([("x", CX), ("y", CY), ("z", CZ)]):
    sig = xw[i] / (np.nanmax(np.abs(xw[i])) + 1e-9)
    lane = 2 - i
    axT.plot(tw, sig * 0.42 + lane, color=c, lw=0.32, rasterized=True)
    axT.text(-0.03 * tw.max(), lane, k, fontsize=5.6, color=c, weight="bold",
             ha="right", va="center")
for d in range(1, 7):
    axT.axvline(d * 24, color="#CFCFCF", lw=0.35, ls=(0, (1, 1.2)))
axT.set_xlim(-0.06 * tw.max(), tw.max()); axT.set_ylim(-0.65, 2.65); axT.axis("off")
txt(18.5, 30.0, "Wrist accelerometer  ·  one week", 5.9, SUB)
txt(18.5, 27.3, "100 Hz native, 30 Hz resampled", 5.4, SUB)

# ============================================================ STAGE 2  TOKENS
txt(42.0, 52.4, "patches", 6.0, INK)
chip_tints = [CXt, CYt, CZt]; chip_edge = [CX, CY, CZ]
for r in range(3):
    for c in range(5):
        cx0 = 37.2 + c * 2.15; cy0 = 48.0 - r * 2.15
        if c == 4:
            txt(cx0 + 0.8, cy0 + 0.8, "…", 6.5, SUB); continue
        bg.add_patch(FancyBboxPatch((cx0, cy0), 1.7, 1.7,
                     boxstyle="round,pad=0,rounding_size=0.4",
                     facecolor=chip_tints[r], edgecolor=chip_edge[r], lw=0.4, zorder=3))
txt(42.0, 39.3, "per axis", 5.4, SUB)
arrow(33.3, YB, 36.6, YB, lw=0.9, ms=5)             # trace -> chips
arrow(47.6, 45.5, 51.5, 56.0, lw=0.9, ms=5, rad=-0.15)   # chips -> encoder A
arrow(47.6, 40.5, 51.5, 30.0, lw=0.9, ms=5, rad=0.15)    # chips -> encoder B

# ============================================================ STAGE 3  ENCODERS (hero)
def encoder(y0, h, fill, ec, title, role, lines):
    x0, w, hh = 52.0, 43.0, 6.2
    rbox(x0, y0, w, h, fill, ec, lw=0.8, rs=1.4, z=2)
    rbox(x0, y0 + h - hh, w, hh, ec, ec, lw=0.8, rs=1.4, z=3)
    txt(x0 + w / 2, y0 + h - 2.2, title, 7.4, "white", "bold")
    txt(x0 + w / 2, y0 + h - 4.7, role, 5.4, "white")
    snowflake(x0 + w - 2.6, y0 + h - hh - 2.3, 1.7)
    yy = y0 + h - hh - 2.9
    for s, sz, col in lines:
        txt(x0 + w / 2, yy, s, sz, col); yy -= 3.0

encoder(46.0, 20.0, AR_FILL, AR_C, "AcceleRest", "cardiorespiratory encoder",
        [("transformer masked autoencoder", 6.0, INK),
         ("30-s patches · 128-min context", 5.4, SUB),
         ("respiratory-amplification pretraining", 5.8, INK),
         ("frozen, weights not updated", 5.4, SUB)])
encoder(19.0, 20.0, HA_FILL, HA_C, "Human-activity model", "movement encoder",
        [("transformer masked autoencoder", 6.0, INK),
         ("10-s patches · 50-min context", 5.4, SUB),
         ("frequency-aware spectrogram pretraining", 5.8, INK),
         ("frozen, weights not updated", 5.4, SUB)])
txt(73.5, 42.5, "pretrained (frozen) on 108,904 UK Biobank recordings", 5.6, SUB)

# ============================================================ STAGE 4  EMBEDDINGS
import matplotlib.cm as cm
vir = cm.get_cmap("viridis").copy(); vir.set_bad("#D9D9D9")
axA = inset(96.0, 49.0, 3.6, 13.0)
axA.imshow(ar_strip, aspect="auto", cmap=vir, interpolation="nearest", rasterized=True)
axA.set_xticks([]); axA.set_yticks([])
for sp in axA.spines.values(): sp.set_edgecolor(AR_C); sp.set_linewidth(0.7)
txt(101.0, 55.5, "256-d", 5.7, INK, ha="left")

axB = inset(96.0, 23.0, 3.6, 13.0)
axB.imshow(ha_strip, aspect="auto", cmap=vir, interpolation="nearest", rasterized=True)
axB.set_xticks([]); axB.set_yticks([])
for sp in axB.spines.values(): sp.set_edgecolor(HA_C); sp.set_linewidth(0.7)
txt(101.0, 29.5, "256-d", 5.7, INK, ha="left")

# ============================================================ STAGE 5  CONCAT
rbox(103.5, 39.5, 11.5, 7.0, "white", OPC, lw=0.55, rs=1.0, z=2)
txt(109.25, 43.0, "concat", 5.9, INK)
arrow(99.8, 55.0, 103.3, 45.5, lw=0.8, ms=4.5, rad=0.2)
arrow(99.8, 29.5, 103.3, 40.5, lw=0.8, ms=4.5, rad=-0.2)
# 512-d strip: real concatenated window vector, split provenance edge
v512 = np.concatenate([ar_strip[:, 60], ha_strip[:, 180]])[:, None]
ax5 = inset(116.4, 36.5, 2.4, 13.0)
ax5.imshow(np.tile(v512, (1, 6)), aspect="auto", cmap=vir, interpolation="nearest", rasterized=True)
ax5.set_xticks([]); ax5.set_yticks([])
for sp in ax5.spines.values(): sp.set_visible(False)
bg.plot([116.2, 116.2], [43.0, 49.5], color=AR_C, lw=1.4, zorder=5)
bg.plot([116.2, 116.2], [36.5, 43.0], color=HA_C, lw=1.4, zorder=5)
txt(118.0, 50.6, "512-d / window", 5.7, INK, ha="center")
arrow(115.2, 43.0, 116.2, 43.0, lw=0.8, ms=4)

# ============================================================ STAGE 6  POOLING (our contribution)
arrow(118.9, YB, 119.4, YB, lw=0.8, ms=4)
rbox(119.5, 29.0, 25.0, 30.0, OUR_FILL, OUR_C, lw=0.7, rs=1.6, z=1)
txt(131.5, 57.0, "Within-day distributional pooling", 5.4, OUR_C, "bold")
# a day's window embeddings as mini viridis columns under a "1 day" brace
ax6 = inset(120.7, 43.5, 9.6, 7.0)
ax6.imshow(ar_strip[:, 120:200][::4], aspect="auto", cmap=vir, interpolation="nearest",
           rasterized=True)
ax6.set_xticks([]); ax6.set_yticks([])
for sp in ax6.spines.values(): sp.set_edgecolor("#BBBBBB"); sp.set_linewidth(0.4)
bg.annotate("", xy=(130.3, 51.6), xytext=(120.7, 51.6),
            arrowprops=dict(arrowstyle="-", color=OUR_C, lw=0.7,
            connectionstyle="bar,fraction=0.2"))
txt(125.5, 53.7, "1 day", 5.3, OUR_C)
arrow(130.5, 47.0, 133.1, 47.0, lw=0.8, ms=4.5, color=OUR_C)
# violins: within-day distribution of 6 dims
axV = inset(133.3, 38.5, 10.3, 11.0)
parts = axV.violinplot([vio[:, j] for j in range(vio.shape[1])], positions=range(6),
                       widths=0.85, showextrema=False, showmedians=True)
for b in parts["bodies"]:
    b.set_facecolor("#C7C0D6"); b.set_edgecolor("#9A8FB5"); b.set_lw(0.35); b.set_alpha(0.95)
parts["cmedians"].set_color(OUR_C); parts["cmedians"].set_lw(0.7)
axV.set_xlim(-0.7, 5.7); axV.set_ylim(-3.0, 3.0); axV.axis("off")
txt(133.5, 36.5, "L-moment summary", 5.3, OUR_C)
txt(133.5, 34.1, "location · scale · skew", 5.2, SUB)

# ============================================================ STAGE 7  ROLLING-WINDOW COX (our step)
arrow(144.6, 43.0, 145.3, 43.0, lw=0.9, ms=4, color=OUR_C)
for i in range(4):
    fc = OUR_FILL if i < 3 else "white"
    bg.add_patch(Rectangle((148.1 + i * 2.0, 49.5), 1.7, 3.0, facecolor=fc,
                 edgecolor=OUR_C, lw=0.5, zorder=3))
bg.add_patch(Rectangle((148.0, 49.3), 8.1, 3.4, facecolor="none",
             edgecolor=OUR_C, lw=0.7, ls=(0, (2, 1.5)), zorder=4))
txt(151.2, 54.1, "3-day windows", 5.4, OUR_C)
rbox(145.3, 33.5, 13.4, 11.5, "white", OUR_C, lw=0.85, rs=1.3, z=2)
txt(152.0, 40.7, "Rolling-window", 5.2, INK, "bold")
txt(152.0, 37.9, "Cox model", 5.2, INK, "bold")
arrow(152.0, 49.1, 152.0, 45.3, lw=0.8, ms=4, color=OUR_C)

# ============================================================ STAGE 8  OUTPUT phenome
arrow(158.9, 41.0, 159.9, 42.0, lw=0.9, ms=5, rad=-0.1, color=INK)
axP = inset(160.2, 33.0, 14.8, 22.0)
order = phe.groupby("PhecodeCategory")["c"].mean().sort_values().index.tolist()
xr = 0
prot = {"Parkinson's disease (Primary)": "PD", "Alzheimer's disease": "AD"}
prot_xy = {}
for cat in order:
    sub = phe[phe.PhecodeCategory == cat].sort_values("c")
    xs = np.arange(xr, xr + len(sub))
    axP.scatter(xs, sub.c, s=2.2, color=F.cat_color(cat), linewidths=0, rasterized=True)
    for (xx, (_, row)) in zip(xs, sub.iterrows()):
        if row.PhecodeString in prot:
            prot_xy[prot[row.PhecodeString]] = (xx, row.c)
    xr += len(sub) + 3
axP.axhline(0.5, color="#9A9A9A", lw=0.6, ls=(0, (3, 2)))
axP.set_ylim(0.46, 0.97); axP.set_xlim(-4, xr + 1)
axP.set_xticks([])
axP.set_yticks([0.5, 0.7, 0.9]); axP.tick_params(length=2, labelsize=5.3, pad=1.2)
for sp in ("top", "right", "bottom"): axP.spines[sp].set_visible(False)
axP.spines["left"].set_linewidth(0.7)
axP.text(-0.02, 1.04, "C-index", transform=axP.transAxes, fontsize=5.4, color=SUB,
         ha="left", va="bottom")
ring = {"PD": (HA_C, (7, 4)), "AD": (AR_C, (7, -8))}     # PD->movement(blue), AD->cardio(coral)
for key, (rc, off) in ring.items():
    if key in prot_xy:
        xx, cc = prot_xy[key]
        axP.scatter([xx], [cc], s=20, facecolors="none", edgecolors=rc,
                    linewidths=1.0, zorder=6)
        axP.annotate(key, (xx, cc), xytext=off, textcoords="offset points",
                     fontsize=5.3, color=rc, fontweight="bold", ha="left",
                     va="center", zorder=7)
txt(167.4, 31.0, "390 incident outcomes", 5.5, INK, "bold")
txt(167.4, 28.6, "six-year hazard", 5.3, SUB)
txt(167.4, 26.4, "held-out test  5,253", 5.3, SUB)

# ============================================================ MODALITY CALLOUT (why two encoders)
cy = 12.5
bg.add_patch(Rectangle((116.0, cy - 0.9), 1.8, 1.8, facecolor=HA_C, edgecolor="none", zorder=5))
txt(118.6, cy, "movement encoder → Parkinson's disease", 5.8, INK, ha="left")
bg.add_patch(Rectangle((116.0, cy - 4.0), 1.8, 1.8, facecolor=AR_C, edgecolor="none", zorder=5))
txt(118.6, cy - 3.1, "cardiorespiratory encoder → Alzheimer's disease", 5.8, INK, ha="left")

fig.savefig(f"{SCRATCH}/fig1_pipeline.pdf")
fig.savefig(f"{SCRATCH}/fig1_pipeline.png", dpi=300)
plt.close(fig)
print("wrote", f"{SCRATCH}/fig1_pipeline.png")
