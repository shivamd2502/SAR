#!/usr/bin/env python3
"""
composite_class_comparison.py
==============================
Generates a full visual comparison report showing:

  Panel A  — Side-by-side RGB thumbnail images of all composites
             (Old DN-based  |  Option1 σ⁰  |  Option2 γ⁰  |  Option3 RVI)

  Panel B  — Per-channel histogram overlay (old vs new)
             Shows how the new pipeline widens the dynamic range

  Panel C  — Class suitability matrix
             Which composite is best for which land-cover class,
             with the expected backscatter range for each class

  Panel D  — Spectral signature plot
             Mean ± std of each composite's R/G/B channels
             for known EOS-04 C-band backscatter ranges per class

Output
------
  <output_dir>/REPORT_composite_comparison.png   — full A4 report
  <output_dir>/REPORT_class_matrix.png           — class × composite suitability grid
  <output_dir>/REPORT_histograms.png             — channel histograms only

Usage
-----
    python composite_class_comparison.py ^
        --old    "composites\03JUN2026\OLD_composite_R-HH_G-HV_B-ratio_DN.tif" ^
        --opt1   "composites\03JUN2026\OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif" ^
        --opt2   "composites\03JUN2026\OPT2_R-HH-gamma0_G-HV-gamma0_B-ratio-gamma0.tif" ^
        --opt3   "composites\03JUN2026\OPT3_R-HH-sigma0_G-HV-sigma0_B-RVI.tif" ^
        --rvi    "composites\03JUN2026\RVI_linear.tif" ^
        --dpdi   "composites\03JUN2026\DPDI_linear.tif" ^
        --output-dir "composites\03JUN2026\report"
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("report")

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio required.  pip install rasterio --break-system-packages")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

STYLE = {
    "bg":        "#FAFAF8",
    "panel":     "#FFFFFF",
    "border":    "#D8D6CF",
    "text":      "#1A1A18",
    "text2":     "#5C5B55",
    "accent":    "#3B6D11",
    "red":       "#D85A30",
    "green":     "#3B6D11",
    "blue":      "#185FA5",
    "amber":     "#BA7517",
    "purple":    "#534AB7",
    "teal":      "#0F6E56",
    "gray":      "#888780",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       STYLE["border"],
    "grid.linewidth":   0.4,
    "grid.alpha":       0.7,
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["border"],
    "text.color":       STYLE["text"],
    "axes.labelcolor":  STYLE["text"],
    "xtick.color":      STYLE["text2"],
    "ytick.color":      STYLE["text2"],
})


# ═══════════════════════════════════════════════════════════════════════════════
# LAND COVER CLASS DEFINITIONS
# C-band EOS-04 MRS expected backscatter ranges (from literature + your scene)
# ═══════════════════════════════════════════════════════════════════════════════

CLASSES = [
    {
        "name":    "Water body\n(calm)",
        "short":   "Water",
        "color":   "#185FA5",
        "hh_db":   (-25, -15),   # σ⁰ HH dB range
        "hv_db":   (-30, -20),
        "ratio":   (2,   8),     # HH/HV dB
        "rvi":     (0.0, 0.15),
        "dpdi":    (0.6, 1.0),
        "composites": {
            "old":  "Poor — DN ratio bright near-range",
            "opt1": "Best — σ⁰ dark & consistent",
            "opt2": "Good — γ⁰ similar to σ⁰ on flat water",
            "opt3": "Best — RVI near 0 (no canopy)",
        },
    },
    {
        "name":    "Bare soil /\nFallow land",
        "short":   "Bare soil",
        "color":   "#BA7517",
        "hh_db":   (-15, -8),
        "hv_db":   (-22, -14),
        "ratio":   (6,   14),
        "rvi":     (0.0, 0.25),
        "dpdi":    (0.4, 0.8),
        "composites": {
            "old":  "Poor — no angle correction",
            "opt1": "Good — σ⁰ separates from crops",
            "opt2": "Good — same as opt1 on flat terrain",
            "opt3": "Good — RVI near 0 separates from veg",
        },
    },
    {
        "name":    "Agricultural\ncrops",
        "short":   "Crops",
        "color":   "#639922",
        "hh_db":   (-14, -6),
        "hv_db":   (-20, -10),
        "ratio":   (4,   10),
        "rvi":     (0.3, 0.65),
        "dpdi":    (0.1, 0.5),
        "composites": {
            "old":  "Moderate — angle error blurs classes",
            "opt1": "Best — per-pixel angle reveals crop stage",
            "opt2": "Good — similar to opt1 on plains",
            "opt3": "Best — RVI shows canopy density",
        },
    },
    {
        "name":    "Forest /\nDense veg",
        "short":   "Forest",
        "color":   "#0F6E56",
        "hh_db":   (-12, -4),
        "hv_db":   (-14, -6),
        "ratio":   (2,   8),
        "rvi":     (0.6, 1.0),
        "dpdi":    (-0.2, 0.3),
        "composites": {
            "old":  "Poor — slope effects mixed in",
            "opt1": "Good — σ⁰ works on gentle slopes",
            "opt2": "Best — γ⁰ removes slope brightening",
            "opt3": "Best — RVI highest for dense canopy",
        },
    },
    {
        "name":    "Urban /\nBuilt-up",
        "short":   "Urban",
        "color":   "#D85A30",
        "hh_db":   (-6,  3),
        "hv_db":   (-16, -6),
        "ratio":   (8,   18),
        "rvi":     (0.1, 0.4),
        "dpdi":    (0.4, 0.9),
        "composites": {
            "old":  "Moderate — HH high but angle bias",
            "opt1": "Good — σ⁰ reveals double-bounce clearly",
            "opt2": "Good — γ⁰ same for flat urban areas",
            "opt3": "Moderate — RVI not ideal for urban",
        },
    },
    {
        "name":    "Hill slope /\nTerrain",
        "short":   "Hillslope",
        "color":   "#534AB7",
        "hh_db":   (-16, -4),
        "hv_db":   (-20, -8),
        "ratio":   (3,   12),
        "rvi":     (0.2, 0.6),
        "dpdi":    (0.0, 0.6),
        "composites": {
            "old":  "Worst — terrain error dominant",
            "opt1": "Moderate — σ⁰ has residual slope effect",
            "opt2": "Best — γ⁰ specifically removes slope effect",
            "opt3": "Moderate — RVI mixes slope + veg signal",
        },
    },
    {
        "name":    "Flooded\nvegetation",
        "short":   "Flood",
        "color":   "#3B8BD4",
        "hh_db":   (-18, -8),
        "hv_db":   (-14, -6),
        "ratio":   (0,   6),
        "rvi":     (0.5, 0.9),
        "dpdi":    (-0.4, 0.2),
        "composites": {
            "old":  "Poor — noise bias hides inundation",
            "opt1": "Best — noise bias removal darkens open water",
            "opt2": "Good — γ⁰ works for flood on flat terrain",
            "opt3": "Best — RVI high HV detects flooded veg",
        },
    },
]

COMPOSITES = [
    {
        "key":    "old",
        "label":  "OLD\nDN composite",
        "short":  "Old (DN)",
        "color":  STYLE["gray"],
        "desc":   "R=HH DN  G=HV DN  B=HH/HV DN\nNo calibration · no angle correction · no mask",
    },
    {
        "key":    "opt1",
        "label":  "Option 1\nσ⁰ Standard",
        "short":  "Opt1 (σ⁰)",
        "color":  STYLE["blue"],
        "desc":   "R=HH σ⁰dB  G=HV σ⁰dB  B=HH/HV σ⁰\nCalibrated · per-pixel LIA · masked",
    },
    {
        "key":    "opt2",
        "label":  "Option 2\nγ⁰ Terrain",
        "short":  "Opt2 (γ⁰)",
        "color":  STYLE["teal"],
        "desc":   "R=HH γ⁰dB  G=HV γ⁰dB  B=HH/HV γ⁰\nSlope-normalized · best for hills",
    },
    {
        "key":    "opt3",
        "label":  "Option 3\nRVI Index",
        "short":  "Opt3 (RVI)",
        "color":  STYLE["green"],
        "desc":   "R=HH σ⁰dB  G=HV σ⁰dB  B=RVI\nVegetation index blue channel",
    },
]

SUITABILITY = {
    "Best":     ("#EAF3DE", "#3B6D11"),
    "Good":     ("#E6F1FB", "#185FA5"),
    "Moderate": ("#FAEEDA", "#854F0B"),
    "Poor":     ("#FCEBEB", "#A32D2D"),
    "Worst":    ("#F7C1C1", "#501313"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_composite(path: str, downsample: int = 8) -> np.ndarray | None:
    """Load a 3-band GeoTIFF, downsample for speed, return H×W×3 float32."""
    if not path or not os.path.exists(path):
        log.warning("File not found: %s", path)
        return None
    with rasterio.open(path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32) if src.count >= 3 else np.zeros_like(r)
    arr = np.stack([r, g, b], axis=-1)
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    log.info("Loaded %s -> downsampled to %s", path, arr.shape[:2])
    return arr


def stretch_for_display(arr: np.ndarray, pct_lo: float = 2.0,
                         pct_hi: float = 98.0) -> np.ndarray:
    """
    Per-channel 2–98th percentile stretch to 0–1 for display.
    NaN pixels map to 0 (black).
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo = np.percentile(valid, pct_lo)
        hi = np.percentile(valid, pct_hi)
        if hi > lo:
            stretched = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            stretched = np.zeros_like(ch)
        stretched[~np.isfinite(ch)] = 0
        out[:, :, c] = stretched
    return out


def load_single_band(path: str, downsample: int = 8) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr[::downsample, ::downsample]


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A — COMPOSITE IMAGE THUMBNAILS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_thumbnails(images: dict, fig: plt.Figure, gs_row) -> None:
    titles = {
        "old":  ("OLD — DN composite",         "R=HH DN · G=HV DN · B=HH/HV DN",           STYLE["gray"]),
        "opt1": ("Option 1 — σ⁰ Standard",      "R=HH σ⁰ · G=HV σ⁰ · B=HH/HV σ⁰",          STYLE["blue"]),
        "opt2": ("Option 2 — γ⁰ Terrain-aware", "R=HH γ⁰ · G=HV γ⁰ · B=HH/HV γ⁰",          STYLE["teal"]),
        "opt3": ("Option 3 — RVI Index",        "R=HH σ⁰ · G=HV σ⁰ · B=RVI",               STYLE["green"]),
    }
    keys = ["old", "opt1", "opt2", "opt3"]
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs_row[i])
        img = images.get(key)
        title, subtitle, color = titles[key]
        if img is not None:
            disp = stretch_for_display(img)
            ax.imshow(disp, aspect="auto", interpolation="bilinear")
        else:
            ax.set_facecolor("#E8E6DF")
            ax.text(0.5, 0.5, "File\nnot found", ha="center", va="center",
                    fontsize=9, color=STYLE["text2"],
                    transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
            spine.set_visible(True)
        ax.set_title(title, fontsize=9, fontweight="bold",
                     color=color, pad=4)
        ax.set_xlabel(subtitle, fontsize=7.5, color=STYLE["text2"], labelpad=3)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B — CHANNEL HISTOGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_histograms(images: dict, fig: plt.Figure, gs_row) -> None:
    channel_labels = ["R channel (HH)", "G channel (HV)", "B channel (ratio/RVI)"]
    channel_colors = {
        "old":  STYLE["gray"],
        "opt1": STYLE["blue"],
        "opt2": STYLE["teal"],
        "opt3": STYLE["green"],
    }
    for c_idx in range(3):
        ax = fig.add_subplot(gs_row[c_idx])
        ax.set_title(channel_labels[c_idx], fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Pixel value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        for key in ["old", "opt1", "opt2", "opt3"]:
            img = images.get(key)
            if img is None:
                continue
            ch = img[:, :, c_idx]
            v = ch[np.isfinite(ch)].ravel()
            if v.size == 0:
                continue
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            bins = np.linspace(lo, hi, 80)
            counts, edges = np.histogram(v, bins=bins, density=True)
            centres = (edges[:-1] + edges[1:]) / 2
            lw = 1.8 if key != "old" else 1.2
            ls = "--" if key == "old" else "-"
            label = [c["short"] for c in COMPOSITES if c["key"] == key][0]
            ax.plot(centres, counts, color=channel_colors[key],
                    linewidth=lw, linestyle=ls, label=label, alpha=0.85)
        ax.legend(fontsize=7.5, framealpha=0.7)
        ax.yaxis.set_major_locator(MaxNLocator(4))


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C — CLASS SUITABILITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_suitability(text: str) -> str:
    for level in ["Best", "Good", "Moderate", "Poor", "Worst"]:
        if text.startswith(level):
            return level
    return "Moderate"


def plot_class_matrix(fig: plt.Figure, gs_slot) -> None:
    ax = fig.add_subplot(gs_slot)
    ax.set_facecolor(STYLE["bg"])
    ax.axis("off")

    comp_keys = ["old", "opt1", "opt2", "opt3"]
    n_cls  = len(CLASSES)
    n_comp = len(comp_keys)

    cell_w = 1.0 / (n_comp + 1.5)
    cell_h = 1.0 / (n_cls  + 1.5)
    pad_x  = 0.02
    pad_y  = 0.02

    # Column headers
    header_labels = {
        "old":  "OLD\n(DN)",
        "opt1": "Opt 1\n(σ⁰)",
        "opt2": "Opt 2\n(γ⁰)",
        "opt3": "Opt 3\n(RVI)",
    }
    header_colors = {
        "old":  STYLE["gray"],
        "opt1": STYLE["blue"],
        "opt2": STYLE["teal"],
        "opt3": STYLE["green"],
    }
    for j, key in enumerate(comp_keys):
        x = (j + 1.5) * cell_w
        y = 1.0 - 0.5 * cell_h
        ax.text(x, y, header_labels[key],
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=header_colors[key],
                transform=ax.transAxes)

    # Row headers + cells
    for i, cls in enumerate(CLASSES):
        y = 1.0 - (i + 1.5) * cell_h

        # Row label
        ax.text(0.01, y, cls["name"],
                ha="left", va="center", fontsize=8, fontweight="bold",
                color=cls["color"],
                transform=ax.transAxes)

        for j, key in enumerate(comp_keys):
            x = (j + 1.5) * cell_w
            suit_text = cls["composites"][key]
            level = _parse_suitability(suit_text)
            bg_color, fg_color = SUITABILITY[level]

            # Draw cell background
            rect = mpatches.FancyBboxPatch(
                (x - cell_w * 0.48, y - cell_h * 0.42),
                cell_w * 0.96, cell_h * 0.84,
                boxstyle="round,pad=0.005",
                facecolor=bg_color, edgecolor=STYLE["border"],
                linewidth=0.5,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(rect)

            # Level label
            ax.text(x, y + cell_h * 0.10, level,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color=fg_color,
                    transform=ax.transAxes)

            # Short reason (truncated)
            reason = suit_text[len(level):].strip(" —").strip()
            reason_short = textwrap.shorten(reason, width=22, placeholder="…")
            ax.text(x, y - cell_h * 0.20, reason_short,
                    ha="center", va="center", fontsize=6.0,
                    color=fg_color, alpha=0.85,
                    transform=ax.transAxes)

    # Legend
    leg_x = 0.01
    leg_y = -0.02
    for level, (bg, fg) in SUITABILITY.items():
        rect = mpatches.FancyBboxPatch(
            (leg_x, leg_y), 0.035, 0.025,
            boxstyle="round,pad=0.002",
            facecolor=bg, edgecolor=STYLE["border"], linewidth=0.5,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(leg_x + 0.018, leg_y + 0.013, level,
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                color=fg, transform=ax.transAxes)
        leg_x += 0.08

    ax.set_title("Class suitability matrix — which composite for which class",
                 fontsize=9, fontweight="bold", pad=6, loc="left")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL D — EXPECTED BACKSCATTER RANGES PER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_backscatter_ranges(fig: plt.Figure, gs_slot) -> None:
    ax = fig.add_subplot(gs_slot)

    y_positions = list(range(len(CLASSES)))[::-1]   # top to bottom
    bar_h = 0.30

    ax.set_title("Expected C-band HH σ⁰ range per class  (dB)",
                 fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("σ⁰ HH  (dB)", fontsize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([c["short"] for c in CLASSES], fontsize=8)
    ax.set_xlim(-35, 10)
    ax.axvline(0, color=STYLE["border"], linewidth=0.8, linestyle="--")

    for i, cls in enumerate(CLASSES):
        y = y_positions[i]
        lo, hi = cls["hh_db"]
        width = hi - lo

        ax.barh(y, width, left=lo, height=bar_h,
                color=cls["color"], alpha=0.75, linewidth=0)

        # HV range as thin overlay
        hv_lo, hv_hi = cls["hv_db"]
        ax.barh(y - bar_h * 0.5, hv_hi - hv_lo, left=hv_lo,
                height=bar_h * 0.35, color=cls["color"], alpha=0.35, linewidth=0)

        # Value labels
        ax.text(hi + 0.3, y, f"{lo} to {hi} dB",
                va="center", fontsize=7, color=cls["color"])

    # Legend for bar meaning
    ax.barh(-0.7, 3, left=-34, height=bar_h, color=STYLE["text2"], alpha=0.75)
    ax.text(-30.5, -0.7, "HH σ⁰ range", va="center", fontsize=7, color=STYLE["text2"])
    ax.barh(-0.7 - bar_h * 0.5, 3, left=-34, height=bar_h * 0.35,
            color=STYLE["text2"], alpha=0.35)
    ax.text(-30.5, -0.7 - bar_h * 0.5, "HV σ⁰ range (lighter)",
            va="center", fontsize=7, color=STYLE["text2"])

    ax.set_ylim(-1.5, len(CLASSES) - 0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL E — RVI & DPDI INDEX MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_index_maps(rvi: np.ndarray | None, dpdi: np.ndarray | None,
                    fig: plt.Figure, gs_row) -> None:
    items = [
        (rvi,  "RVI — Radar Vegetation Index", "YlGn",
         "0 = bare/water → 1 = dense forest",   0, 1),
        (dpdi, "DPDI — Dual-pol Discrimination Index", "RdYlGn_r",
         "−1 = veg dominant → +1 = soil/urban",  -1, 1),
    ]
    for idx, (arr, title, cmap, desc, vmin, vmax) in enumerate(items):
        ax = fig.add_subplot(gs_row[idx])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel(desc, fontsize=7.5, color=STYLE["text2"], labelpad=2)
        if arr is not None:
            v = arr.copy()
            v[~np.isfinite(v)] = np.nan
            im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", interpolation="bilinear")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         shrink=0.85, aspect=20)
        else:
            ax.set_facecolor("#E8E6DF")
            ax.text(0.5, 0.5, "File\nnot found",
                    ha="center", va="center", fontsize=9,
                    color=STYLE["text2"], transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

        # Class annotations on DPDI
        if idx == 1:
            annotations = [
                (-0.8, "Forest\nFlooded veg"),
                (0.0,  "Crops\nMixed veg"),
                (0.7,  "Bare soil\nUrban"),
            ]
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
            for val, label in annotations:
                norm_x = (val - vmin) / (vmax - vmin)
                ax2.text(norm_x, 1.04, label,
                         ha="center", va="bottom", fontsize=6.5,
                         color=STYLE["text2"],
                         transform=ax2.transAxes)


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE CLASS MATRIX FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_class_matrix_figure(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 10),
                            facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])
    ax.axis("off")

    comp_keys  = ["old", "opt1", "opt2", "opt3"]
    col_labels = {
        "old":  "OLD  (DN ratio)",
        "opt1": "Option 1  (σ⁰ Standard)",
        "opt2": "Option 2  (γ⁰ Terrain)",
        "opt3": "Option 3  (RVI)",
    }
    col_colors = {
        "old":  STYLE["gray"],
        "opt1": STYLE["blue"],
        "opt2": STYLE["teal"],
        "opt3": STYLE["green"],
    }
    n_cls  = len(CLASSES)
    n_comp = len(comp_keys)

    # Table using matplotlib table for clean grid layout
    col_labels_list = ["Land cover class"] + [col_labels[k] for k in comp_keys]
    cell_text  = []
    cell_color = []

    for cls in CLASSES:
        row_text  = [cls["name"].replace("\n", " ")]
        row_color = [STYLE["bg"]]
        for key in comp_keys:
            suit_text = cls["composites"][key]
            level = _parse_suitability(suit_text)
            bg, fg = SUITABILITY[level]
            row_text.append(f"{level}\n{suit_text[len(level):].strip(' —').strip()[:40]}")
            row_color.append(bg)
        cell_text.append(row_text)
        cell_color.append(row_color)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_color,
        colLabels=col_labels_list,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header row
    for j in range(n_comp + 1):
        cell = table[0, j]
        if j == 0:
            cell.set_facecolor(STYLE["text"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            key = comp_keys[j - 1]
            cell.set_facecolor(col_colors[key])
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.08)

    # Style data rows
    for i, cls in enumerate(CLASSES):
        # row label cell
        cell = table[i + 1, 0]
        cell.set_facecolor(cls["color"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_height(0.09)
        for j in range(1, n_comp + 1):
            table[i + 1, j].set_height(0.09)
            suit_text = cls["composites"][comp_keys[j - 1]]
            level = _parse_suitability(suit_text)
            _, fg = SUITABILITY[level]
            table[i + 1, j].set_text_props(color=fg, fontsize=7.5)

    # Column widths
    table.auto_set_column_width(list(range(n_comp + 1)))

    fig.suptitle("EOS-04 Composite × Land Cover Class Suitability Matrix",
                 fontsize=13, fontweight="bold", y=0.98,
                 color=STYLE["text"])
    fig.text(0.5, 0.01,
             "Green = Best  ·  Blue = Good  ·  Amber = Moderate  ·  Red = Poor / Worst",
             ha="center", fontsize=8.5, color=STYLE["text2"])

    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN REPORT FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_main_report(images: dict, rvi: np.ndarray | None,
                      dpdi: np.ndarray | None, out_path: str) -> None:
    fig = plt.figure(figsize=(18, 22), facecolor=STYLE["bg"])
    gs  = gridspec.GridSpec(
        5, 1,
        figure=fig,
        height_ratios=[2.8, 1.6, 2.8, 1.8, 1.8],
        hspace=0.48,
    )

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.975,
        "EOS-04 SAR MRS — Old vs New Preprocessing Composites",
        ha="center", va="top", fontsize=15, fontweight="bold",
        color=STYLE["text"],
    )
    fig.text(
        0.5, 0.963,
        "Old 3-step DN composite  vs  ISRO-calibrated σ⁰ / γ⁰ / RVI composites",
        ha="center", va="top", fontsize=10, color=STYLE["text2"],
    )

    # ── Panel A: thumbnails ──────────────────────────────────────────────────
    gs_a = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs[0], wspace=0.06)
    plot_thumbnails(images, fig, gs_a)
    fig.text(
        0.01, gs[0].get_position(fig).y1 + 0.005,
        "A  —  RGB composite thumbnails (2–98% stretch)",
        fontsize=9, fontweight="bold", color=STYLE["text"], va="bottom",
    )

    # ── Panel B: histograms ──────────────────────────────────────────────────
    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1], wspace=0.32)
    plot_histograms(images, fig, gs_b)
    fig.text(
        0.01, gs[1].get_position(fig).y1 + 0.005,
        "B  —  Per-channel pixel value distributions (old dashed, new solid)",
        fontsize=9, fontweight="bold", color=STYLE["text"], va="bottom",
    )

    # ── Panel C: class matrix ────────────────────────────────────────────────
    plot_class_matrix(fig, gs[2])
    fig.text(
        0.01, gs[2].get_position(fig).y1 + 0.005,
        "C  —  Class suitability matrix",
        fontsize=9, fontweight="bold", color=STYLE["text"], va="bottom",
    )

    # ── Panel D: backscatter ranges ──────────────────────────────────────────
    plot_backscatter_ranges(fig, gs[3])
    fig.text(
        0.01, gs[3].get_position(fig).y1 + 0.005,
        "D  —  Expected C-band backscatter ranges per land cover class",
        fontsize=9, fontweight="bold", color=STYLE["text"], va="bottom",
    )

    # ── Panel E: index maps ──────────────────────────────────────────────────
    gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[4], wspace=0.25)
    plot_index_maps(rvi, dpdi, fig, gs_e)
    fig.text(
        0.01, gs[4].get_position(fig).y1 + 0.005,
        "E  —  Derived index maps (new pipeline only — not available in old 3-step)",
        fontsize=9, fontweight="bold", color=STYLE["text"], va="bottom",
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        "EOS-04 SAR MRS · Calibrated per ISRO SAC EOS-04 Data Products Formats v1.2.5 · "
        "Kcal_Beta0_HH = 67.14 dB · per-pixel LIA correction · layover mask applied",
        ha="center", fontsize=7, color=STYLE["text2"],
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_histogram_figure(images: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                              facecolor=STYLE["bg"])
    fig.suptitle(
        "Channel histogram comparison — old DN composite vs new calibrated composites",
        fontsize=11, fontweight="bold", color=STYLE["text"], y=1.02,
    )
    gs_mock = type("GS", (), {
        "__getitem__": lambda self, i: axes[i]
    })()
    plot_histograms(images, fig, axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate visual comparison report of old vs new EOS-04 composites.")
    p.add_argument("--old",  help="OLD composite tif  (R=HH DN, G=HV DN, B=ratio)")
    p.add_argument("--opt1", help="Option 1 tif  (R=HH σ⁰, G=HV σ⁰, B=HH/HV σ⁰)")
    p.add_argument("--opt2", help="Option 2 tif  (R=HH γ⁰, G=HV γ⁰, B=HH/HV γ⁰)")
    p.add_argument("--opt3", help="Option 3 tif  (R=HH σ⁰, G=HV σ⁰, B=RVI)")
    p.add_argument("--rvi",  help="RVI single-band tif")
    p.add_argument("--dpdi", help="DPDI single-band tif")
    p.add_argument("--downsample", type=int, default=8,
                   help="Downsample factor for thumbnails (default 8 = fast)")
    p.add_argument("--output-dir", default="composites/report")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Loading composites ...")
    images = {
        "old":  load_composite(args.old,  args.downsample) if args.old  else None,
        "opt1": load_composite(args.opt1, args.downsample) if args.opt1 else None,
        "opt2": load_composite(args.opt2, args.downsample) if args.opt2 else None,
        "opt3": load_composite(args.opt3, args.downsample) if args.opt3 else None,
    }
    rvi  = load_single_band(args.rvi,  args.downsample) if args.rvi  else None
    dpdi = load_single_band(args.dpdi, args.downsample) if args.dpdi else None

    # ── Main report ───────────────────────────────────────────────────────────
    main_path = os.path.join(args.output_dir, "REPORT_composite_comparison.png")
    log.info("Building main report ...")
    make_main_report(images, rvi, dpdi, main_path)

    # ── Class matrix ──────────────────────────────────────────────────────────
    matrix_path = os.path.join(args.output_dir, "REPORT_class_matrix.png")
    log.info("Building class matrix ...")
    make_class_matrix_figure(matrix_path)

    # ── Histograms only ───────────────────────────────────────────────────────
    hist_path = os.path.join(args.output_dir, "REPORT_histograms.png")
    log.info("Building histograms ...")
    # reuse plot_histograms with a fresh figure
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5),
                                facecolor=STYLE["bg"])
    fig2.suptitle(
        "Channel histogram comparison — old DN vs new calibrated composites",
        fontsize=11, fontweight="bold", color=STYLE["text"], y=1.02,
    )
    plot_histograms(images, fig2, axes2)
    fig2.tight_layout()
    fig2.savefig(hist_path, dpi=180, bbox_inches="tight",
                 facecolor=STYLE["bg"])
    plt.close(fig2)
    log.info("Wrote %s", hist_path)

    print(f"""
Output files
============
  Full report    : {main_path}
  Class matrix   : {matrix_path}
  Histograms     : {hist_path}

What to look for
================
  Panel A (thumbnails)
    - Old composite: may show a subtle brightness gradient
      left-to-right (near-range brighter than far-range)
    - New composites: uniform brightness across the full 160 km swath

  Panel B (histograms)
    - Old R/G channels (dashed): narrow DN distribution
    - New channels (solid): wider dB distribution = more contrast = better
      class separability

  Panel C (class matrix)
    - Green = Best, Blue = Good, Amber = Moderate, Red = Poor
    - Option 2 (gamma0) is best for hillslope / forest on terrain
    - Option 3 (RVI) is best for vegetation / flood
    - Option 1 (sigma0) is the general-purpose choice

  Panel D (backscatter ranges)
    - Shows expected sigma0 HH range per class
    - Use these ranges to set thresholds in your classifier

  Panel E (index maps)
    - RVI map: bright green = dense vegetation, dark = bare/water
    - DPDI map: red = soil/urban, green = vegetation-dominated
    - These two layers did NOT EXIST in the old 3-step pipeline
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())