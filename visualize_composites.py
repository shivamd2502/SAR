#!/usr/bin/env python3
"""
visualize_composites.py
=======================
Renders every RGB composite produced by ``generate_rgb_composites.py`` onto a
single figure so they can be compared side by side at a glance, instead of
opening each GeoTIFF one at a time in QGIS.

It looks in a composite directory (the ``--output-dir`` you passed to
``generate_rgb_composites.py``) and lays out, on one page:

  * OLD baseline    (R=HH_DN,      G=HV_DN,      B=HH/HV_DN)
  * Option 1 σ⁰     (R=HH_sigma0,  G=HV_sigma0,  B=HH/HV_sigma0)
  * Option 2 γ⁰     (R=HH_gamma0,  G=HV_gamma0,  B=HH/HV_gamma0)
  * Option 3 RVI    (R=HH_sigma0,  G=HV_sigma0,  B=RVI)
  * RVI  standalone (single-band, viridis colormap)
  * DPDI standalone (single-band, RdYlGn colormap)

Each 3-band composite is contrast-stretched per channel (2nd–98th percentile,
the same stretch used for the ``*_PREVIEW.tif`` files) so faint scenes are still
visible.  Single-band index layers are shown with a colour map and a colour bar.

Usage
-----
    # simplest: point at the folder generate_rgb_composites.py wrote to
    python visualize_composites.py --composite-dir composites/03JUN2026

    # or hand it individual files (any subset, in any order)
    python visualize_composites.py \
        --tif composites/03JUN2026/OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif \
        --tif composites/03JUN2026/RVI_linear.tif

    # control the output / DPI / downsampling for very large scenes
    python visualize_composites.py --composite-dir composites/03JUN2026 \
        --output composites/03JUN2026/ALL_composites.png --dpi 200 --downsample 4
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viz")

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio required.  pip install rasterio --break-system-packages")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")  # headless: write files without needing a display
import matplotlib.pyplot as plt


# ==========================================================================
# Which files to look for, in display order.  Each entry is:
#   (title, list-of-filename-glob-patterns, kind, colormap-for-single-band)
# The first pattern that matches inside the composite dir wins, so both the
# float32 composites and their *_PREVIEW.tif variants are picked up.
# ==========================================================================
COMPOSITE_SPECS = [
    ("OLD baseline\n(R=HH_DN, G=HV_DN, B=HH/HV_DN)",
     ["OLD_composite*_PREVIEW.tif", "OLD_composite*.tif"], "rgb", None),
    ("Option 1 — σ⁰ standard\n(R=HH σ⁰, G=HV σ⁰, B=HH/HV σ⁰)",
     ["OPT1_*_PREVIEW.tif", "OPT1_*.tif"], "rgb", None),
    ("Option 2 — γ⁰ terrain-aware\n(R=HH γ⁰, G=HV γ⁰, B=HH/HV γ⁰)",
     ["OPT2_*_PREVIEW.tif", "OPT2_*.tif"], "rgb", None),
    ("Option 3 — index-enhanced\n(R=HH σ⁰, G=HV σ⁰, B=RVI)",
     ["OPT3_*_PREVIEW.tif", "OPT3_*.tif"], "rgb", None),
    ("RVI (vegetation, 0–1)",
     ["RVI_linear.tif", "RVI*.tif"], "single", "viridis"),
    ("DPDI (soil vs veg, −1…+1)",
     ["DPDI_linear.tif", "DPDI*.tif"], "single", "RdYlGn"),
]


# ==========================================================================
# Reading / display helpers
# ==========================================================================

def _stretch(arr: np.ndarray) -> np.ndarray:
    """2nd–98th percentile stretch to [0, 1] for a single channel."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(arr)] = 0.0
    return out.astype(np.float32)


def load_rgb(path: str, downsample: int) -> np.ndarray:
    """Read a 3-band GeoTIFF -> stretched (H, W, 3) float array in [0, 1]."""
    with rasterio.open(path) as src:
        n = min(3, src.count)
        bands = [src.read(i + 1).astype(np.float32) for i in range(n)]
    if downsample > 1:
        bands = [b[::downsample, ::downsample] for b in bands]
    while len(bands) < 3:              # grayscale source -> replicate
        bands.append(bands[-1])
    return np.stack([_stretch(b) for b in bands], axis=-1)


def load_single(path: str, downsample: int) -> tuple[np.ndarray, float, float]:
    """Read band 1 of a GeoTIFF -> (array, vmin, vmax) using 2–98 pct limits."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return arr, 0.0, 1.0
    return arr, float(np.percentile(valid, 2)), float(np.percentile(valid, 98))


def find_first(composite_dir: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(composite_dir, pat)))
        if hits:
            return hits[0]
    return None


def classify_tif(path: str) -> tuple[str, str, str | None]:
    """Best-effort (title, kind, cmap) for an explicitly supplied --tif file."""
    name = os.path.basename(path)
    upper = name.upper()
    for title, patterns, kind, cmap in COMPOSITE_SPECS:
        for pat in patterns:
            # crude prefix match on the glob's leading token (e.g. "OPT1_")
            token = pat.split("*")[0].upper()
            if token and upper.startswith(token):
                return title, kind, cmap
    # fall back to band count
    try:
        with rasterio.open(path) as src:
            count = src.count
    except Exception:
        count = 3
    if count >= 3:
        return name, "rgb", None
    return name, "single", "viridis"


# ==========================================================================
# Main plotting
# ==========================================================================

def collect_panels(args) -> list[tuple[str, str, str, str | None]]:
    """Return a list of (path, title, kind, cmap) to render, in order."""
    panels: list[tuple[str, str, str, str | None]] = []

    if args.tif:
        for path in args.tif:
            if not os.path.isfile(path):
                log.warning("Skipping missing file: %s", path)
                continue
            title, kind, cmap = classify_tif(path)
            panels.append((path, title, kind, cmap))
        return panels

    for title, patterns, kind, cmap in COMPOSITE_SPECS:
        path = find_first(args.composite_dir, patterns)
        if path:
            panels.append((path, title, kind, cmap))
        else:
            log.info("Not found (skipping): %s", patterns[0])
    return panels


def render(panels, args) -> str:
    n = len(panels)
    ncols = min(args.cols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(args.panel_size * ncols, args.panel_size * nrows),
        squeeze=False,
    )
    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    axes_flat = axes.flatten()

    for ax, (path, title, kind, cmap) in zip(axes_flat, panels):
        log.info("Rendering %s", os.path.basename(path))
        try:
            if kind == "rgb":
                ax.imshow(load_rgb(path, args.downsample))
            else:
                arr, vmin, vmax = load_single(path, args.downsample)
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception as exc:                       # noqa: BLE001
            log.error("Failed to render %s: %s", path, exc)
            ax.text(0.5, 0.5, f"Could not read\n{os.path.basename(path)}",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes_flat[n:]:  # hide any leftover empty cells
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return args.output


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Render all RGB composites onto a single comparison page.")
    p.add_argument("--composite-dir", default="composites",
                   help="Folder written by generate_rgb_composites.py "
                        "(its --output-dir).")
    p.add_argument("--tif", action="append",
                   help="Explicit GeoTIFF(s) to include (repeatable). "
                        "Overrides --composite-dir auto-discovery.")
    p.add_argument("--output", default=None,
                   help="Output image path (.png or .pdf). "
                        "Default: <composite-dir>/ALL_composites.png")
    p.add_argument("--cols", type=int, default=3,
                   help="Number of panels per row (default 3).")
    p.add_argument("--panel-size", type=float, default=5.0,
                   help="Size in inches of each panel (default 5).")
    p.add_argument("--downsample", type=int, default=1,
                   help="Show every Nth pixel to speed up huge scenes "
                        "(default 1 = full resolution).")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output resolution (default 150).")
    p.add_argument("--title", default="EOS-04 RGB Composites — comparison",
                   help="Figure super-title.")
    args = p.parse_args(argv)

    if args.output is None:
        base = args.composite_dir if not args.tif else "."
        args.output = os.path.join(base, "ALL_composites.png")

    panels = collect_panels(args)
    if not panels:
        log.error("No composites found. Check --composite-dir / --tif.")
        return 1

    out = render(panels, args)
    log.info("Wrote single-page comparison with %d panel(s): %s",
             len(panels), out)
    print(f"\nDone — open {out} to see all composites on one page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
