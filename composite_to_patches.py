#!/usr/bin/env python3
"""
composite_to_patches.py
=======================
Single script that:
  1. Reads every composite GeoTIFF from composites/03JUN2026
     (OPT1 sigma0, OPT2 gamma0, OPT3 RVI, OLD baseline + standalone
      RVI / DPDI layers)
  2. Cuts each composite into fixed-size PNG patches using a sliding
     grid (configurable patch size & stride)
  3. Writes a Shapefile (.shp) for EVERY composite — each feature in
     the SHP is one patch polygon with attributes that let you verify
     in QGIS whether that patch is useful for labelling a particular
     class
  4. Writes one master SHP that covers all composites so you can
     cross-compare overlapping patches across options

Output layout
-------------
patches/
  03JUN2026/
    OPT1_sigma0/
      patches/          <- PNG files  e.g. OPT1_r0000_c0000.png
      OPT1_sigma0.shp   <- shapefile (+ .dbf .prj .shx)
    OPT2_gamma0/
      patches/
      OPT2_gamma0.shp
    OPT3_rvi/
      patches/
      OPT3_rvi.shp
    OLD_baseline/
      patches/
      OLD_baseline.shp
    master_all_composites.shp  <- one SHP, all composites

SHP attribute table (one row per patch)
----------------------------------------
  patch_id   : unique string  e.g. "OPT1_r0012_c0034"
  composite  : composite name  e.g. "OPT1_sigma0"
  row_idx    : patch row index in the grid
  col_idx    : patch column index in the grid
  px_row     : top-left pixel row in source raster
  px_col     : top-left pixel col in source raster
  valid_pct  : % of pixels that are finite (not NaN/masked)
  mean_r     : mean of R channel in patch
  mean_g     : mean of G channel in patch
  mean_b     : mean of B channel in patch
  std_r      : std of R channel in patch
  hh_hv_rat  : mean HH/HV ratio of the patch (B channel for OPT1/OPT2)
  rvi_mean   : mean RVI value (B channel for OPT3, else NaN)
  png_path   : relative path to PNG file
  use_flag   : initialised to 1 (set to 0 in QGIS for patches to skip)
  label      : empty string — fill this in QGIS for the class name

HOW TO USE IN QGIS
-------------------
1. Open QGIS → Drag the .shp file onto the canvas
2. Also drag the PNG folder (use "Add Raster Layer" for individual patches
   OR install the "Image Viewer" plugin to see thumbnails)
3. Open the Attribute Table of the SHP layer (F6)
4. Enable editing mode (pencil icon)
5. For each patch row, type the land-cover class name in the "label" column
   (or set use_flag=0 for patches that are too noisy / over water)
6. Use "Field Calculator" to bulk-label patches by their mean_r / rvi_mean:
     e.g.  rvi_mean > 0.5  → label = "Dense vegetation"
           mean_r  < -18   → label = "Water"
7. Save edits → Export → Save Features As → CSV for downstream use

Dependencies
------------
    pip install rasterio geopandas Pillow numpy
    (pyproj is installed automatically with geopandas)

Usage examples
--------------
  # default 256x256 patches, 256 stride (no overlap)
  python composite_to_patches.py ^
      --composite-dir "composites\\03JUN2026" ^
      --output-dir    "patches\\03JUN2026" ^
      --patch-size 256 --stride 256

  # 512x512 with 50% overlap (better for classification training)
  python composite_to_patches.py ^
      --composite-dir "composites\\03JUN2026" ^
      --output-dir    "patches\\03JUN2026" ^
      --patch-size 512 --stride 256

  # skip patches with >30% masked pixels
  python composite_to_patches.py ^
      --composite-dir "composites\\03JUN2026" ^
      --output-dir    "patches\\03JUN2026" ^
      --patch-size 256 --stride 256 --min-valid 70
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("patches")

# ── optional dependencies with friendly errors ───────────────────────────────
try:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.crs import CRS
except ImportError:
    print("ERROR: pip install rasterio"); sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: pip install Pillow"); sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
except ImportError:
    print("ERROR: pip install geopandas shapely"); sys.exit(1)


# =============================================================================
# 1.  COMPOSITE REGISTRY
#     Tells the script which files to look for and what each channel means
# =============================================================================

@dataclass
class CompositeSpec:
    name: str          # short name used in file paths & SHP attribute
    filename: str      # filename in composite_dir (float32 version, not preview)
    r_meaning: str     # human label for R channel (for SHP metadata)
    g_meaning: str     # human label for G channel
    b_meaning: str     # human label for B channel
    b_is_rvi: bool = False   # True when B channel is RVI (0-1), affects stats


# The five composites produced by generate_rgb_composites.py
COMPOSITE_REGISTRY: list[CompositeSpec] = [
    CompositeSpec(
        name="OPT1_sigma0",
        filename="OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif",
        r_meaning="HH_sigma0_dB",
        g_meaning="HV_sigma0_dB",
        b_meaning="HH_HV_ratio_sigma0_dB",
    ),
    CompositeSpec(
        name="OPT2_gamma0",
        filename="OPT2_R-HH-gamma0_G-HV-gamma0_B-ratio-gamma0.tif",
        r_meaning="HH_gamma0_dB",
        g_meaning="HV_gamma0_dB",
        b_meaning="HH_HV_ratio_gamma0_dB",
    ),
    CompositeSpec(
        name="OPT3_rvi",
        filename="OPT3_R-HH-sigma0_G-HV-sigma0_B-RVI.tif",
        r_meaning="HH_sigma0_dB",
        g_meaning="HV_sigma0_dB",
        b_meaning="RVI_0to1",
        b_is_rvi=True,
    ),
    CompositeSpec(
        name="OLD_baseline",
        filename="OLD_composite_R-HH_G-HV_B-ratio_DN.tif",
        r_meaning="HH_raw_DN",
        g_meaning="HV_raw_DN",
        b_meaning="HH_HV_ratio_DN",
    ),
]


# =============================================================================
# 2.  RASTER I/O
# =============================================================================

def read_composite(path: str) -> tuple[np.ndarray, object, object]:
    """
    Returns (array, transform, crs) where array is shape (3, H, W) float32.
    Band order: (R, G, B).
    """
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"{path} has only {src.count} band(s), expected 3.")
        arr = src.read([1, 2, 3]).astype(np.float32)  # shape (3, H, W)
        transform = src.transform
        crs = src.crs
    log.info("  Read %s  shape=%s", Path(path).name, arr.shape)
    return arr, transform, crs


# =============================================================================
# 3.  PATCH STRETCH TO PNG
#     Each channel is stretched independently using 2nd–98th percentile
#     so extreme outliers don't crush the visible contrast.
# =============================================================================

def stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Per-channel 2–98 percentile linear stretch → uint8 (0-255).
    NaN/inf treated as 0 in output.
    Input shape: (3, H, W) float32.
    Output shape: (H, W, 3) uint8 — PIL expects HWC order.
    """
    out = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        ch = arr[i]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi <= lo:
            continue
        stretched = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        stretched[~np.isfinite(ch)] = 0
        out[:, :, i] = stretched.astype(np.uint8)
    return out


def save_png(path: str, uint8_hwc: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(uint8_hwc, mode="RGB").save(path, optimize=False)


# =============================================================================
# 4.  GEOGRAPHIC PATCH BBOX
#     Converts pixel row/col extent to a geographic polygon so the SHP
#     can be overlaid correctly on the satellite image in QGIS.
# =============================================================================

def patch_bbox_geo(transform, row_start: int, col_start: int,
                   patch_size: int):
    """
    Returns a Shapely box polygon in the CRS of the raster for the patch
    covering pixel rows [row_start, row_start+patch_size) and
    cols [col_start, col_start+patch_size).
    rasterio transform: (col, row) → (x, y).
    """
    # top-left corner of patch
    x0, y0 = rasterio.transform.xy(transform, row_start,     col_start,     offset="ul")
    # bottom-right corner of patch
    x1, y1 = rasterio.transform.xy(transform, row_start + patch_size,
                                               col_start + patch_size, offset="ul")
    return shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# =============================================================================
# 5.  PATCH STATISTICS  (stored in SHP attribute table)
# =============================================================================

def patch_stats(patch: np.ndarray, spec: CompositeSpec) -> dict:
    """
    patch shape: (3, patch_size, patch_size) float32
    Returns a dict of scalar statistics for the SHP attribute table.
    """
    r, g, b = patch[0], patch[1], patch[2]

    total = r.size
    valid_mask = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
    valid_pct = 100.0 * valid_mask.sum() / total if total > 0 else 0.0

    def safe_stat(arr, func):
        v = arr[np.isfinite(arr)]
        return float(func(v)) if v.size > 0 else float("nan")

    mean_r = safe_stat(r, np.mean)
    mean_g = safe_stat(g, np.mean)
    mean_b = safe_stat(b, np.mean)
    std_r  = safe_stat(r, np.std)

    # B channel is HH/HV ratio in dB for OPT1/OPT2 and RVI for OPT3
    hh_hv_rat = mean_b if not spec.b_is_rvi else float("nan")
    rvi_mean  = mean_b if spec.b_is_rvi  else float("nan")

    return {
        "valid_pct": round(valid_pct, 2),
        "mean_r":    round(mean_r, 4) if math.isfinite(mean_r) else -9999.0,
        "mean_g":    round(mean_g, 4) if math.isfinite(mean_g) else -9999.0,
        "mean_b":    round(mean_b, 4) if math.isfinite(mean_b) else -9999.0,
        "std_r":     round(std_r,  4) if math.isfinite(std_r)  else -9999.0,
        "hh_hv_rat": round(hh_hv_rat, 4) if math.isfinite(hh_hv_rat) else -9999.0,
        "rvi_mean":  round(rvi_mean,  4) if math.isfinite(rvi_mean)  else -9999.0,
    }


# =============================================================================
# 6.  CORE PATCH EXTRACTION LOOP
# =============================================================================

def extract_patches(
    composite_path: str,
    spec: CompositeSpec,
    out_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    min_valid_pct: float = 50.0,
) -> list[dict]:
    """
    Slices the composite into patches, saves each as PNG, and returns a
    list of record dicts (one per patch) ready to be turned into a GeoDataFrame.

    Patches are on a regular grid:
        row_start = 0, stride, 2*stride, ...
        col_start = 0, stride, 2*stride, ...

    Edge patches (where the image boundary clips the patch) are SKIPPED
    to keep all patches the same exact pixel size — required for ML training.
    """
    patches_dir = os.path.join(out_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    arr, transform, crs = read_composite(composite_path)
    _, H, W = arr.shape

    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    total  = n_rows * n_cols
    log.info("  Grid: %d rows × %d cols = %d patches  (patch=%d stride=%d)",
             n_rows, n_cols, total, patch_size, stride)

    records = []
    saved = skipped_valid = skipped_edge = 0

    for ri in range(n_rows):
        for ci in range(n_cols):
            row0 = ri * stride
            col0 = ci * stride
            row1 = row0 + patch_size
            col1 = col0 + patch_size

            # edge guard (should not trigger given the grid formula above)
            if row1 > H or col1 > W:
                skipped_edge += 1
                continue

            patch = arr[:, row0:row1, col0:col1]   # (3, P, P)
            stats = patch_stats(patch, spec)

            # skip low-quality patches
            if stats["valid_pct"] < min_valid_pct:
                skipped_valid += 1
                continue

            # ── PNG filename ────────────────────────────────────────────
            png_name = f"{spec.name}_r{ri:04d}_c{ci:04d}.png"
            png_path = os.path.join(patches_dir, png_name)
            png_rel  = os.path.join("patches", png_name)   # relative path for SHP

            uint8 = stretch_to_uint8(patch)
            save_png(png_path, uint8)
            saved += 1

            # ── geographic bbox ─────────────────────────────────────────
            geom = patch_bbox_geo(transform, row0, col0, patch_size)

            # ── build record ────────────────────────────────────────────
            patch_id = f"{spec.name}_r{ri:04d}_c{ci:04d}"
            rec = {
                "patch_id":  patch_id,
                "composite": spec.name,
                "r_ch":      spec.r_meaning,
                "g_ch":      spec.g_meaning,
                "b_ch":      spec.b_meaning,
                "row_idx":   ri,
                "col_idx":   ci,
                "px_row":    row0,
                "px_col":    col0,
                "p_size":    patch_size,
                "stride":    stride,
                **stats,
                "png_path":  png_rel,
                "use_flag":  1,     # set to 0 in QGIS to exclude from training
                "label":     "",    # fill in QGIS with land-cover class name
                "geometry":  geom,
                "_crs":      crs,   # not a SHP field, used for GeoDataFrame init
            }
            records.append(rec)

    log.info("  Saved %d patches  (skipped: %d low-valid  %d edge)",
             saved, skipped_valid, skipped_edge)
    return records


# =============================================================================
# 7.  SHAPEFILE WRITER
# =============================================================================

def records_to_shp(records: list[dict], shp_path: str) -> None:
    """
    Converts the list of patch record dicts to a GeoDataFrame and saves as SHP.
    The '_crs' key is popped before creating the DataFrame.
    """
    if not records:
        log.warning("No records to write for %s — skipping.", shp_path)
        return

    crs = records[0].pop("_crs")
    for r in records[1:]:
        r.pop("_crs", None)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # SHP attribute names are max 10 chars — rename long keys
    gdf = gdf.rename(columns={
        "composite": "composite",
        "patch_id":  "patch_id",
        "valid_pct": "valid_pct",
        "hh_hv_rat": "hh_hv_rat",
        "rvi_mean":  "rvi_mean",
        "png_path":  "png_path",
        "use_flag":  "use_flag",
    })

    os.makedirs(os.path.dirname(shp_path) or ".", exist_ok=True)
    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    log.info("  SHP saved: %s  (%d features)", shp_path, len(gdf))


# =============================================================================
# 8.  MAIN
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Cut EOS-04 composites into PNG patches + SHP files for QGIS labelling.")
    p.add_argument("--composite-dir", required=True,
                    help='Folder containing composite GeoTIFFs '
                         '(e.g. "composites\\03JUN2026")')
    p.add_argument("--output-dir",    required=True,
                    help='Root output folder (e.g. "patches\\03JUN2026")')
    p.add_argument("--patch-size", type=int, default=256,
                    help="Patch height and width in pixels (default 256)")
    p.add_argument("--stride",     type=int, default=None,
                    help="Stride in pixels. Defaults to patch-size (no overlap). "
                         "Use half patch-size for 50%% overlap.")
    p.add_argument("--min-valid",  type=float, default=50.0,
                    help="Minimum %% of finite (non-NaN) pixels required to "
                         "save a patch (default 50).")
    p.add_argument("--composites", nargs="*", default=None,
                    help="Subset of composite names to process. "
                         "Options: OPT1_sigma0 OPT2_gamma0 OPT3_rvi OLD_baseline. "
                         "Default: all four.")
    args = p.parse_args(argv)

    stride = args.stride if args.stride else args.patch_size
    if stride > args.patch_size:
        log.warning("stride (%d) > patch_size (%d): there will be GAPS between patches.",
                    stride, args.patch_size)

    # filter registry if user requested a subset
    registry = COMPOSITE_REGISTRY
    if args.composites:
        registry = [s for s in COMPOSITE_REGISTRY if s.name in args.composites]
        if not registry:
            log.error("None of the requested composites match. "
                      "Valid names: %s", [s.name for s in COMPOSITE_REGISTRY])
            return 2

    all_records: list[dict] = []   # collects across all composites for master SHP

    for spec in registry:
        comp_path = os.path.join(args.composite_dir, spec.filename)
        if not os.path.isfile(comp_path):
            log.warning("Composite not found, skipping: %s", comp_path)
            continue

        log.info("━━ Processing composite: %s", spec.name)
        out_sub = os.path.join(args.output_dir, spec.name)

        records = extract_patches(
            composite_path=comp_path,
            spec=spec,
            out_dir=out_sub,
            patch_size=args.patch_size,
            stride=stride,
            min_valid_pct=args.min_valid,
        )

        if not records:
            log.warning("  No valid patches generated for %s.", spec.name)
            continue

        # per-composite SHP
        shp_path = os.path.join(out_sub, f"{spec.name}.shp")
        records_to_shp([dict(r) for r in records], shp_path)

        # accumulate for master SHP
        all_records.extend(records)

    # master SHP — all composites in one file for cross-comparison
    if all_records:
        master_shp = os.path.join(args.output_dir, "master_all_composites.shp")
        log.info("━━ Writing master SHP (%d total features) ...", len(all_records))
        records_to_shp(all_records, master_shp)

    # ── print QGIS usage guide ─────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              PATCH GENERATION COMPLETE — QGIS NEXT STEPS            ║
╚══════════════════════════════════════════════════════════════════════╝

Output folder: {args.output_dir}

Files per composite:
  {args.output_dir}/<composite_name>/
      patches/           ← PNG files (one per patch)
      <composite>.shp    ← shapefile (open this in QGIS)

Master shapefile (all composites):
  {args.output_dir}/master_all_composites.shp

──────────────────────────────────────────────────────────────────────
HOW TO LABEL IN QGIS  (step by step)
──────────────────────────────────────────────────────────────────────

STEP 1 — Load a composite shapefile
  • Open QGIS → Layer menu → Add Layer → Add Vector Layer
  • Select e.g.  patches/03JUN2026/OPT1_sigma0/OPT1_sigma0.shp
  • The patches appear as a grid of rectangles on the map

STEP 2 — Load the original satellite image for context
  • Layer menu → Add Raster Layer → hh.tif  (or the _PREVIEW.tif)
  • Move it below the SHP layer in the Layers panel

STEP 3 — Style the SHP by composite channel value
  • Right-click SHP layer → Properties → Symbology
  • Change "Single Symbol" to "Graduated"
  • Column: rvi_mean  (or mean_r / hh_hv_rat)
  • Classify → Apply
  • Now patches are colour-coded by mean RVI value —
    dark = low vegetation, bright = dense vegetation

STEP 4 — Open the Attribute Table and fill in the "label" column
  • Press F6 to open Attribute Table
  • Click the pencil icon to start editing
  • Click any row's "label" cell and type the class name:
      "Water", "Dense_forest", "Cropland", "Urban", "Barren", etc.
  • To bulk-label using rvi_mean: use Field Calculator
      Expression:  CASE WHEN rvi_mean > 0.6 THEN 'Dense_forest'
                        WHEN rvi_mean > 0.35 THEN 'Vegetation'
                        WHEN mean_r < -18   THEN 'Water'
                        ELSE 'Unclear' END
      Output field: label  (String, length 30)
  • Set use_flag = 0 for patches you want to EXCLUDE from training
    (e.g. over clouds, at image edges, mixed land/water)

STEP 5 — Save and export
  • Save edits (Ctrl+S)
  • Layer → Export → Save Features As → Format: CSV
  • This CSV has patch_id + label columns — feed directly into
    your classification training script

──────────────────────────────────────────────────────────────────────
WHICH COMPOSITE TO LABEL FOR WHICH CLASS
──────────────────────────────────────────────────────────────────────

  OPT1_sigma0.shp   → best for  water, urban, crops, bare soil
      Key attribute:  mean_r (HH σ⁰), mean_g (HV σ⁰), hh_hv_rat
      Water body:     mean_r < -18 dB
      Urban:          mean_r > -8 dB  AND  hh_hv_rat > 10 dB
      Dense crop:     mean_g > -14 dB AND  hh_hv_rat < 8 dB

  OPT2_gamma0.shp   → best for  forest on slopes (Himachal terrain)
      Key attribute:  mean_r (HH γ⁰), mean_g (HV γ⁰), hh_hv_rat
      Forested hill:  mean_g > -12 dB AND  hh_hv_rat < 6 dB

  OPT3_rvi.shp      → best for  vegetation health & crop stages
      Key attribute:  rvi_mean  (0 = bare, 1 = dense vegetation)
      Bare soil:      rvi_mean < 0.2
      Sparse veg:     0.2 < rvi_mean < 0.45
      Dense veg:      rvi_mean > 0.45
      Water:          rvi_mean < 0.1  AND  mean_r < -18 dB

  OLD_baseline.shp  → for COMPARISON only — do NOT use for final labels
      Shows uncalibrated patches so you can see the improvement

  master_all_composites.shp → use this if you want to see patches from
      ALL composites side-by-side and pick the best composite per region

""")
    return 0


if __name__ == "__main__":
    sys.exit(main())