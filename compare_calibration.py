#!/usr/bin/env python3
"""
compare_calibration.py
======================
Quantitatively demonstrates WHY the ISRO-documented EOS-04 pipeline
produces physically more correct backscatter than a generic 3-step
DN -> σ⁰ approach.

What it does
------------
1. Simulates what a "generic 3-step" pipeline would produce from the
   same DN values (wrong reference plane, no noise-bias, scene-centre
   incidence angle only).
2. Compares it pixel-by-pixel against the ISRO-documented Sigma0 output.
3. Prints a table of statistical differences and physical validity checks.
4. Saves a side-by-side difference GeoTIFF you can open in QGIS/SNAP.

Usage
-----
    python compare_calibration.py ^
        --isro-sigma0-db  calibrated\03JUN2026_HH\HH_sigma0_dB.tif ^
        --isro-sigma0-lin calibrated\03JUN2026_HH\HH_sigma0_linear.tif ^
        --band-meta "raw\...\BAND_META.txt" ^
        --lia "raw\...\..._lia.tif" ^
        --image "raw\...\hh.tif" ^
        --pol HH ^
        --output-dir calibrated\comparison

    # Optionally compare HH vs HV polarization ratio (physical validity)
    python compare_calibration.py ^
        --isro-sigma0-db  calibrated\03JUN2026_HH\HH_sigma0_dB.tif ^
        --isro-sigma0-lin calibrated\03JUN2026_HH\HH_sigma0_linear.tif ^
        --hv-sigma0-db    calibrated\03JUN2026_HV\HV_sigma0_dB.tif ^
        --band-meta "raw\...\BAND_META.txt" ^
        --lia "raw\...\..._lia.tif" ^
        --image "raw\...\hh.tif" ^
        --pol HH ^
        --output-dir calibrated\comparison
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare")

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio is required.  pip install rasterio")
    sys.exit(1)

# ── reuse calibration helpers from the main pipeline ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from eos04_radiometric_preprocessing import (
    BandMeta, read_geotiff, write_geotiff,
    kcal_linear_from_db, apply_noise_bias,
    compute_beta0, compute_sigma0, compute_gamma0,
    apply_valid_mask, MaskValue, to_db,
)


# ============================================================================
# 1.  GENERIC "3-STEP" BASELINE  (what most tutorials / SNAP defaults do)
# ============================================================================

def generic_3step_sigma0(dn: np.ndarray, kcal_db: float,
                          scene_centre_incidence_deg: float) -> np.ndarray:
    """
    Simulates a naïve 3-step pipeline:
      Step 1 – DN² (no noise-bias removal)
      Step 2 – divide by a SINGLE global calibration constant (no per-pixel lia)
      Step 3 – multiply by sin(scene_centre_incidence) for the WHOLE image.

    This is wrong for EOS-04 because:
      (a) EOS-04 DN is Beta0-referenced, not Sigma0-referenced
      (b) incidence angle across a 160 km MRS swath varies by ~5-10°
      (c) no noise bias removal degrades low-backscatter pixels
    """
    kcal_lin = kcal_linear_from_db(kcal_db)
    dn_sq = dn.astype(np.float64) ** 2          # Step 1: no noise-bias removal
    sigma0 = dn_sq / kcal_lin                    # Step 2: single constant
    # Step 3: single scene-centre angle for entire image
    sigma0 *= np.sin(np.deg2rad(scene_centre_incidence_deg))
    return sigma0


# ============================================================================
# 2.  STATISTICS HELPERS
# ============================================================================

def finite_stats(arr: np.ndarray, label: str) -> dict:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        log.warning("%s: no finite pixels found", label)
        return {}
    stats = {
        "label":   label,
        "n_valid": int(v.size),
        "mean_dB": float(np.mean(v)),
        "std_dB":  float(np.std(v)),
        "min_dB":  float(np.min(v)),
        "max_dB":  float(np.max(v)),
        "p5_dB":   float(np.percentile(v, 5)),
        "p50_dB":  float(np.percentile(v, 50)),
        "p95_dB":  float(np.percentile(v, 95)),
        "dynamic_range_dB": float(np.percentile(v, 95) - np.percentile(v, 5)),
    }
    return stats


def print_stats_table(stats_list: list[dict]) -> None:
    hdr = f"{'Label':<35} {'N valid':>10} {'Mean':>7} {'Std':>6} "
    hdr += f"{'P5':>7} {'P50':>7} {'P95':>7} {'Dyn.Range':>10}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for s in stats_list:
        if not s:
            continue
        print(
            f"{s['label']:<35} {s['n_valid']:>10,} "
            f"{s['mean_dB']:>7.2f} {s['std_dB']:>6.2f} "
            f"{s['p5_dB']:>7.2f} {s['p50_dB']:>7.2f} {s['p95_dB']:>7.2f} "
            f"{s['dynamic_range_dB']:>10.2f}"
        )
    print("=" * len(hdr))
    print("All dB values.  P5/P50/P95 = 5th/50th/95th percentile.\n")


def physical_validity_check(sigma0_db: np.ndarray, label: str) -> None:
    """
    EOS-04 MRS C-band typical Sigma0 ranges for natural land covers
    (well-established literature values; useful as a sanity check):

      Dense vegetation / forest  : -10 to -5  dB
      Agricultural crops         : -15 to -8  dB
      Open water (calm)          : -25 to -15 dB
      Urban (HH)                 :  -5 to +3  dB

    Flag values outside the physically plausible range -40 to +5 dB.
    """
    v = sigma0_db[np.isfinite(sigma0_db)]
    pct_unrealistic = 100.0 * np.sum((v < -40) | (v > 5)) / v.size if v.size else 0
    log.info("[%s] Pixels outside physically plausible σ⁰ range [-40,+5 dB]: %.2f%%",
             label, pct_unrealistic)
    if pct_unrealistic > 5:
        log.warning("[%s] >5%% of pixels are outside plausible σ⁰ range -- "
                    "check calibration.", label)
    else:
        log.info("[%s] Physical validity check PASSED (<5%% unrealistic pixels).",
                 label)
    return pct_unrealistic


def noise_floor_check(sigma0_db: np.ndarray, label: str,
                       expected_nesz_db: float = -25.0) -> None:
    """
    MRS C-band Noise Equivalent Sigma Zero (NESZ) is typically around
    -23 to -27 dB (from EOS-04 Table 2.0-4, radiometric resolution 3.1 dB
    for SLC; ground-range products are slightly better). Check what
    fraction of valid pixels is near or below the noise floor.
    """
    v = sigma0_db[np.isfinite(sigma0_db)]
    if v.size == 0:
        return
    pct_near_noise = 100.0 * np.sum(v <= expected_nesz_db) / v.size
    log.info("[%s] Pixels at/below NESZ (%.1f dB): %.2f%%",
             label, expected_nesz_db, pct_near_noise)


# ============================================================================
# 3.  DIFFERENCE ANALYSIS
# ============================================================================

def pixel_difference_stats(isro_db: np.ndarray, generic_db: np.ndarray) -> None:
    diff = isro_db - generic_db
    fin = np.isfinite(diff)
    d = diff[fin]
    if d.size == 0:
        log.warning("No finite pixels for difference computation.")
        return

    print("\n--- Pixel-level difference: ISRO pipeline − generic 3-step (dB) ---")
    print(f"  Mean bias   : {np.mean(d):.3f} dB")
    print(f"  Std Dev     : {np.std(d):.3f} dB")
    print(f"  Min diff    : {np.min(d):.3f} dB")
    print(f"  Max diff    : {np.max(d):.3f} dB")
    print(f"  |diff| > 1 dB : {100*np.mean(np.abs(d)>1):.1f}% of pixels")
    print(f"  |diff| > 2 dB : {100*np.mean(np.abs(d)>2):.1f}% of pixels")
    print(f"  |diff| > 3 dB : {100*np.mean(np.abs(d)>3):.1f}% of pixels")
    print()

    # Explain why the bias exists
    expected_bias_approx = (
        "  WHY: EOS-04 DN is Beta0-referenced, not Sigma0-referenced.\n"
        "  The generic pipeline mis-applies sin(scene_centre_incidence) once\n"
        "  but uses the same Kcal; the ISRO pipeline correctly converts Beta0\n"
        "  → Sigma0 per pixel using the local incidence angle.\n"
        "  A non-zero mean bias here directly proves the generic approach\n"
        "  introduces a systematic error across the swath.\n"
    )
    print(expected_bias_approx)


# ============================================================================
# 4.  HH / HV RATIO (physical cross-check)
# ============================================================================

def hh_hv_ratio_check(hh_sigma0_db: np.ndarray,
                       hv_sigma0_db: np.ndarray) -> None:
    """
    For C-band SAR, the HH/HV ratio for natural land cover is physically
    constrained to roughly +5 to +15 dB (vegetation) with urban targets
    higher. A well-calibrated dual-pol product should show this range.
    If the generic 3-step pipeline is used, both channels use the SAME
    scene-centre angle and no noise-bias correction, so the ratio is
    artificially flat.
    """
    ratio_db = hh_sigma0_db - hv_sigma0_db          # dB subtraction = linear ratio
    v = ratio_db[np.isfinite(ratio_db)]
    if v.size == 0:
        log.warning("HH/HV ratio: no finite pixels.")
        return
    pct_physical = 100.0 * np.sum((v >= 3) & (v <= 20)) / v.size
    print("--- HH/HV polarization ratio (σ⁰_HH - σ⁰_HV, in dB) ---")
    print(f"  Mean  : {np.mean(v):.2f} dB")
    print(f"  Std   : {np.std(v):.2f} dB")
    print(f"  P5    : {np.percentile(v, 5):.2f} dB")
    print(f"  P95   : {np.percentile(v, 95):.2f} dB")
    print(f"  Pixels in physical range [3,20 dB]: {pct_physical:.1f}%")
    if pct_physical > 70:
        print("  → GOOD: majority of pixels have a physically realistic HH/HV ratio.")
    else:
        print("  → WARNING: low fraction of physically realistic HH/HV ratios; "
              "check calibration or land-cover composition.")
    print()


# ============================================================================
# 5.  INCIDENCE ANGLE DEPENDENCY CHECK
# ============================================================================

def incidence_angle_dependency(sigma0_db: np.ndarray,
                                lia: np.ndarray,
                                label: str,
                                n_bins: int = 10) -> None:
    """
    For a well-calibrated Sigma0 image, the mean backscatter should be
    relatively flat across incidence angle bins (the sin(i) correction
    removes the look-angle trend). A generic pipeline that uses a single
    scene-centre angle will show a pronounced monotonic trend -- that is
    the calibration error.
    """
    fin = np.isfinite(sigma0_db) & np.isfinite(lia) & (lia > 0) & (lia < 90)
    s = sigma0_db[fin]
    i = lia[fin]
    if s.size == 0:
        return

    i_min, i_max = np.percentile(i, 1), np.percentile(i, 99)
    edges = np.linspace(i_min, i_max, n_bins + 1)

    print(f"\n--- Incidence angle dependency for [{label}] ---")
    print(f"  {'Incidence bin (°)':<22} {'Mean σ⁰ (dB)':>13} {'Count':>10}")
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask_bin = (i >= lo) & (i < hi)
        if mask_bin.sum() == 0:
            continue
        m = float(np.mean(s[mask_bin]))
        means.append(m)
        print(f"  {lo:.1f} – {hi:.1f}°{'':<10} {m:>13.3f} {mask_bin.sum():>10,}")

    if len(means) > 2:
        trend = np.polyfit(np.arange(len(means)), means, 1)[0]
        print(f"\n  Trend across bins: {trend:+.3f} dB / bin")
        if abs(trend) < 0.3:
            print("  → GOOD: σ⁰ is flat across incidence angles — "
                  "per-pixel lia correction is working.")
        else:
            print("  → NOTE: residual trend detected — could indicate terrain "
                  "variation across swath (normal for MRS) or calibration issue.")
    print()


# ============================================================================
# 6.  MAIN
# ============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compare ISRO-documented EOS-04 calibration vs generic "
                    "3-step approach.")
    p.add_argument("--isro-sigma0-db",  required=True,
                    help="ISRO pipeline output: HH_sigma0_dB.tif")
    p.add_argument("--isro-sigma0-lin", required=True,
                    help="ISRO pipeline output: HH_sigma0_linear.tif")
    p.add_argument("--hv-sigma0-db",
                    help="(optional) HV sigma0_dB.tif for HH/HV ratio check")
    p.add_argument("--band-meta",  required=True)
    p.add_argument("--lia",        required=True,
                    help="Local incidence angle GeoTIFF (*_lia.tif)")
    p.add_argument("--image",      required=True,
                    help="Original DN GeoTIFF (hh.tif)")
    p.add_argument("--mask",       help="Layover/shadow mask GeoTIFF (optional)")
    p.add_argument("--pol",        required=True)
    p.add_argument("--output-dir", default="calibrated\\comparison")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────
    log.info("Loading ISRO Sigma0 (dB) ...")
    isro_s0_db  = read_geotiff(args.isro_sigma0_db).array
    profile     = read_geotiff(args.isro_sigma0_db).profile
    isro_s0_lin = read_geotiff(args.isro_sigma0_lin).array

    log.info("Loading original DN ...")
    dn_raster   = read_geotiff(args.image)
    dn          = dn_raster.array

    log.info("Loading local incidence angle ...")
    lia         = read_geotiff(args.lia).array

    mask = None
    if args.mask:
        mask = read_geotiff(args.mask).array.astype(np.uint16)

    meta = BandMeta(args.band_meta)
    kcal_db   = meta.calibration_constant_db(args.pol, plane="Beta0")
    scene_inc = meta.get_float("IncidenceAngle") or float(np.nanmedian(lia[lia > 0]))
    log.info("Scene-centre incidence angle used for generic pipeline: %.2f°", scene_inc)

    # ── generic 3-step ─────────────────────────────────────────────────────
    log.info("Computing generic 3-step Sigma0 ...")
    gen_s0_lin = generic_3step_sigma0(dn, kcal_db, scene_inc)
    gen_s0_db  = to_db(gen_s0_lin)

    # apply mask so comparison is apples-to-apples
    if mask is not None:
        isro_s0_db  = apply_valid_mask(isro_s0_db,  mask, valid_only=True)
        gen_s0_db   = apply_valid_mask(gen_s0_db,   mask, valid_only=True)
        isro_s0_lin = apply_valid_mask(isro_s0_lin, mask, valid_only=True)

    # ── statistics comparison ──────────────────────────────────────────────
    stats = [
        finite_stats(isro_s0_db, f"ISRO documented σ⁰_dB ({args.pol})"),
        finite_stats(gen_s0_db,  f"Generic 3-step  σ⁰_dB ({args.pol})"),
    ]
    print_stats_table(stats)

    # ── pixel-level difference ─────────────────────────────────────────────
    pixel_difference_stats(isro_s0_db, gen_s0_db)

    # ── physical validity ─────────────────────────────────────────────────
    print("--- Physical validity checks ---")
    pct_bad_isro    = physical_validity_check(isro_s0_db, f"ISRO ({args.pol})")
    pct_bad_generic = physical_validity_check(gen_s0_db,  f"Generic ({args.pol})")
    print(f"  ISRO pipeline unrealistic pixels : {pct_bad_isro:.2f}%")
    print(f"  Generic pipeline unrealistic pixels: {pct_bad_generic:.2f}%")

    # ── noise floor ────────────────────────────────────────────────────────
    noise_floor_check(isro_s0_db, f"ISRO ({args.pol})")
    noise_floor_check(gen_s0_db,  f"Generic ({args.pol})")

    # ── incidence-angle dependency (key test) ──────────────────────────────
    incidence_angle_dependency(isro_s0_db, lia,
                                f"ISRO pipeline σ⁰ ({args.pol})")
    incidence_angle_dependency(gen_s0_db,  lia,
                                f"Generic 3-step σ⁰ ({args.pol})")

    # ── HH/HV ratio ────────────────────────────────────────────────────────
    if args.hv_sigma0_db:
        log.info("Loading HV Sigma0 for ratio check ...")
        hv_db = read_geotiff(args.hv_sigma0_db).array
        if mask is not None:
            hv_db = apply_valid_mask(hv_db, mask, valid_only=True)
        hh_hv_ratio_check(isro_s0_db, hv_db)

    # ── write difference GeoTIFF ───────────────────────────────────────────
    diff = isro_s0_db - gen_s0_db
    diff_path = os.path.join(args.output_dir,
                              f"{args.pol}_isro_minus_generic_dB.tif")
    write_geotiff(diff_path, diff.astype(np.float32), profile, nodata=np.nan)
    log.info("Difference GeoTIFF (ISRO - generic) written to: %s", diff_path)
    print(f"\nOpen {diff_path} in QGIS / SNAP to visualize the spatial "
          f"pattern of the calibration difference across the swath.\n"
          f"A 'gradient stripe' from near-range to far-range in this file\n"
          f"directly shows the incidence-angle error in the generic pipeline.\n")

    # ── summary verdict ────────────────────────────────────────────────────
    diff_vals = diff[np.isfinite(diff)]
    mean_bias = float(np.mean(diff_vals)) if diff_vals.size else float("nan")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mean calibration bias (ISRO - generic) : {mean_bias:+.3f} dB")
    if abs(mean_bias) > 0.5:
        print(f"  ✓ Significant systematic difference detected.")
        print(f"    The generic pipeline introduces a ~{abs(mean_bias):.1f} dB bias")
        print(f"    because it misidentifies Beta0-referenced EOS-04 DN as Sigma0.")
    else:
        print(f"  ~ Pipelines agree within 0.5 dB -- likely similar scene geometry.")
    if pct_bad_generic > pct_bad_isro:
        print(f"  ✓ ISRO pipeline has FEWER physically unrealistic pixels "
              f"({pct_bad_isro:.2f}% vs {pct_bad_generic:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())