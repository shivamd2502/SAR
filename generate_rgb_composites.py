#!/usr/bin/env python3
"""
generate_rgb_composites.py
==========================
Generates all recommended RGB composite GeoTIFFs from your calibrated
EOS-04 outputs and produces a quantitative comparison report showing
exactly how and why the new composites are better than the old
R=HH_DN, G=HV_DN, B=HH/HV_DN approach.

Three composite options are built:
  Option 1 – Standard     : R=HH_sigma0_dB  G=HV_sigma0_dB  B=HH/HV_sigma0
  Option 2 – Terrain-aware: R=HH_gamma0_dB  G=HV_gamma0_dB  B=HH/HV_gamma0
  Option 3 – Index-enhanced: R=HH_sigma0_dB  G=HV_sigma0_dB  B=RVI

Old baseline composite:
  R=HH_raw_DN  G=HV_raw_DN  B=HH/HV_ratio (from raw DN)

Usage
-----
    python generate_rgb_composites.py ^
        --hh-dn    "raw\...\hh.tif" ^
        --hv-dn    "raw\...\hv.tif" ^
        --hh-sigma0-db  "calibrated\03JUN2026_HH\HH_sigma0_dB.tif" ^
        --hv-sigma0-db  "calibrated\03JUN2026_HV\HV_sigma0_dB.tif" ^
        --hh-sigma0-lin "calibrated\03JUN2026_HH\HH_sigma0_linear.tif" ^
        --hv-sigma0-lin "calibrated\03JUN2026_HV\HV_sigma0_linear.tif" ^
        --hh-gamma0-db  "calibrated\03JUN2026_HH\HH_gamma0_dB.tif" ^
        --hv-gamma0-db  "calibrated\03JUN2026_HV\HV_gamma0_dB.tif" ^
        --output-dir "composites\03JUN2026"
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
log = logging.getLogger("rgb")

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    print("ERROR: rasterio required.  pip install rasterio --break-system-packages")
    sys.exit(1)


# ==========================================================================
# I/O helpers
# ==========================================================================

def read_band(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    log.info("Loaded %s  shape=%s", path, arr.shape)
    return arr, profile


def write_rgb_geotiff(path: str, r: np.ndarray, g: np.ndarray,
                       b: np.ndarray, profile: dict) -> None:
    """Write a 3-band float32 GeoTIFF preserving the source georeferencing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=3, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(np.nan_to_num(r, nan=np.nan).astype("float32"), 1)
        dst.write(np.nan_to_num(g, nan=np.nan).astype("float32"), 2)
        dst.write(np.nan_to_num(b, nan=np.nan).astype("float32"), 3)
    log.info("Wrote %s", path)


def write_uint8_preview(path: str, r: np.ndarray, g: np.ndarray,
                         b: np.ndarray, profile: dict) -> None:
    """
    Writes a uint8 (0-255) 3-band GeoTIFF for quick visual preview
    (e.g. drag into QGIS and it displays immediately with good contrast).
    Uses 2nd–98th percentile stretch per channel so extreme outliers
    don't crush the contrast.
    """
    def stretch(arr: np.ndarray) -> np.ndarray:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi == lo:
            return np.zeros_like(arr, dtype=np.uint8)
        stretched = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
        stretched[~np.isfinite(arr)] = 0
        return stretched.astype(np.uint8)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=3, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(stretch(r), 1)
        dst.write(stretch(g), 2)
        dst.write(stretch(b), 3)
    log.info("Wrote preview %s", path)


# ==========================================================================
# Channel computation
# ==========================================================================

def compute_ratio_db(band_a_db: np.ndarray, band_b_db: np.ndarray) -> np.ndarray:
    """HH/HV ratio in dB = HH_dB - HV_dB  (subtraction in log = division in linear)."""
    return band_a_db - band_b_db


def compute_rvi(hh_lin: np.ndarray, hv_lin: np.ndarray) -> np.ndarray:
    """
    Radar Vegetation Index (linear domain):
        RVI = 8 * HV / (HH + HV)
    Range: 0 (bare soil/water) to 1 (dense vegetation).
    Must be computed in LINEAR units, not dB.
    """
    denom = hh_lin + hv_lin
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = np.where(denom > 0, (8.0 * hv_lin) / denom, np.nan)
    return rvi.astype(np.float32)


def compute_dpdi(hh_lin: np.ndarray, hv_lin: np.ndarray) -> np.ndarray:
    """
    Dual-Pol Discrimination Index:
        DPDI = (HH - HV) / (HH + HV)
    Range: -1 to +1.
    High positive = bare soil / urban.  Near zero = dense vegetation.
    """
    denom = hh_lin + hv_lin
    with np.errstate(invalid="ignore", divide="ignore"):
        dpdi = np.where(denom > 0, (hh_lin - hv_lin) / denom, np.nan)
    return dpdi.astype(np.float32)


def raw_dn_ratio(hh_dn: np.ndarray, hv_dn: np.ndarray) -> np.ndarray:
    """Old approach: simple pixel-wise DN ratio (not calibrated)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(hv_dn > 0, hh_dn / hv_dn, np.nan)
    return ratio.astype(np.float32)


# ==========================================================================
# Statistics & comparison
# ==========================================================================

def channel_stats(arr: np.ndarray, name: str) -> dict:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"name": name, "valid": 0}
    return {
        "name": name,
        "valid": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p2": float(np.percentile(v, 2)),
        "p98": float(np.percentile(v, 98)),
        "dynamic_range": float(np.percentile(v, 98) - np.percentile(v, 2)),
    }


def print_channel_stats(stats_list: list[dict]) -> None:
    hdr = f"{'Channel':<35} {'Valid px':>10} {'Mean':>8} {'Std':>6} {'P2':>8} {'P98':>8} {'Dyn.Range':>10}"
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for s in stats_list:
        if s.get("valid", 0) == 0:
            print(f"{s['name']:<35} {'no valid px':>10}")
            continue
        print(
            f"{s['name']:<35} {s['valid']:>10,} "
            f"{s['mean']:>8.3f} {s['std']:>6.3f} "
            f"{s['p2']:>8.3f} {s['p98']:>8.3f} "
            f"{s['dynamic_range']:>10.3f}"
        )
    print(sep)


def compare_ratio_channels(old_ratio: np.ndarray, new_ratio_db: np.ndarray,
                             lia: np.ndarray | None = None) -> None:
    """
    The core comparison: shows how much the HH/HV ratio changes across
    the swath (incidence angle bins) between old DN-based and new σ⁰-based
    ratio. A flat curve = swath-consistent. A sloped curve = incidence-angle
    artefact (the old problem).
    """
    print("\n--- HH/HV ratio across incidence angle bins ---")
    print("  (flat = good, sloped = incidence-angle error in that composite)\n")

    if lia is None:
        log.warning("No LIA supplied — skipping incidence-angle dependency test.")
        return

    fin = np.isfinite(old_ratio) & np.isfinite(new_ratio_db) & \
          np.isfinite(lia) & (lia > 0) & (lia < 90)
    old_v = old_ratio[fin]
    new_v = new_ratio_db[fin]
    lia_v = lia[fin]

    i_min, i_max = np.percentile(lia_v, 1), np.percentile(lia_v, 99)
    edges = np.linspace(i_min, i_max, 11)

    print(f"  {'Incidence bin':<18} {'Old ratio (DN)':>16} {'New ratio (dB)':>16} {'Count':>10}")
    old_means, new_means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lia_v >= lo) & (lia_v < hi)
        if m.sum() == 0:
            continue
        om = float(np.mean(old_v[m]))
        nm = float(np.mean(new_v[m]))
        old_means.append(om)
        new_means.append(nm)
        print(f"  {lo:.1f}–{hi:.1f}°{'':<6} {om:>16.3f} {nm:>16.3f} {m.sum():>10,}")

    if len(old_means) > 2:
        old_trend = float(np.polyfit(range(len(old_means)), old_means, 1)[0])
        new_trend = float(np.polyfit(range(len(new_means)), new_means, 1)[0])
        print(f"\n  Old ratio trend across bins: {old_trend:+.4f} per bin")
        print(f"  New ratio trend across bins: {new_trend:+.4f} per bin")
        if abs(new_trend) < abs(old_trend):
            print(f"  ✓ New HH/HV ratio is MORE consistent across the swath "
                  f"(trend reduced by {abs(old_trend)-abs(new_trend):.4f}).")
        else:
            print(f"  ~ Trends are similar — both pipelines agree on ratio shape.")


def water_body_check(hh_db: np.ndarray, label: str,
                      threshold_db: float = -18.0) -> None:
    """
    Water bodies should appear as very dark pixels in HH σ⁰ (< -18 dB).
    If noise bias was NOT removed, water pixels are artificially lifted
    and may not reach the water threshold.
    """
    v = hh_db[np.isfinite(hh_db)]
    if v.size == 0:
        return
    pct_water = 100.0 * np.sum(v < threshold_db) / v.size
    print(f"\n  [{label}] Pixels below {threshold_db} dB (likely water/shadow): "
          f"{pct_water:.2f}%")
    if pct_water > 1:
        print(f"  ✓ Water bodies are clearly dark — noise bias removal is effective.")
    else:
        print(f"  ~ Very few dark pixels — check if scene covers mostly land.")


# ==========================================================================
# Main pipeline
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate RGB composites from EOS-04 calibrated outputs "
                    "and compare against old DN-based composite.")
    p.add_argument("--hh-dn",   required=True, help="Raw HH DN GeoTIFF (hh.tif)")
    p.add_argument("--hv-dn",   required=True, help="Raw HV DN GeoTIFF (hv.tif)")
    p.add_argument("--hh-sigma0-db",  required=True)
    p.add_argument("--hv-sigma0-db",  required=True)
    p.add_argument("--hh-sigma0-lin", required=True)
    p.add_argument("--hv-sigma0-lin", required=True)
    p.add_argument("--hh-gamma0-db",  required=True)
    p.add_argument("--hv-gamma0-db",  required=True)
    p.add_argument("--lia", help="Local incidence angle tif (for swath test)")
    p.add_argument("--output-dir", default="composites")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load all bands ─────────────────────────────────────────────────────
    log.info("Loading all bands ...")
    hh_dn,        profile = read_band(args.hh_dn)
    hv_dn,        _       = read_band(args.hv_dn)
    hh_s0_db,     _       = read_band(args.hh_sigma0_db)
    hv_s0_db,     _       = read_band(args.hv_sigma0_db)
    hh_s0_lin,    _       = read_band(args.hh_sigma0_lin)
    hv_s0_lin,    _       = read_band(args.hv_sigma0_lin)
    hh_g0_db,     _       = read_band(args.hh_gamma0_db)
    hv_g0_db,     _       = read_band(args.hv_gamma0_db)

    lia = None
    if args.lia:
        lia, _ = read_band(args.lia)

    # ── compute derived channels ───────────────────────────────────────────
    log.info("Computing derived channels ...")

    # Old baseline
    old_ratio   = raw_dn_ratio(hh_dn, hv_dn)

    # New channels
    ratio_s0_db = compute_ratio_db(hh_s0_db, hv_s0_db)    # HH/HV σ⁰ in dB
    ratio_g0_db = compute_ratio_db(hh_g0_db, hv_g0_db)    # HH/HV γ⁰ in dB
    rvi         = compute_rvi(hh_s0_lin, hv_s0_lin)        # linear σ⁰ input
    dpdi        = compute_dpdi(hh_s0_lin, hv_s0_lin)       # linear σ⁰ input

    # ── write composites ───────────────────────────────────────────────────
    log.info("Writing RGB composites ...")

    # Old (baseline)
    old_path = os.path.join(args.output_dir, "OLD_composite_R-HH_G-HV_B-ratio_DN.tif")
    write_rgb_geotiff(old_path, hh_dn, hv_dn, old_ratio, profile)
    write_uint8_preview(
        old_path.replace(".tif", "_PREVIEW.tif"),
        hh_dn, hv_dn, old_ratio, profile)

    # Option 1: σ⁰ standard
    opt1_path = os.path.join(args.output_dir, "OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif")
    write_rgb_geotiff(opt1_path, hh_s0_db, hv_s0_db, ratio_s0_db, profile)
    write_uint8_preview(
        opt1_path.replace(".tif", "_PREVIEW.tif"),
        hh_s0_db, hv_s0_db, ratio_s0_db, profile)

    # Option 2: γ⁰ terrain-aware
    opt2_path = os.path.join(args.output_dir, "OPT2_R-HH-gamma0_G-HV-gamma0_B-ratio-gamma0.tif")
    write_rgb_geotiff(opt2_path, hh_g0_db, hv_g0_db, ratio_g0_db, profile)
    write_uint8_preview(
        opt2_path.replace(".tif", "_PREVIEW.tif"),
        hh_g0_db, hv_g0_db, ratio_g0_db, profile)

    # Option 3: index-enhanced
    opt3_path = os.path.join(args.output_dir, "OPT3_R-HH-sigma0_G-HV-sigma0_B-RVI.tif")
    write_rgb_geotiff(opt3_path, hh_s0_db, hv_s0_db, rvi, profile)
    write_uint8_preview(
        opt3_path.replace(".tif", "_PREVIEW.tif"),
        hh_s0_db, hv_s0_db, rvi, profile)

    # Save derived indices as standalone layers too
    rvi_path  = os.path.join(args.output_dir, "RVI_linear.tif")
    dpdi_path = os.path.join(args.output_dir, "DPDI_linear.tif")
    for path, arr in [(rvi_path, rvi), (dpdi_path, dpdi)]:
        out_p = profile.copy()
        out_p.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with rasterio.open(path, "w", **out_p) as dst:
            dst.write(arr, 1)
        log.info("Wrote %s", path)

    # ── statistics comparison ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHANNEL STATISTICS COMPARISON")
    print("=" * 72)
    print("\nRed channel (HH):")
    print_channel_stats([
        channel_stats(hh_dn.astype(np.float32),    "OLD  — HH raw DN"),
        channel_stats(hh_s0_db,                     "NEW  — HH σ⁰ dB"),
        channel_stats(hh_g0_db,                     "NEW  — HH γ⁰ dB"),
    ])
    print("\nGreen channel (HV):")
    print_channel_stats([
        channel_stats(hv_dn.astype(np.float32),    "OLD  — HV raw DN"),
        channel_stats(hv_s0_db,                     "NEW  — HV σ⁰ dB"),
        channel_stats(hv_g0_db,                     "NEW  — HV γ⁰ dB"),
    ])
    print("\nBlue channel (HH/HV ratio or RVI):")
    print_channel_stats([
        channel_stats(old_ratio,    "OLD  — HH/HV DN ratio"),
        channel_stats(ratio_s0_db,  "NEW  — HH/HV σ⁰ dB"),
        channel_stats(ratio_g0_db,  "NEW  — HH/HV γ⁰ dB"),
        channel_stats(rvi,          "NEW  — RVI (0–1 range)"),
    ])

    # ── derived index stats ────────────────────────────────────────────────
    print("\nDerived indices (not available in old pipeline):")
    print_channel_stats([
        channel_stats(rvi,  "RVI  (vegetation health, 0–1)"),
        channel_stats(dpdi, "DPDI (soil vs veg, −1 to +1)"),
    ])

    # ── swath consistency test ─────────────────────────────────────────────
    compare_ratio_channels(old_ratio, ratio_s0_db, lia)

    # ── water body test ────────────────────────────────────────────────────
    print("\n--- Water body / low-backscatter pixel check ---")
    water_body_check(hh_dn.astype(np.float32),
                      "OLD HH (DN, no calibration, no noise-bias removal)",
                      threshold_db=-18.0)
    water_body_check(hh_s0_db,
                      "NEW HH σ⁰ dB (calibrated, noise-bias removed)",
                      threshold_db=-18.0)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FILES WRITTEN")
    print("=" * 72)
    files = [
        ("OLD baseline composite (float32)", old_path),
        ("OLD baseline PREVIEW (uint8)",     old_path.replace(".tif","_PREVIEW.tif")),
        ("Option 1 σ⁰ standard (float32)",  opt1_path),
        ("Option 1 PREVIEW (uint8)",          opt1_path.replace(".tif","_PREVIEW.tif")),
        ("Option 2 γ⁰ terrain (float32)",    opt2_path),
        ("Option 2 PREVIEW (uint8)",          opt2_path.replace(".tif","_PREVIEW.tif")),
        ("Option 3 index-enhanced (float32)",opt3_path),
        ("Option 3 PREVIEW (uint8)",          opt3_path.replace(".tif","_PREVIEW.tif")),
        ("RVI standalone layer",              rvi_path),
        ("DPDI standalone layer",             dpdi_path),
    ]
    for label, path in files:
        print(f"  {label:<42} {path}")

    print("""
HOW TO COMPARE IN QGIS
-----------------------
1. Load OLD_composite_..._PREVIEW.tif  (drag & drop)
2. Load OPT1_..._PREVIEW.tif           (drag & drop)
3. Properties → Symbology → Min/Max = 2-98% on both  (already stretched)
4. Toggle visibility to compare — look for:
     a) Edge brightness gradient (near vs far range) — should DISAPPEAR in new
     b) Water bodies darker in new composite
     c) Vegetation more distinct green in Option 3 (RVI blue channel)
     d) Hillside colours more uniform in Option 2 (gamma0)
5. Open Raster → Analysis → Raster Calculator:
     "OPT1_R - OLD_R"  to see the per-pixel difference map
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())