"""
STEP 1: SAR DATA PREPROCESSING
================================
This script reads raw EOS-04 SAR TIF files (HH and HV),
applies radiometric calibration, converts to dB scale,
and saves the processed images as .npy files.

Folder structure expected:
ISRO_14/
└── raw/
    ├── E04_SAR_MRS_03JUN2026_.../
    │   ├── hh.tif    ← HH polarization image
    │   └── hv.tif    ← HV polarization image
    ├── E04_SAR_MRS_17MAY2026_.../
    │   ├── hh.tif
    │   └── hv.tif
    ... (all scene folders)
"""

import os
import numpy as np
import rasterio
import glob
import json
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — change BASE_DIR to your actual path
# ─────────────────────────────────────────────
BASE_DIR = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
RAW_DIR       = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Calibration constants from your BAND_META
# (These are default EOS-04 MRS L2B values — 
#  ideally parse from each scene's BAND_META file)
CALIB_CONST_HH   = 69.255
CALIB_CONST_HV   = 69.255   # usually same — check your HV BAND_META
NOISE_BIAS_HH    = 14546.164
NOISE_BIAS_HV    = 14546.164  # check your HV BAND_META


def parse_band_meta(meta_path):
    """
    Parse BAND_META text file to extract calibration constants.
    Returns dict with keys: calib_hh, calib_hv, noise_hh, noise_hv
    Falls back to hardcoded defaults if file not found.
    """
    defaults = {
        "calib_hh": CALIB_CONST_HH,
        "calib_hv": CALIB_CONST_HV,
        "noise_hh": NOISE_BIAS_HH,
        "noise_hv": NOISE_BIAS_HV,
    }
    if not os.path.exists(meta_path):
        print(f"  [WARN] BAND_META not found at {meta_path}, using defaults.")
        return defaults

    params = defaults.copy()
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if "Calibration_Constant_HH" in line:
                params["calib_hh"] = float(line.split("=")[-1].strip())
            elif "Calibration_Constant_HV" in line:
                params["calib_hv"] = float(line.split("=")[-1].strip())
            elif "Image_Noise_Bias_HH" in line:
                params["noise_hh"] = float(line.split("=")[-1].strip())
            elif "Image_Noise_Bias_HV" in line:
                params["noise_hv"] = float(line.split("=")[-1].strip())
    return params


def read_tif(tif_path):
    """
    Read a GeoTIFF file and return:
      - data: 2D numpy array of raw pixel values (float32)
      - profile: rasterio metadata (CRS, transform, etc.)
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)  # band 1 only
        profile = src.profile
    print(f"  Read TIF: {os.path.basename(tif_path)}, shape={data.shape}, dtype={data.dtype}")
    print(f"  Raw pixel range: min={data.min():.1f}, max={data.max():.1f}")
    return data, profile


def calibrate_to_db(raw_data, calib_const, noise_bias):
    """
    Convert raw 16-bit DN values → sigma-naught in dB scale.

    Formula (standard NRSC EOS-04 L2B calibration):
        sigma0_dB = 10 * log10(DN^2 + 1) - Calibration_Constant

    Output values typically range from:
      -25 dB  → very smooth surface (calm water, bare soil)
       +5 dB  → very rough surface (urban buildings)
    """
    dn = np.clip(raw_data, 0, None)

    sigma0_db = 10.0 * np.log10((dn ** 2) + 1.0) - calib_const

    print(f"  Calibrated dB range: min={sigma0_db.min():.2f}, max={sigma0_db.max():.2f}, "
          f"mean={sigma0_db.mean():.2f}")
    return sigma0_db


def find_tif_in_folder(folder_path):
    """
    Find the largest .tif file in a folder.
    (Avoids picking up small thumbnail TIFs)
    """
    tifs = glob.glob(os.path.join(folder_path, "*.tif"))
    tifs += glob.glob(os.path.join(folder_path, "*.TIF"))
    if not tifs:
        raise FileNotFoundError(f"No TIF files found in {folder_path}")
    # Pick the largest file = the full resolution image
    tifs_sorted = sorted(tifs, key=lambda x: os.path.getsize(x), reverse=True)
    return tifs_sorted[0]


def process_scene(scene_dir):
    """
    Process one scene folder (one date's acquisition).
    Returns: stacked numpy array of shape (H, W, 2) — [HH, HV]
    """
    scene_name = os.path.basename(scene_dir)
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")

    # Find BAND_META in scene root
    meta_path = os.path.join(scene_dir, "BAND_META")
    if not os.path.exists(meta_path):
        # Try .txt extension
        meta_path = os.path.join(scene_dir, "BAND_META.txt")
    params = parse_band_meta(meta_path)
    print(f"  Calibration constants → HH: {params['calib_hh']}, HV: {params['calib_hv']}")
    print(f"  Noise bias            → HH: {params['noise_hh']}, HV: {params['noise_hv']}")

    # Process HH channel — hh.tif sits directly in scene folder
    print(f"\n[HH Channel]")
    hh_tif = os.path.join(scene_dir, "hh.tif")
    if not os.path.exists(hh_tif):
        raise FileNotFoundError(f"hh.tif not found in {scene_dir}")
    hh_raw, profile = read_tif(hh_tif)
    hh_db = calibrate_to_db(hh_raw, params["calib_hh"], params["noise_hh"])

    # Process HV channel — hv.tif sits directly in scene folder
    print(f"\n[HV Channel]")
    hv_tif = os.path.join(scene_dir, "hv.tif")
    if not os.path.exists(hv_tif):
        raise FileNotFoundError(f"hv.tif not found in {scene_dir}")
    hv_raw, _ = read_tif(hv_tif)
    hv_db = calibrate_to_db(hv_raw, params["calib_hv"], params["noise_hv"])

    # Stack into (H, W, 2) array → channel 0 = HH, channel 1 = HV
    stacked = np.stack([hh_db, hv_db], axis=-1)  # shape: (H, W, 2)
    print(f"\n  Stacked shape: {stacked.shape}  (H x W x 2 channels)")

    return stacked, profile


def save_processed(stacked, scene_name, profile):
    """
    Save processed scene as:
    1. .npy file — fast loading for ML
    2. metadata .json — records shape, stats, CRS info
    """
    out_name = scene_name[:50]  # truncate very long folder names
    npy_path = os.path.join(PROCESSED_DIR, f"{out_name}.npy")
    np.save(npy_path, stacked)
    print(f"  Saved: {npy_path}")

    # Save metadata
    meta = {
        "scene": scene_name,
        "shape": list(stacked.shape),
        "channels": ["HH_dB", "HV_dB"],
        "HH_stats": {
            "min": float(stacked[:, :, 0].min()),
            "max": float(stacked[:, :, 0].max()),
            "mean": float(stacked[:, :, 0].mean()),
            "std":  float(stacked[:, :, 0].std()),
        },
        "HV_stats": {
            "min": float(stacked[:, :, 1].min()),
            "max": float(stacked[:, :, 1].max()),
            "mean": float(stacked[:, :, 1].mean()),
            "std":  float(stacked[:, :, 1].std()),
        },
        "crs": str(profile.get("crs", "unknown")),
    }
    json_path = os.path.join(PROCESSED_DIR, f"{out_name}_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {json_path}")
    return npy_path


def main():
    # Discover all scene folders inside raw/
    scene_dirs = sorted([
        d for d in glob.glob(os.path.join(RAW_DIR, "E04_SAR_*"))
        if os.path.isdir(d)
    ])

    if not scene_dirs:
        print(f"ERROR: No scene folders found in {RAW_DIR}")
        print("Expected folders starting with 'E04_SAR_...' inside raw/")
        return

    print(f"Found {len(scene_dirs)} scene(s):")
    for d in scene_dirs:
        print(f"  - {os.path.basename(d)}")

    processed_files = []
    for scene_dir in scene_dirs:
        try:
            stacked, profile = process_scene(scene_dir)
            npy_path = save_processed(stacked, os.path.basename(scene_dir), profile)
            processed_files.append(npy_path)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {scene_dir}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"DONE! Processed {len(processed_files)} scenes.")
    print(f"Files saved to: {PROCESSED_DIR}")
    print(f"\nNext step: Run step2_patch_extraction.py")


if __name__ == "__main__":
    main()