"""
GENERATE QGIS SHAPEFILE OF PATCH BOUNDARIES + LABELS
=======================================================
Converts your patch grid (row_start, col_start in pixel space)
into real-world geographic polygons using the original GeoTIFF's
CRS and transform, then writes a shapefile you can open directly
in QGIS next to a Google Satellite / ESRI Satellite basemap.

This lets you visually verify: does the patch labeled "Urban Dense"
actually sit on top of buildings/roads in the satellite image?

Input:
    ISRO_14/raw/<scene_folder>/hh.tif          <- for CRS + geotransform
    ISRO_14/patches/patch_index.json            <- row_start/col_start per patch
    ISRO_14/Labels/*.csv                         <- your classification labels
                                                     (optional — script runs without
                                                      labels too, just shows "Unlabeled")

Output:
    ISRO_14/qgis/patches_<scene>.shp  (+ .shx .dbf .prj)
        One polygon per patch, with attributes:
            patch_id, scene, row_start, col_start, label, lon_center, lat_center

Usage:
    python generate_patch_shapefile.py
    python generate_patch_shapefile.py --scene E04_SAR_MRS_03JUN2026_154005259612_23736_STUC00ZTD_32242_8_DH_D_R_N31497_E076829
"""

import os
import sys
import glob
import json
import argparse
import pandas as pd
import numpy as np

try:
    import rasterio
    from rasterio.transform import xy
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except ImportError:
    print("ERROR: geopandas/shapely not installed.")
    print("Run: pip install geopandas shapely fiona pyproj")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
RAW_DIR      = os.path.join(BASE_DIR, "raw")
PATCHES_DIR  = os.path.join(BASE_DIR, "patches")
LABELS_DIR   = os.path.join(BASE_DIR, "Labels")
QGIS_DIR     = os.path.join(BASE_DIR, "qgis")
os.makedirs(QGIS_DIR, exist_ok=True)

PATCH_SIZE = 256


# ─────────────────────────────────────────────
# LOAD PATCH INDEX
# ─────────────────────────────────────────────
def load_patch_index(scene_filter=None):
    index_path = os.path.join(PATCHES_DIR, "patch_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"patch_index.json not found at {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    scenes_available = sorted(set(p["scene"] for p in index))
    print("Scenes found in patch_index.json:")
    for s in scenes_available:
        n = sum(1 for p in index if p["scene"] == s)
        print(f"  {s}  ({n} patches)")

    if scene_filter is None:
        print(f"\nNo --scene given, generating shapefiles for ALL scenes.")
        return index, scenes_available

    matched = [p for p in index if p["scene"] == scene_filter]
    if not matched:
        raise ValueError(f"No patches found for scene '{scene_filter}'. "
                          f"Available: {scenes_available}")
    return matched, [scene_filter]


# ─────────────────────────────────────────────
# FIND THE ORIGINAL TIF FOR A SCENE (for CRS + transform)
# ─────────────────────────────────────────────
def find_scene_tif(scene_name):
    """
    The 'scene' field in patch_index.json matches the .npy filename stem,
    which was truncated from the original raw folder name (first 50 chars).
    We match by prefix against folders under raw/.
    """
    candidates = glob.glob(os.path.join(RAW_DIR, scene_name + "*"))
    if not candidates:
        # Try matching just by the truncated prefix used in step1 (scene_name[:50])
        all_folders = [d for d in glob.glob(os.path.join(RAW_DIR, "*")) if os.path.isdir(d)]
        candidates = [d for d in all_folders if os.path.basename(d).startswith(scene_name[:40])]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find raw scene folder for '{scene_name}' under {RAW_DIR}"
        )

    scene_folder = candidates[0]
    hh_path = os.path.join(scene_folder, "hh.tif")
    if not os.path.exists(hh_path):
        raise FileNotFoundError(f"hh.tif not found in {scene_folder}")

    return hh_path


# ─────────────────────────────────────────────
# LOAD LABELS (optional — merges classification CSVs)
# ─────────────────────────────────────────────
def load_labels_lookup():
    """
    Returns dict: { 'patch_00123.png': 'Urban Dense', ... }
    Returns empty dict if no label CSVs found (patches will show 'Unlabeled').
    """
    csv_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.csv")))
    if not csv_files:
        print("\n[INFO] No label CSVs found — patches will be tagged 'Unlabeled'.")
        return {}

    lookup = {}
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()

        img_col = None
        for col in df.columns:
            if col.lower().strip() == "image_name":
                img_col = col
                break
        if img_col is None:
            for col in df.columns:
                if "name" in col.lower() and "unnamed" not in col.lower():
                    img_col = col
                    break
        if img_col is None or "class" not in df.columns:
            continue

        df["class"] = df["class"].fillna("None").astype(str).str.strip()
        df[img_col] = df[img_col].fillna("").astype(str).str.strip()

        for _, row in df.iterrows():
            name = row[img_col]
            if name:
                lookup[name] = row["class"]

    print(f"\nLoaded {len(lookup)} labels from {len(csv_files)} CSV file(s).")
    return lookup


def normalize_class(c):
    cl = c.lower()
    if "urban" in cl or "dense" in cl:
        return "Urban Dense"
    elif "hill" in cl or "rural" in cl or "sparse" in cl:
        return "Hills Rural"
    else:
        return "None"


# ─────────────────────────────────────────────
# BUILD POLYGONS FOR ONE SCENE
# ─────────────────────────────────────────────
def build_scene_shapefile(scene_name, patches_meta, label_lookup):
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")

    tif_path = find_scene_tif(scene_name)
    print(f"Using GeoTIFF for CRS/transform: {tif_path}")

    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs
        print(f"CRS: {crs}")

        records = []
        for p in patches_meta:
            row, col = p["row_start"], p["col_start"]
            patch_id = p["patch_id"]

            # Pixel corners of this patch (row, col) in image space
            r0, c0 = row, col
            r1, c1 = row + PATCH_SIZE, col + PATCH_SIZE

            # Convert all 4 corners to map coordinates (x=lon/easting, y=lat/northing)
            # rasterio xy(row, col) gives the CENTER of a pixel by default
            x_ul, y_ul = transform * (c0, r0)
            x_ur, y_ur = transform * (c1, r0)
            x_lr, y_lr = transform * (c1, r1)
            x_ll, y_ll = transform * (c0, r1)

            poly = Polygon([(x_ul, y_ul), (x_ur, y_ur), (x_lr, y_lr), (x_ll, y_ll)])

            # Center point for reference / labeling
            x_center, y_center = transform * (c0 + PATCH_SIZE / 2, r0 + PATCH_SIZE / 2)

            img_name = f"{patch_id}.png"
            raw_label = label_lookup.get(img_name, None)
            label = normalize_class(raw_label) if raw_label is not None else "Unlabeled"

            records.append({
                "patch_id": patch_id,
                "scene": scene_name[:40],   # shapefile field name/length limits
                "row_start": row,
                "col_start": col,
                "label": label,
                "x_center": x_center,
                "y_center": y_center,
                "geometry": poly,
            })

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Reproject to WGS84 (lat/lon) so it overlays correctly with
    # Google Satellite / OSM basemaps in QGIS (which use EPSG:4326 / 3857)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    safe_scene_name = "".join(c if c.isalnum() else "_" for c in scene_name[:40])
    out_path = os.path.join(QGIS_DIR, f"patches_{safe_scene_name}.shp")
    gdf_wgs84.to_file(out_path)

    print(f"\nSaved shapefile: {out_path}")
    print(f"Total patches: {len(gdf_wgs84)}")
    print(f"\nLabel distribution:")
    print(gdf_wgs84["label"].value_counts().to_string())

    return out_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None,
                         help="Specific scene name to process. "
                              "If omitted, processes all scenes found.")
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATE QGIS PATCH SHAPEFILE")
    print("=" * 60)

    label_lookup = load_labels_lookup()

    all_patches, scene_names = load_patch_index(args.scene)

    if args.scene:
        scenes_to_process = {args.scene: all_patches}
    else:
        scenes_to_process = {}
        for s in scene_names:
            scenes_to_process[s] = [p for p in all_patches if p["scene"] == s]

    out_paths = []
    for scene_name, patches_meta in scenes_to_process.items():
        try:
            out_path = build_scene_shapefile(scene_name, patches_meta, label_lookup)
            out_paths.append(out_path)
        except Exception as e:
            print(f"\n[ERROR] Failed on scene '{scene_name}': {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Shapefiles saved in: {QGIS_DIR}")
    for p in out_paths:
        print(f"  {p}")
    print("\nNext step: open these .shp files directly in QGIS")
    print("(drag-and-drop the .shp file — the .shx/.dbf/.prj load automatically)")
    print("=" * 60)


if __name__ == "__main__":
    main()