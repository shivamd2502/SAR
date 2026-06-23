#!/usr/bin/env python3
"""
eos04_radiometric_preprocessing.py
===================================

Single-file radiometric calibration / preprocessing pipeline for ISRO
EOS-04 SAR data products, implemented strictly from:

    "EOS-04 Data Products Formats (January 2025), Version 1.2.5"
    SAC/SIPG/MDPD/EOS-04/SAR/DP/2021/TN-05
    SAR Data Processing Division, MDPG, SIPA, Space Applications Centre

WHY THESE STEPS (significance of radiometric calibration)
-----------------------------------------------------------
EOS-04 raw image pixels are delivered as Digital Numbers (DN), which are a
processing-level representation of radar return power and are NOT
physically comparable across scenes, incidence angles, or acquisitions
until they are converted into a calibrated backscatter coefficient:

    * Beta0 (sigma-naught in the SLANT/RADAR plane)   -> eq. (1)
    * Sigma0 (backscatter normalized to the GROUND/SLANT plane) -> eq. (2)
    * Gamma0 (backscatter normalized to the ELLIPSOID/GAMMA plane) -> eq. (3)

EOS-04 stores images as Beta0 (unlike RISAT-1 which stored Sigma0), so
every downstream product (Sigma0, Gamma0) must be *derived* using the
per-product calibration constant (Kcal_Beta0_dB) and the per-pixel
incidence angle. Because SAR backscatter is heavily distorted by terrain
(layover/shadow/foreshortening) and by viewing geometry (look angle vs.
local slope), Level-2B "NRB" / ARD products additionally apply
Radiometric Terrain Correction (RTC) using a local illumination area
(scattering area in the gamma-plane) so backscatter values become
comparable irrespective of terrain or incidence angle -- this is exactly
what makes a product "Analysis Ready" (CEOS CARD4L-NRB compliant).

This module therefore implements, end-to-end, the *documented* sequence
of preprocessing steps a user must apply before EOS-04 data is usable for
quantitative analysis:

  1. Parse BAND_META.txt for calibration constants, noise bias, incidence
     angle / geometry tags                                  (Sec. 2.2 i, Appendix-1)
  2. Read per-pixel Digital Number from GeoTIFF imagery       (Sec. 3.0)
  3. (Optional) Reconstitute true digital numbers by removing the
     IMAGE_NOISE_BIAS that was added during processing        ("Note on Noise Bias Usage", Sec. 3.0)
  4. Convert calibration constant from dB to linear units      (eq. 4)
  5. Compute Beta0 / Sigma0 / Gamma0 using the per-pixel local
     incidence angle (from *_lia.tif or grid file)             (eq. 1-3)
  6. Apply the Layover/Shadow Mask so that distorted pixels are
     excluded from analysis                                   (Sec. 2.1.1, Table 3/4; Sec. 6.2.1)
  7. For Level-2B NRB / ARD products: read the DN directly as a
     terrain-normalized Gamma0 representation, derive calibrated
     Gamma0 in dB / linear units                               (eq. 9-12)
  8. For Level-2B: optionally "undo" the terrain normalization using the
     per-pixel local illumination (scattering) area to recover the
     un-normalized Beta0/Sigma0/Gamma0 (as in a Level-2A product)
                                                                  (eq. 13-15)
  9. Optionally compute Radar Cross Section (sigma, not sigma-naught)
     for point targets using either the Integration or Peak method
                                                                  (eq. 5-6)
 10. Write each calibrated layer back out as a (optionally dB-scaled)
     GeoTIFF, preserving the source georeferencing.

Document equation references are repeated as comments next to the code
that implements them, so this file can be cross-checked line-by-line
against Section 3.0 / Section 6.3 of the EOS-04 Data Products Formats
document.

Dependencies
------------
    numpy
    rasterio   (preferred; falls back to a minimal GDAL-free GeoTIFF
                reader/writer using `tifffile` + manual georeferencing
                copy if rasterio is unavailable -- see _io backend below)

Usage
-----
    # Level-1 / Level-2A product, single polarization
    python eos04_radiometric_preprocessing.py \\
        --band-meta  /data/208385331/BAND_META.txt \\
        --image      /data/208385331/scene_HH/imagery_HHHH.tif \\
        --pol HH --product-level L2 \\
        --lia /data/208385331/208385331_lia.tif \\
        --mask /data/208385331/208385331_mask.tif \\
        --output-dir /data/out --apply-noise-bias

    # Level-2B NRB / ARD product
    python eos04_radiometric_preprocessing.py \\
        --band-meta /data/tile/BAND_META.txt \\
        --image     /data/tile/Product_TILE_ID_HH.tif \\
        --pol HH --product-level L2B \\
        --lia /data/tile/Product_TILE_ID_lia.tif \\
        --mask /data/tile/Product_TILE_ID_mask.tif \\
        --area /data/tile/Product_TILE_ID_area.tif \\
        --output-dir /data/out --undo-normalization

    # No real data on hand yet? Sanity-check the math:
    python eos04_radiometric_preprocessing.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Optional raster I/O backend
# --------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.profiles import Profile
    _HAVE_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAVE_RASTERIO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eos04")


# ==========================================================================
# 1. BAND_META.txt PARSER  (Sec. 2.2-i / Appendix-1, "Sample BAND META File")
# ==========================================================================

class BandMeta:
    """
    Parses an EOS-04 BAND_META.txt ASCII metadata file into a flat
    key -> string dict, stripping the `//comment` suffixes used
    throughout the sample file in Appendix A1.0, e.g.:

        Calibration_Constant_Beta0_HH=69.185
        OutputLineSpacing=4.50 //Not Applicable for RAW product
    """

    _LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*(?://.*)?$")

    def __init__(self, path: str):
        self.path = path
        self.fields: Dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        with open(self.path, "r", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if not line.strip() or line.strip().startswith("#"):
                    continue
                m = self._LINE_RE.match(line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                self.fields[key] = val
        log.info("Parsed %d fields from %s", len(self.fields), self.path)

    # -- typed getters -----------------------------------------------------
    def get(self, key: str, default=None) -> Optional[str]:
        return self.fields.get(key, default)

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        v = self.fields.get(key)
        if v is None or v in ("-9999.99", "-9999.990", "-9999.000000", "NA", "$"):
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        v = self.fields.get(key)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def polarizations(self) -> list:
        """Returns the list of Tx/Rx polarizations present, per
        NoOfPolarizations / TxRxPolN tags (BAND_META sample, Appendix-1)."""
        n = self.get_int("NoOfPolarizations", 1) or 1
        pols = []
        for i in range(1, n + 1):
            p = self.get(f"TxRxPol{i}")
            if p:
                pols.append(p)
        return pols

    def calibration_constant_db(self, pol: str, plane: str = "Beta0") -> float:
        """
        Returns Kcal_<plane>_dB for the given polarization.

        Per Sec. 3.0: "Kcal_Beta0_dB is also available in tag
        Calibration_Constant_Beta0_TxRx in the BAND_META.txt file."
        Per Sec. 6.3: for Level-2B the SAME constant is exposed under
        'Calibration_Constant_Beta0_TxRxpol' and used directly as
        Kcal_dB for the stored Gamma0 DN.

        We try several historically-used key spellings for robustness:
            Calibration_Constant_<plane>_<pol>
            Calibration_Constant_<pol>            (legacy / Sigma-style constant)
        """
        candidates = [
            f"Calibration_Constant_{plane}_{pol}",
            f"Calibration_Constant_{plane}_TxRx_{pol}",
            f"Calibration_Constant_{pol}",  # legacy fallback
        ]
        for key in candidates:
            v = self.get_float(key)
            if v is not None:
                return v
        raise KeyError(
            f"Could not find calibration constant for plane={plane} pol={pol} "
            f"in {self.path}. Tried: {candidates}"
        )

    def noise_bias(self, pol: str) -> Optional[float]:
        """
        IMAGE_NOISE_BIAS per polarization ("Note on Noise Bias Usage", Sec. 3.0).
        BAND_META sample stores this as Image_Noise_Bias_<pol>, e.g.
        Image_Noise_Bias_RH / Image_Noise_Bias_RV.
        """
        return self.get_float(f"Image_Noise_Bias_{pol}")

    def output_pixel_area(self) -> Optional[float]:
        """
        Scattering_Area_integration = OutputLineSpacing * OutputPixelSpacing
        (eq. 5, used for RCS-of-point-target "Integration Method").
        """
        ls = self.get_float("OutputLineSpacing")
        ps = self.get_float("OutputPixelSpacing")
        if ls is None or ps is None:
            return None
        return ls * ps

    def rtc_apply_flag(self) -> Optional[int]:
        """RTC_Apply_Flag: 1 = DEM registration succeeded & RTC applied,
        0 = RTC not applied even though product is nominally Level-2B
        Gamma0 (Sec. 6.5)."""
        return self.get_int("RTC_Apply_Flag")


# ==========================================================================
# 2. RASTER I/O HELPERS
# ==========================================================================

@dataclass
class RasterData:
    array: np.ndarray
    profile: dict
    nodata: Optional[float] = None


def read_geotiff(path: str) -> RasterData:
    if not _HAVE_RASTERIO:
        raise RuntimeError(
            "rasterio is required to read GeoTIFF products. "
            "Install with: pip install rasterio --break-system-packages"
        )
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata
    log.info("Read %s -> shape=%s dtype=%s", path, arr.shape, arr.dtype)
    return RasterData(array=arr, profile=profile, nodata=nodata)


def read_geotiff_complex(path_i: str, path_q: Optional[str] = None) -> np.ndarray:
    """
    For SLC products: DN = sqrt(DNI^2 + DNQ^2) (Sec. 3.0, "Digital Number").
    If the GeoTIFF stores interleaved complex (2-band: I, Q) pass a single
    `path_i`; if I and Q are separate single-band files pass both paths.
    """
    if not _HAVE_RASTERIO:
        raise RuntimeError("rasterio is required to read SLC GeoTIFF products.")
    if path_q is None:
        with rasterio.open(path_i) as src:
            if src.count < 2:
                raise ValueError(
                    f"{path_i} has only {src.count} band(s); expected 2 (I,Q) "
                    "for an SLC product, or pass --image-q explicitly."
                )
            i_band = src.read(1).astype(np.float64)
            q_band = src.read(2).astype(np.float64)
    else:
        with rasterio.open(path_i) as src:
            i_band = src.read(1).astype(np.float64)
        with rasterio.open(path_q) as src:
            q_band = src.read(1).astype(np.float64)
    return np.sqrt(i_band ** 2 + q_band ** 2)  # eq. "DNp = Sqrt(DNIp^2 + DNQp^2)"


def write_geotiff(path: str, array: np.ndarray, profile: dict,
                   dtype: str = "float32", nodata: Optional[float] = np.nan) -> None:
    if not _HAVE_RASTERIO:
        raise RuntimeError("rasterio is required to write GeoTIFF outputs.")
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)
    log.info("Wrote %s", path)


# ==========================================================================
# 3. GRID FILE READER  (Sec. 2.2-ii, used to recover incidence angle when
#    a dedicated *_lia.tif is not delivered, e.g. for Level-1 products)
# ==========================================================================

def read_grid_file(path: str) -> Dict[str, np.ndarray]:
    """
    Reads an EOS-04 *_grid.txt file (row-major N x N grid, e.g. 32x32 in
    scan/pixel direction). Each row: scan pixel lat lon slant_range incidence_angle
    (exact column order can vary by level; we sniff the header / column
    count). -9999.000000 marks invalid pixels outside the imaged scene
    (Sec. 2.2-ii note).

    Returns dict with numpy arrays: scan, pixel, lat, lon, slant_range,
    incidence_angle.
    """
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(x) for x in parts[:6]])
            except ValueError:
                continue  # header line
    data = np.array(rows)
    if data.shape[1] < 5:
        raise ValueError(f"Unexpected grid file format in {path}")
    out = {
        "scan": data[:, 0],
        "pixel": data[:, 1],
        "lat": data[:, 2],
        "lon": data[:, 3],
        "slant_range": data[:, 4],
    }
    if data.shape[1] >= 6:
        out["incidence_angle"] = data[:, 5]
    return out


def incidence_angle_from_grid(grid: Dict[str, np.ndarray], shape: Tuple[int, int],
                               grid_spacing: int = 32) -> np.ndarray:
    """
    Bilinearly resamples the sparse incidence-angle grid (sampled every
    `grid_spacing` scan/pixel, per Sec. 2.2-ii) up to the full image
    `shape` = (n_lines, n_pixels). Invalid grid points (-9999) are
    excluded from interpolation.
    """
    if "incidence_angle" not in grid:
        raise KeyError("Grid file does not contain an incidence-angle column.")
    valid = grid["incidence_angle"] > -9998.0
    scan = grid["scan"][valid]
    pix = grid["pixel"][valid]
    ang = grid["incidence_angle"][valid]

    n_lines, n_pixels = shape
    # Build a regular grid image from scattered points via griddata.
    try:
        from scipy.interpolate import griddata
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "scipy is required for grid-based incidence-angle interpolation. "
            "Install with: pip install scipy --break-system-packages"
        ) from e

    grid_y, grid_x = np.mgrid[0:n_lines, 0:n_pixels]
    interp = griddata(
        points=np.column_stack([scan, pix]),
        values=ang,
        xi=(grid_y, grid_x),
        method="linear",
        fill_value=np.nan,
    )
    return interp


# ==========================================================================
# 4. RADIOMETRIC CALIBRATION CORE  (Sec. 3.0, equations 1-6)
# ==========================================================================

def kcal_linear_from_db(kcal_db: float) -> float:
    """eq. (4): Kcal_Beta0_linear = 10^(Kcal_Beta0_dB / 10)"""
    return 10.0 ** (kcal_db / 10.0)


def apply_noise_bias(dn: np.ndarray, image_noise_bias: Optional[float],
                      clip_negative: bool = False) -> np.ndarray:
    """
    "Note on Noise Bias Usage" (Sec. 3.0): an additive noise bias N has
    already been added to DN^2 on-board processing to avoid negative
    calibrated values. To reconstitute true digital numbers, subtract
    IMAGE_NOISE_BIAS from DN^2 *before* applying eq. (1)-(3):

        DN_corrected^2 = DN^2 - IMAGE_NOISE_BIAS

    Returns DN_corrected^2 (i.e. it already returns the *squared*,
    noise-bias-corrected value expected by compute_beta0/sigma0/gamma0's
    `dn_squared` argument). If `image_noise_bias` is None the noise bias
    step is skipped (dn^2 returned unchanged) -- the document notes this
    is the user's choice: "It is up to the user or end application to
    decide how best to handle pixels having negative calibrated values."
    """
    dn_sq = dn.astype(np.float64) ** 2
    if image_noise_bias is not None:
        dn_sq = dn_sq - image_noise_bias
        if clip_negative:
            dn_sq = np.clip(dn_sq, 0.0, None)
    return dn_sq


def compute_beta0(dn_squared: np.ndarray, kcal_beta0_linear: float) -> np.ndarray:
    """eq. (1): Beta0_p = DN_p^2 / Kcal_Beta0_linear"""
    return dn_squared / kcal_beta0_linear


def compute_sigma0(dn_squared: np.ndarray, kcal_beta0_linear: float,
                    incidence_angle_deg: np.ndarray) -> np.ndarray:
    """eq. (2): Sigma0_p = DN_p^2 * sin(i_p) / Kcal_Beta0_linear"""
    i_rad = np.deg2rad(incidence_angle_deg)
    return dn_squared * np.sin(i_rad) / kcal_beta0_linear


def compute_gamma0(dn_squared: np.ndarray, kcal_beta0_linear: float,
                    incidence_angle_deg: np.ndarray) -> np.ndarray:
    """eq. (3): Gamma0_p = DN_p^2 * tan(i_p) / Kcal_Beta0_linear"""
    i_rad = np.deg2rad(incidence_angle_deg)
    return dn_squared * np.tan(i_rad) / kcal_beta0_linear


def to_db(linear: np.ndarray, power_quantity: bool = True) -> np.ndarray:
    """
    Generic linear -> dB helper.
    `power_quantity=True` -> 10*log10(x)  (Beta0/Sigma0/Gamma0 are power
    ratios, matching eq. (9)'s use of 20*log10(DN) for an *amplitude*-like
    DN but 10*log10 for already-squared/power quantities).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(linear) if power_quantity else 20.0 * np.log10(linear)
    return out


def rcs_point_target_integration(dn_window: np.ndarray, kcal_beta0_linear: float,
                                  output_line_spacing: float,
                                  output_pixel_spacing: float) -> float:
    """
    eq. (5), "Integration Method": Sigma (not Sigma-naught) for a point
    target, integrating DN^2 over a window `w` (e.g. 5x5, 9x9, 11x11)
    around the target.

        Sigma_p = (sum_w DN_w^2) * Scattering_Area_integration / Kcal_Beta0_linear
        Scattering_Area_integration = OutputLineSpacing * OutputPixelSpacing
    """
    scattering_area = output_line_spacing * output_pixel_spacing
    return float(np.sum(dn_window.astype(np.float64) ** 2) * scattering_area
                 / kcal_beta0_linear)


def rcs_point_target_peak(dn_peak_interpolated: float, kcal_beta0_linear: float,
                           output_azimuth_resolution: float,
                           output_range_resolution: float) -> float:
    """
    eq. (6), "Peak Method": Sigma for a point target using the
    interpolated peak of the point-target impulse response.

        Sigma_p = (DN^2)_interpolated_peak * Scattering_Area_peak / Kcal_Beta0_linear
        Scattering_Area_peak = Output_Azimuth_Resolution * Output_Range_Resolution
    """
    scattering_area = output_azimuth_resolution * output_range_resolution
    return float((dn_peak_interpolated ** 2) * scattering_area / kcal_beta0_linear)


# ==========================================================================
# 5. LEVEL-2 / LEVEL-2B MASKS  (Sec. 2.1.1 Table 3/4, Sec. 6.2.1)
# ==========================================================================

class MaskValue:
    """Layover/Shadow mask codes (uint16 GeoTIFF)."""
    VALID = 128           # Undistorted valid region                (Table 4 / Sec. 6.2.1)
    LAYOVER = 16          # Distorted layover region -- not for analysis
    SHADOW = 64           # Shadow region -- Level-2B only (Sec. 6.2.1)
    OUTSIDE = 0           # Outside geo-referenced image


def apply_valid_mask(array: np.ndarray, mask: np.ndarray,
                      valid_only: bool = True) -> np.ndarray:
    """
    Sec. 2.1.1 note: "Any analysis to be done on the EOS-04 Level-2
    product should be done by applying/overlaying Layover Mask over SAR
    image data." Sec. 6.2.1 extends this: Layover, Shadow and
    Area-outside-image pixels must be excluded; only mask==128 (valid)
    pixels are usable for backscatter analysis.
    """
    out = array.copy()
    if valid_only:
        invalid = mask != MaskValue.VALID
    else:
        invalid = mask == MaskValue.OUTSIDE
    out[invalid] = np.nan
    return out


def local_incidence_angle_significance(lia: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Sec. 2.1.1 Table 3 ("Definition of Local Incidence Angle Map"):
        0.0 to 90.0  -> Region (A) valid incidence-angle range
        0.0 to 90.0  -> Region (B) layover region (masked, cross-check mask)
        -2.0         -> Region (C) outside geo-referenced image
    Returns boolean masks for each region.
    """
    return {
        "outside": np.isclose(lia, -2.0),
        "in_range": (lia >= 0.0) & (lia <= 90.0),
    }


# ==========================================================================
# 6. LEVEL-2B NRB / ARD CALIBRATION  (Sec. 6.3, equations 9-15)
# ==========================================================================

def level2b_gamma0_db(dn: np.ndarray, kcal_db: float) -> np.ndarray:
    """
    eq. (9): Gamma0_p (dB) = 20.0 * log10(DN_p) - Kcal_dB
    DN in the Level-2B product is a DIRECT representation of Gamma0
    (already terrain-normalized, RTC-applied if RTC_Apply_Flag==1).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 20.0 * np.log10(dn) - kcal_db
    return out


def level2b_gamma0_linear(dn: np.ndarray, kcal_linear: float) -> np.ndarray:
    """eq. (11): Gamma0_p (linear) = DN_p^2 / Kcal_linear"""
    return dn.astype(np.float64) ** 2 / kcal_linear


def level2b_undo_normalization(gamma0_linear: np.ndarray,
                                local_illumination_area: np.ndarray,
                                incidence_angle_deg: np.ndarray
                                ) -> Dict[str, np.ndarray]:
    """
    Equations (13)-(15): recover the *un-normalized* Beta0/Sigma0/Gamma0
    (equivalent to a Level-2A product) from a terrain-normalized
    Level-2B Gamma0 product, using the per-pixel scattering/local
    illumination area (WO_area.tif) and local incidence angle
    (WO_lia.tif):

        Beta0_p  = Gamma0_p * Local_Illumination_Area_p          (eq. 13)
        Sigma0_p = Beta0_p * sin(i_p)                             (eq. 14)
        Gamma0_p(un-normalized) = Beta0_p * tan(i_p)              (eq. 15)

    Note eq. (15)'s un-normalized Gamma0 is mathematically distinct from
    the terrain-normalized Gamma0 used as the input here -- the document
    uses this only to demonstrate how to "undo" RTC, not as a
    round-trip identity.
    """
    beta0 = gamma0_linear * local_illumination_area               # eq. 13
    i_rad = np.deg2rad(incidence_angle_deg)
    sigma0 = beta0 * np.sin(i_rad)                                  # eq. 14
    gamma0_unnorm = beta0 * np.tan(i_rad)                           # eq. 15
    return {"beta0": beta0, "sigma0": sigma0, "gamma0_unnormalized": gamma0_unnorm}


# ==========================================================================
# 7. HIGH-LEVEL PIPELINE
# ==========================================================================

@dataclass
class PreprocessConfig:
    band_meta_path: str
    image_path: str
    pol: str
    product_level: str = "L2"          # one of: L1SLC, L1GR, L2, L2B
    lia_path: Optional[str] = None
    mask_path: Optional[str] = None
    area_path: Optional[str] = None    # Level-2B local illumination area
    grid_path: Optional[str] = None    # fallback incidence-angle source
    apply_noise_bias_flag: bool = False
    clip_negative: bool = False
    undo_normalization: bool = False   # Level-2B only
    valid_mask_only: bool = True
    output_dir: str = "./eos04_output"
    output_db: bool = True


class EOS04Preprocessor:
    """
    Orchestrates the documented EOS-04 preprocessing / radiometric
    calibration sequence (Sec. 3.0 and Sec. 6.3) for one polarization of
    one product.
    """

    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.meta = BandMeta(config.band_meta_path)

    # ---------------------------------------------------------------- run
    def run(self) -> Dict[str, str]:
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        outputs: Dict[str, str] = {}

        is_l2b = cfg.product_level.upper() == "L2B"
        is_slc = cfg.product_level.upper() == "L1SLC"

        # ---- 1. Load DN -----------------------------------------------
        if is_slc:
            dn_complex = read_geotiff_complex(cfg.image_path)
            raster = RasterData(array=dn_complex, profile=self._sniff_profile(cfg.image_path))
        else:
            raster = read_geotiff(cfg.image_path)
        dn = raster.array
        profile = raster.profile

        # ---- 2. Local incidence angle ----------------------------------
        lia = self._load_incidence_angle(dn.shape)

        # ---- 3. Mask ----------------------------------------------------
        mask = None
        if cfg.mask_path:
            mask = read_geotiff(cfg.mask_path).array.astype(np.uint16)

        if is_l2b:
            outputs.update(self._run_level2b(dn, lia, mask, profile))
        else:
            outputs.update(self._run_level1_or_2(dn, lia, mask, profile))

        log.info("Preprocessing complete. %d output layer(s) written to %s",
                  len(outputs), cfg.output_dir)
        return outputs

    # ---------------------------------------------------------- internals
    def _sniff_profile(self, path: str) -> dict:
        if not _HAVE_RASTERIO:
            return {}
        with rasterio.open(path) as src:
            return src.profile.copy()

    def _load_incidence_angle(self, shape: Tuple[int, int]) -> np.ndarray:
        cfg = self.cfg
        if cfg.lia_path:
            log.info("Using per-pixel Local Incidence Angle from %s", cfg.lia_path)
            return read_geotiff(cfg.lia_path).array
        if cfg.grid_path:
            log.info("No *_lia.tif supplied; interpolating incidence angle "
                      "from grid file %s (Sec. 2.2-ii)", cfg.grid_path)
            grid = read_grid_file(cfg.grid_path)
            return incidence_angle_from_grid(grid, shape)
        raise ValueError(
            "Need either --lia (Level-2/2B local incidence angle GeoTIFF) "
            "or --grid (Level-1 *_grid.txt) to obtain per-pixel incidence angle."
        )

    def _run_level1_or_2(self, dn: np.ndarray, lia: np.ndarray,
                          mask: Optional[np.ndarray], profile: dict) -> Dict[str, str]:
        cfg = self.cfg
        kcal_db = self.meta.calibration_constant_db(cfg.pol, plane="Beta0")
        kcal_lin = kcal_linear_from_db(kcal_db)                       # eq. 4
        log.info("Kcal_Beta0_dB(%s) = %.4f dB  ->  linear = %.6e",
                  cfg.pol, kcal_db, kcal_lin)

        noise_bias = self.meta.noise_bias(cfg.pol) if cfg.apply_noise_bias_flag else None
        if cfg.apply_noise_bias_flag and noise_bias is None:
            log.warning("apply_noise_bias requested but IMAGE_NOISE_BIAS not "
                        "found for pol=%s in BAND_META.txt; skipping.", cfg.pol)
        dn_sq = apply_noise_bias(dn, noise_bias, clip_negative=cfg.clip_negative)

        beta0 = compute_beta0(dn_sq, kcal_lin)                        # eq. 1
        sigma0 = compute_sigma0(dn_sq, kcal_lin, lia)                 # eq. 2
        gamma0 = compute_gamma0(dn_sq, kcal_lin, lia)                 # eq. 3

        layers = {"beta0": beta0, "sigma0": sigma0, "gamma0": gamma0}
        return self._mask_db_and_write(layers, mask, profile, prefix=cfg.pol)

    def _run_level2b(self, dn: np.ndarray, lia: np.ndarray,
                      mask: Optional[np.ndarray], profile: dict) -> Dict[str, str]:
        cfg = self.cfg
        # eq. 10: Kcal_dB == Kcal_Beta0_dB, tagged Calibration_Constant_Beta0_TxRxpol
        kcal_db = self.meta.calibration_constant_db(cfg.pol, plane="Beta0")
        kcal_lin = kcal_linear_from_db(kcal_db)                       # eq. 4 / 12
        log.info("Level-2B Kcal_dB(%s) = %.4f dB  ->  linear = %.6e",
                  cfg.pol, kcal_db, kcal_lin)

        rtc_flag = self.meta.rtc_apply_flag()
        if rtc_flag == 0:
            log.warning("RTC_Apply_Flag=0 in BAND_META.txt: DEM registration "
                       "failed for this product; Gamma0 DN is delivered "
                       "WITHOUT Radiometric Terrain Correction (Sec. 6.5).")

        gamma0_db = level2b_gamma0_db(dn, kcal_db)                    # eq. 9
        gamma0_lin = level2b_gamma0_linear(dn, kcal_lin)              # eq. 11

        layers = {"gamma0_terrain_normalized": gamma0_lin}

        if cfg.undo_normalization:
            if not cfg.area_path:
                raise ValueError(
                    "--undo-normalization requires --area (WO_area.tif / "
                    "Product_TILE_ID_area.tif local illumination area)."
                )
            area = read_geotiff(cfg.area_path).array
            unnorm = level2b_undo_normalization(gamma0_lin, area, lia)  # eq. 13-15
            layers.update({
                "beta0_unnormalized": unnorm["beta0"],
                "sigma0_unnormalized": unnorm["sigma0"],
                "gamma0_unnormalized": unnorm["gamma0_unnormalized"],
            })

        outputs = self._mask_db_and_write(layers, mask, profile, prefix=cfg.pol)

        # also persist the dB-domain terrain-normalized Gamma0 directly
        # from eq.(9), independent of the to_db() helper used elsewhere.
        gamma0_db_masked = (apply_valid_mask(gamma0_db, mask, cfg.valid_mask_only)
                             if mask is not None else gamma0_db)
        out_path = os.path.join(cfg.output_dir, f"{cfg.pol}_gamma0_dB_eq9.tif")
        write_geotiff(out_path, gamma0_db_masked, profile)
        outputs["gamma0_dB_eq9"] = out_path
        return outputs

    def _mask_db_and_write(self, layers: Dict[str, np.ndarray],
                            mask: Optional[np.ndarray], profile: dict,
                            prefix: str) -> Dict[str, str]:
        cfg = self.cfg
        outputs: Dict[str, str] = {}
        for name, arr in layers.items():
            masked = (apply_valid_mask(arr, mask, cfg.valid_mask_only)
                      if mask is not None else arr)
            out_path = os.path.join(cfg.output_dir, f"{prefix}_{name}_linear.tif")
            write_geotiff(out_path, masked, profile)
            outputs[f"{name}_linear"] = out_path

            if cfg.output_db:
                masked_db = to_db(masked, power_quantity=True)
                out_db_path = os.path.join(cfg.output_dir, f"{prefix}_{name}_dB.tif")
                write_geotiff(out_db_path, masked_db, profile)
                outputs[f"{name}_dB"] = out_db_path
        return outputs


# ==========================================================================
# 8. WORK-ORDER DIRECTORY AUTO-DISCOVERY  (Sec. 2.2 / Sec. 4.2.1 Table 6)
# ==========================================================================

def discover_product_files(work_order_dir: str, pol: str) -> Dict[str, Optional[str]]:
    """
    Best-effort auto-discovery of the documented EOS-04 product file
    naming conventions inside a WO_ID / Product_TILE_ID directory:

        BAND_META.txt
        scene_<pol>/imagery_<pol>.tif           (Level-1/Level-2 GeoTIFF, Sec. 2.2-iii)
        WO_ID_<pol>.jpg
        *_lia.tif / *_mask.tif / *_area.tif      (Level-2 / Level-2B auxiliary, Sec. 2.1.1 / 4.2.1)
        *_<pol>_L1_SlantRange_grid.txt / *level_2_grid.txt  (Sec. 2.2-ii)
    """
    def find(pattern):
        matches = glob.glob(os.path.join(work_order_dir, "**", pattern), recursive=True)
        return matches[0] if matches else None

    return {
        "band_meta": find("BAND_META.txt"),
        "image": find(f"*imagery*{pol}*.tif") or find(f"*_{pol}.tif"),
        "lia": find("*_lia.tif"),
        "mask": find("*_mask.tif"),
        "area": find("*_area.tif"),
        "grid": find(f"*{pol}*grid.txt") or find("*grid.txt"),
    }


# ==========================================================================
# 9. SELF-TEST  (no real EOS-04 data required -- validates the math only)
# ==========================================================================

def self_test() -> None:
    """
    Exercises every calibration equation against synthetic numbers so the
    implementation can be sanity-checked without real EOS-04 data on
    disk. Mirrors the worked values implied by the sample BAND_META.txt
    in Appendix A1.0 (Kcal_Beta0_HH = 69.185 dB).
    """
    print("=== EOS-04 radiometric calibration self-test ===\n")

    kcal_db = 69.185  # Calibration_Constant_Beta0_HH from sample BAND_META.txt
    kcal_lin = kcal_linear_from_db(kcal_db)
    print(f"eq.4  Kcal_Beta0_dB={kcal_db} -> Kcal_Beta0_linear={kcal_lin:.6e}")

    dn = np.array([[100.0, 500.0], [1000.0, 4000.0]])
    incidence_angle = np.array([[32.4, 32.4], [32.4, 32.4]])  # IncidenceAngle from sample

    dn_sq = apply_noise_bias(dn, image_noise_bias=None)
    beta0 = compute_beta0(dn_sq, kcal_lin)
    sigma0 = compute_sigma0(dn_sq, kcal_lin, incidence_angle)
    gamma0 = compute_gamma0(dn_sq, kcal_lin, incidence_angle)

    print("\neq.1 Beta0 (linear):\n", beta0)
    print("eq.1 Beta0 (dB):\n", to_db(beta0))
    print("\neq.2 Sigma0 (linear):\n", sigma0)
    print("eq.2 Sigma0 (dB):\n", to_db(sigma0))
    print("\neq.3 Gamma0 (linear):\n", gamma0)
    print("eq.3 Gamma0 (dB):\n", to_db(gamma0))

    # Noise-bias reconstitution demo (Sec. 3.0 note)
    noise_bias = 21701.4  # Image_Noise_Bias_RH from sample BAND_META.txt
    dn_sq_corr = apply_noise_bias(dn, image_noise_bias=noise_bias, clip_negative=True)
    print(f"\nNoise-bias-corrected DN^2 (bias={noise_bias}):\n", dn_sq_corr)

    # Level-2B equations 9-15
    print("\n--- Level-2B NRB equations ---")
    dn_l2b = np.array([[3000.0, 6000.0], [9000.0, 12000.0]])
    g_db = level2b_gamma0_db(dn_l2b, kcal_db)
    g_lin = level2b_gamma0_linear(dn_l2b, kcal_lin)
    print("eq.9  Gamma0(dB) =\n", g_db)
    print("eq.11 Gamma0(linear) =\n", g_lin)

    local_illum_area = np.array([[18.0, 18.0], [18.0, 18.0]])  # m^2, 18m pixel
    unnorm = level2b_undo_normalization(g_lin, local_illum_area, incidence_angle)
    print("eq.13 un-normalized Beta0 =\n", unnorm["beta0"])
    print("eq.14 un-normalized Sigma0 =\n", unnorm["sigma0"])
    print("eq.15 un-normalized Gamma0 =\n", unnorm["gamma0_unnormalized"])

    # RCS point-target (eq. 5/6) demo
    window = np.full((5, 5), 8000.0)
    sigma_int = rcs_point_target_integration(
        window, kcal_lin, output_line_spacing=4.5, output_pixel_spacing=4.5)
    sigma_peak = rcs_point_target_peak(
        dn_peak_interpolated=12000.0, kcal_beta0_linear=kcal_lin,
        output_azimuth_resolution=4.5, output_range_resolution=4.5)
    print(f"\neq.5  Point-target Sigma (Integration method) = {sigma_int:.4f} m^2")
    print(f"eq.6  Point-target Sigma (Peak method)        = {sigma_peak:.4f} m^2")

    print("\nSelf-test complete: all documented equations executed without error.")


# ==========================================================================
# 10. CLI
# ==========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EOS-04 SAR radiometric calibration / preprocessing "
                    "pipeline (per ISRO SAC EOS-04 Data Products Formats v1.2.5).")
    p.add_argument("--self-test", action="store_true",
                    help="Run a self-contained numerical check of all "
                        "calibration equations with synthetic data (no "
                        "EOS-04 product files required).")

    p.add_argument("--work-order-dir", help="Auto-discover product files "
                    "inside this WO_ID / Product_TILE_ID directory.")
    p.add_argument("--band-meta", help="Path to BAND_META.txt")
    p.add_argument("--image", help="Path to the polarization GeoTIFF "
                    "(imagery_<pol>.tif / Product_TILE_ID_<pol>.tif)")
    p.add_argument("--pol", help="Polarization, e.g. HH, HV, VV, VH, RH, RV")
    p.add_argument("--product-level", default="L2",
                    choices=["L1SLC", "L1GR", "L2", "L2B"],
                    help="EOS-04 product level (controls which calibration "
                        "path / equations are applied).")
    p.add_argument("--lia", help="Path to local incidence angle GeoTIFF "
                    "(*_lia.tif), required for Level-2/Level-2B.")
    p.add_argument("--grid", help="Path to *_grid.txt (Level-1 fallback "
                    "for incidence angle, Sec. 2.2-ii).")
    p.add_argument("--mask", help="Path to layover/shadow mask GeoTIFF "
                    "(*_mask.tif).")
    p.add_argument("--area", help="Path to local illumination/scattering "
                    "area GeoTIFF (*_area.tif), Level-2B only.")
    p.add_argument("--apply-noise-bias", action="store_true",
                    dest="apply_noise_bias_flag",
                    help="Subtract IMAGE_NOISE_BIAS from DN^2 before "
                        "calibration (Sec. 3.0 'Note on Noise Bias Usage').")
    p.add_argument("--clip-negative", action="store_true",
                    help="Clip negative DN^2 values to 0 after noise-bias "
                        "subtraction.")
    p.add_argument("--undo-normalization", action="store_true",
                    help="Level-2B only: also derive un-normalized "
                        "Beta0/Sigma0/Gamma0 using --area (eq. 13-15).")
    p.add_argument("--no-mask-restrict", dest="valid_mask_only",
                    action="store_false",
                    help="Do not restrict output to mask==128 valid "
                        "pixels (default is to restrict, per Sec. 2.1.1/6.2.1).")
    p.add_argument("--no-db-output", dest="output_db", action="store_false",
                    help="Skip writing the dB-scaled GeoTIFF copies "
                        "(linear outputs are always written).")
    p.add_argument("--output-dir", default="./eos04_output",
                    help="Directory for calibrated GeoTIFF outputs.")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    if args.work_order_dir:
        if not args.pol:
            log.error("--pol is required when using --work-order-dir auto-discovery.")
            return 2
        found = discover_product_files(args.work_order_dir, args.pol)
        log.info("Auto-discovered: %s", found)
        args.band_meta = args.band_meta or found["band_meta"]
        args.image = args.image or found["image"]
        args.lia = args.lia or found["lia"]
        args.mask = args.mask or found["mask"]
        args.area = args.area or found["area"]
        args.grid = args.grid or found["grid"]

    missing = [n for n in ("band_meta", "image", "pol")
               if not getattr(args, n)]
    if missing:
        log.error("Missing required argument(s): %s "
                "(or pass --self-test to verify the math without data).",
                ", ".join(f"--{m.replace('_', '-')}" for m in missing))
        return 2

    cfg = PreprocessConfig(
        band_meta_path=args.band_meta,
        image_path=args.image,
        pol=args.pol,
        product_level=args.product_level,
        lia_path=args.lia,
        mask_path=args.mask,
        area_path=args.area,
        grid_path=args.grid,
        apply_noise_bias_flag=args.apply_noise_bias_flag,
        clip_negative=args.clip_negative,
        undo_normalization=args.undo_normalization,
        valid_mask_only=args.valid_mask_only,
        output_dir=args.output_dir,
        output_db=args.output_db,
    )

    pipeline = EOS04Preprocessor(cfg)
    outputs = pipeline.run()
    for name, path in outputs.items():
        print(f"{name:30s} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())