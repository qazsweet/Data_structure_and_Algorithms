#!/usr/bin/env python3
"""
Fringe Pattern Analysis for Optical Alignment
==============================================
Analyzes interference fringe images from folders 1 and 2.
ROI: [1100:2100, 2000:3100]
- Extracts static image statistics
- Detects dark and bright fringe lines
- Applies 3-step Phase-Shifting Interferometry (PSI)
- Identifies the zero-order fringe "flag" in image 3 using images 1 & 2 as reference
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ROI_ROWS = slice(1100, 2100)   # y
ROI_COLS = slice(2000, 3100)   # x

FOLDER1_IMGS = sorted(Path("1").glob("*.bmp"))
FOLDER2_IMGS = sorted(Path("2").glob("*.bmp"))


# ─────────────────────────────────────────────
# 1. Image loading
# ─────────────────────────────────────────────
def load_images(folder_imgs):
    """Load all BMP images from a list of paths, return list of full images."""
    images = []
    for p in folder_imgs:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot load {p}")
        images.append(img)
    return images


def extract_roi(images):
    """Crop ROI from each image, return as float64 list."""
    return [img[ROI_ROWS, ROI_COLS].astype(np.float64) for img in images]


# ─────────────────────────────────────────────
# 2. Static information
# ─────────────────────────────────────────────
def static_info(roi, name=""):
    """Compute and print static statistics for an ROI."""
    info = {
        "name":   name,
        "shape":  roi.shape,
        "dtype":  str(roi.dtype),
        "min":    float(roi.min()),
        "max":    float(roi.max()),
        "mean":   float(roi.mean()),
        "std":    float(roi.std()),
        "median": float(np.median(roi)),
    }
    # Percentiles
    for p in (5, 25, 75, 95):
        info[f"p{p}"] = float(np.percentile(roi, p))
    # Dynamic range
    info["dynamic_range"] = info["max"] - info["min"]
    # Contrast = std/mean
    info["contrast"] = info["std"] / (info["mean"] + 1e-9)
    return info


def print_static_info(info):
    print(f"\n{'─'*60}")
    print(f"  {info['name']}")
    print(f"{'─'*60}")
    print(f"  Shape  : {info['shape']}")
    print(f"  dtype  : {info['dtype']}")
    print(f"  Min    : {info['min']:.1f}    Max     : {info['max']:.1f}")
    print(f"  Mean   : {info['mean']:.2f}   Median  : {info['median']:.2f}")
    print(f"  Std    : {info['std']:.2f}    Contrast: {info['contrast']:.4f}")
    print(f"  P5     : {info['p5']:.1f}    P95     : {info['p95']:.1f}")
    print(f"  Dynamic range: {info['dynamic_range']:.1f}")


# ─────────────────────────────────────────────
# 3. Dark / Bright fringe line extraction
# ─────────────────────────────────────────────
def extract_fringe_lines(roi, smooth_sigma=2.0, prominence_factor=0.05):
    """
    Extract dark (valley) and bright (peak) fringe lines from an ROI.

    Strategy:
      - Average the ROI horizontally to get a vertical intensity profile.
      - Average vertically to get a horizontal intensity profile.
      - Find peaks/valleys in each profile.
      - Also compute 2D local extrema maps.

    Returns dict with:
      dark_rows, bright_rows  – row indices of dark/bright horizontal fringes
      dark_cols, bright_cols  – col indices of dark/bright vertical fringes
      row_profile, col_profile
      dark_map, bright_map    – binary 2D masks
    """
    h, w = roi.shape

    # ── Row-averaged profile (averaged along columns → 1-D over rows)
    row_profile = roi.mean(axis=1)
    row_smooth  = ndimage.gaussian_filter1d(row_profile, smooth_sigma)
    rng         = row_smooth.max() - row_smooth.min()
    prom        = rng * prominence_factor

    bright_rows, _ = find_peaks( row_smooth, prominence=prom, distance=5)
    dark_rows,   _ = find_peaks(-row_smooth, prominence=prom, distance=5)

    # ── Col-averaged profile
    col_profile = roi.mean(axis=0)
    col_smooth  = ndimage.gaussian_filter1d(col_profile, smooth_sigma)
    rng_c       = col_smooth.max() - col_smooth.min()
    prom_c      = rng_c * prominence_factor

    bright_cols, _ = find_peaks( col_smooth, prominence=prom_c, distance=5)
    dark_cols,   _ = find_peaks(-col_smooth, prominence=prom_c, distance=5)

    # ── 2-D local extrema maps via dilation
    roi_u8 = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(roi_u8, (0, 0), smooth_sigma)
    kernel  = np.ones((9, 9), np.uint8)
    local_max = cv2.dilate(blurred, kernel)
    local_min = cv2.erode(blurred, kernel)
    bright_map = (blurred == local_max).astype(np.uint8) * 255
    dark_map   = (blurred == local_min).astype(np.uint8) * 255

    return {
        "dark_rows":    dark_rows,
        "bright_rows":  bright_rows,
        "dark_cols":    dark_cols,
        "bright_cols":  bright_cols,
        "row_profile":  row_profile,
        "col_profile":  col_profile,
        "row_smooth":   row_smooth,
        "col_smooth":   col_smooth,
        "dark_map":     dark_map,
        "bright_map":   bright_map,
    }


# ─────────────────────────────────────────────
# 4. Three-step Phase-Shifting Interferometry
# ─────────────────────────────────────────────
def three_step_psi(I1, I2, I3):
    """
    Standard 3-step PSI (equal 2π/3 phase steps).

    Phase  = arctan[ √3·(I1 − I3) / (2·I2 − I1 − I3) ]
    Modulation = √[ 3·(I1−I3)² + (2I2−I1−I3)² ] / (I1+I2+I3)
    Background = (I1 + I2 + I3) / 3

    Returns: phase [-π, π], modulation [0,1], background
    """
    num   = np.sqrt(3.0) * (I1 - I3)
    denom = 2.0 * I2 - I1 - I3
    phase = np.arctan2(num, denom)

    mod_raw = np.sqrt(3.0 * (I1 - I3)**2 + denom**2)
    bg      = (I1 + I2 + I3) / 3.0
    modulation = mod_raw / (3.0 * bg + 1e-9)   # normalised [0,1]

    return phase, modulation, bg


def find_zero_order_fringe(I1, I2, I3, phase, modulation, min_mod=0.015):
    """
    Locate the 'flag': the zero-order dark fringe at the bull's-eye centre.

    The fringe centre is where the fringe pattern has the highest spatial
    frequency (most compressed rings) – i.e. the phase gradient is maximum.
    This corresponds to the topological singularity of the wrapped phase.

    Strategy:
      1. Compute the wrapped-phase gradient magnitude.
      2. Mask with modulation to keep only real-fringe pixels.
      3. Apply Gaussian weighting biased toward the ROI centre.
      4. Pick the peak of (phase_gradient × modulation × centre_weight).

    Returns (row, col) in ROI coords, phase value at that point,
    and the high-modulation boolean mask.
    """
    h, w  = phase.shape
    mod_mask = modulation > min_mod

    # ── Phase gradient magnitude
    # Use the sine/cosine representation for wrap-safe gradients
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    gx_c  = cv2.Sobel(cos_p, cv2.CV_64F, 1, 0, ksize=3)
    gy_c  = cv2.Sobel(cos_p, cv2.CV_64F, 0, 1, ksize=3)
    gx_s  = cv2.Sobel(sin_p, cv2.CV_64F, 1, 0, ksize=3)
    gy_s  = cv2.Sobel(sin_p, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx_c**2 + gy_c**2 + gx_s**2 + gy_s**2)

    # Smooth gradient to reduce noise
    grad_smooth = cv2.GaussianBlur(grad_mag.astype(np.float32), (21, 21), 7)

    # ── Modulation-weighted score
    score = grad_smooth * modulation
    score[~mod_mask] = 0.0

    # ── Gaussian centre weight (prefer regions near ROI centre)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    sigma  = max(h, w) * 0.4
    centre_weight = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
    score *= centre_weight

    # ── Peak
    flag_row, flag_col = np.unravel_index(np.argmax(score), score.shape)

    return int(flag_row), int(flag_col), mod_mask


# ─────────────────────────────────────────────
# 5. Prediction / verification with image 3
# ─────────────────────────────────────────────
def estimate_phase_from_two(I1, I2):
    """
    Given two images, estimate the phase difference map and
    predict the expected intensity modulation (AC amplitude) map.

    Uses: diff = I2 - I1  →  captures 2π/3 phase shift contribution.
    We compute the gradient magnitude to find active fringe regions.
    Returns the differential phase proxy and active-fringe mask.
    """
    diff  = (I2 - I1).astype(np.float64)
    # Gradient of difference reveals fringe locations
    gx    = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
    gmag  = np.sqrt(gx**2 + gy**2)
    # Normalise
    gmag_norm = (gmag / (gmag.max() + 1e-9))
    # Active fringes: high gradient = fringe boundary
    active_mask = gmag_norm > 0.1
    return diff, gmag_norm, active_mask


def find_flag_in_third_image(rois_12, roi3, folder_name=""):
    """
    Full pipeline: use images 1 & 2 to locate the flag in image 3.

    Steps:
      1. PSI on all 3 images → wrapped phase + modulation
      2. From images 1 & 2 alone → estimate fringe density / activity map
      3. Validate: the flag location from PSI must agree with high
         activity region found from images 1 & 2.
      4. Return flag location in ROI and full-image coordinates.
    """
    I1, I2, I3 = rois_12[0], rois_12[1], roi3

    # Full PSI
    phase, modulation, bg = three_step_psi(I1, I2, I3)

    # Flag from full PSI (ground truth using all 3)
    flag_r, flag_c, mod_mask = find_zero_order_fringe(I1, I2, I3, phase, modulation)

    # Predict flag from images 1 & 2 only (without image 3)
    # Use the same gradient-of-phase approach on a 2-image pseudo-phase
    # Approximate phase ≈ arctan(I1 - I2) direction via raw difference gradient
    diff_map, grad_norm, active_mask = estimate_phase_from_two(I1, I2)

    h, w = phase.shape

    def _fringe_centre_from_single(Im):
        """Find the fringe bull's-eye from a single intensity image."""
        Im_f  = Im.astype(np.float32)
        blur  = cv2.GaussianBlur(Im_f, (5, 5), 1.5)
        # Wrap-safe gradient using image as proxy
        gx    = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy    = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        gmag  = np.sqrt(gx**2 + gy**2)
        gsmth = cv2.GaussianBlur(gmag.astype(np.float32), (31, 31), 10)
        return gsmth

    grad1 = _fringe_centre_from_single(I1)
    grad2 = _fringe_centre_from_single(I2)
    combined_grad = (grad1 + grad2) / 2.0

    # Centre Gaussian weight
    yy, xx = np.mgrid[0:h, 0:w]
    sigma  = max(h, w) * 0.4
    cw     = np.exp(-((yy - h/2)**2 + (xx - w/2)**2) / (2 * sigma**2))

    score_12 = combined_grad * cw
    score_12[~mod_mask] = 0.0
    pred_r, pred_c = np.unravel_index(np.argmax(score_12), score_12.shape)

    return {
        "folder":           folder_name,
        "flag_row_roi":     int(flag_r) if flag_r is not None else None,
        "flag_col_roi":     int(flag_c) if flag_c is not None else None,
        "flag_row_full":    int(flag_r + ROI_ROWS.start) if flag_r is not None else None,
        "flag_col_full":    int(flag_c + ROI_COLS.start) if flag_c is not None else None,
        "flag_phase_val":   float(phase[flag_r, flag_c]) if flag_r is not None else None,
        "predicted_from_12_row_roi": int(pred_r),
        "predicted_from_12_col_roi": int(pred_c),
        "predicted_from_12_row_full": int(pred_r + ROI_ROWS.start),
        "predicted_from_12_col_full": int(pred_c + ROI_COLS.start),
        "max_modulation":   float(modulation.max()),
        "mean_modulation":  float(modulation.mean()),
        "phase":            phase,
        "modulation":       modulation,
        "score_12":         score_12,
        "fringe_lines":     extract_fringe_lines(roi3),
    }


# ─────────────────────────────────────────────
# 6. Visualisation
# ─────────────────────────────────────────────
def save_visualisations(result, rois, prefix):
    phase  = result["phase"]
    mod    = result["modulation"]
    fl     = result["fringe_lines"]
    flag_r = result["flag_row_roi"]
    flag_c = result["flag_col_roi"]
    pred_r = result["predicted_from_12_row_roi"]
    pred_c = result["predicted_from_12_col_roi"]
    I1, I2, I3 = rois[0], rois[1], rois[2]

    def mark_flag(img, r, c, label, color_outer=(255,255,255), color_dot=(0,0,255)):
        if r is None:
            return
        cv2.circle(img, (c, r), 22, color_outer, 3)
        cv2.circle(img, (c, r),  7, color_dot, -1)
        cv2.putText(img, label, (c + 26, r + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_outer, 2)

    # ── Phase map (wrapped, colourised)
    phase_vis   = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    phase_color = cv2.applyColorMap(phase_vis, cv2.COLORMAP_HSV)
    mark_flag(phase_color, flag_r, flag_c, "FLAG (PSI)")
    mark_flag(phase_color, pred_r, pred_c, "PRED (Img1+2)", (0,255,0), (0,200,0))
    cv2.imwrite(f"{prefix}_phase.png", phase_color)

    # ── Modulation map with flag
    mod_vis   = (mod * 255 / (mod.max() + 1e-9)).astype(np.uint8)
    mod_color = cv2.applyColorMap(mod_vis, cv2.COLORMAP_HOT)
    mark_flag(mod_color, flag_r, flag_c, "FLAG")
    cv2.imwrite(f"{prefix}_modulation.png", mod_color)

    # ── Zero-order dark region overlay on average image
    Imean = (I1 + I2 + I3) / 3.0
    mean_u8  = cv2.normalize(Imean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_bgr = cv2.cvtColor(mean_u8, cv2.COLOR_GRAY2BGR)

    # Otsu threshold dark region
    smooth   = cv2.GaussianBlur(mean_u8, (15, 15), 5)
    _, dark_mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Overlay dark region in blue
    dark_overlay = mean_bgr.copy()
    dark_overlay[dark_mask > 0] = (200, 80, 0)   # blue tint for dark fringes
    # Overlay bright peaks in yellow
    _, bright_mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_overlay[bright_mask > 0, 0] = 0
    dark_overlay[bright_mask > 0, 1] = min(255, int(dark_overlay[bright_mask > 0, 1].mean()) + 30)
    overlay_out = cv2.addWeighted(mean_bgr, 0.55, dark_overlay, 0.45, 0)
    mark_flag(overlay_out, flag_r, flag_c, "FLAG (zero-order)")
    mark_flag(overlay_out, pred_r, pred_c, "PRED", (0,255,0), (0,200,0))
    cv2.imwrite(f"{prefix}_dark_bright_overlay.png", overlay_out)

    # ── Score map from images 1 & 2
    sc     = result["score_12"]
    sc_vis = (sc / (sc.max() + 1e-9) * 255).astype(np.uint8)
    sc_color = cv2.applyColorMap(sc_vis, cv2.COLORMAP_JET)
    mark_flag(sc_color, pred_r, pred_c, "PRED (Img1+2 only)", (255,255,255), (255,255,255))
    cv2.imwrite(f"{prefix}_score12.png", sc_color)

    # ── Fringe line profiles (horizontal + vertical)
    row_prof = fl["row_smooth"]
    col_prof = fl["col_smooth"]

    # Draw row profile chart
    h_chart, w_chart = 300, 800
    chart_r = np.full((h_chart, w_chart, 3), 30, dtype=np.uint8)
    r_min, r_max = row_prof.min(), row_prof.max()
    pts_r = []
    for i, v in enumerate(row_prof):
        x = int(i / len(row_prof) * w_chart)
        y = int((1 - (v - r_min) / (r_max - r_min + 1e-9)) * (h_chart - 20) + 10)
        pts_r.append((x, y))
    for i in range(1, len(pts_r)):
        cv2.line(chart_r, pts_r[i-1], pts_r[i], (0, 200, 255), 1)
    for r in fl["dark_rows"]:
        x = int(r / len(row_prof) * w_chart)
        cv2.line(chart_r, (x, 0), (x, h_chart), (50, 50, 200), 1)
    for r in fl["bright_rows"]:
        x = int(r / len(row_prof) * w_chart)
        cv2.line(chart_r, (x, 0), (x, h_chart), (0, 200, 50), 1)
    cv2.putText(chart_r, "Row profile (avg across cols)  |dark|bright",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imwrite(f"{prefix}_row_profile.png", chart_r)

    print(f"  Saved: {prefix}_phase.png")
    print(f"  Saved: {prefix}_modulation.png")
    print(f"  Saved: {prefix}_dark_bright_overlay.png")
    print(f"  Saved: {prefix}_score12.png")
    print(f"  Saved: {prefix}_row_profile.png")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────
def analyse_folder(folder_imgs, folder_name):
    print(f"\n{'='*60}")
    print(f"  FOLDER {folder_name}  –  {len(folder_imgs)} images")
    print(f"{'='*60}")

    if len(folder_imgs) < 3:
        print("  Need at least 3 images. Skipping.")
        return None

    images = load_images(folder_imgs)
    rois   = extract_roi(images)

    # ── Static information for each image
    static_results = []
    for i, (p, roi) in enumerate(zip(folder_imgs, rois)):
        label = f"Folder {folder_name} | Image {i+1} | {p.name}"
        info  = static_info(roi, label)
        print_static_info(info)
        static_results.append(info)

    # ── Fringe extraction for each image
    print(f"\n  Fringe Line Analysis (ROI {ROI_ROWS.start}:{ROI_ROWS.stop}, "
          f"{ROI_COLS.start}:{ROI_COLS.stop}):")
    for i, roi in enumerate(rois):
        fl = extract_fringe_lines(roi)
        print(f"    Image {i+1}: "
              f"{len(fl['dark_rows'])} dark rows, {len(fl['bright_rows'])} bright rows | "
              f"{len(fl['dark_cols'])} dark cols, {len(fl['bright_cols'])} bright cols")
        if len(fl['dark_rows']) > 0:
            print(f"      Dark row positions (ROI):   {fl['dark_rows'][:10]}")
        if len(fl['bright_rows']) > 0:
            print(f"      Bright row positions (ROI): {fl['bright_rows'][:10]}")

    # ── Phase-Shifting Interferometry on all 3 images
    print(f"\n  3-Step PSI Analysis:")
    phase, mod, bg = three_step_psi(rois[0], rois[1], rois[2])
    print(f"    Wrapped phase range : {phase.min():.4f} to {phase.max():.4f} rad")
    print(f"    Modulation max/mean : {mod.max():.4f} / {mod.mean():.4f}")
    print(f"    Background mean     : {bg.mean():.2f}")

    # ── Find flag using images 1 & 2, verify in image 3
    print(f"\n  FLAG Detection:")
    result = find_flag_in_third_image(rois[:2], rois[2], folder_name)

    print(f"    ┌─ FLAG (from full PSI on all 3 images)")
    print(f"    │   ROI  coords : row={result['flag_row_roi']},  col={result['flag_col_roi']}")
    print(f"    │   FULL coords : row={result['flag_row_full']}, col={result['flag_col_full']}")
    print(f"    │   Phase value : {result['flag_phase_val']:.4f} rad")
    print(f"    │")
    print(f"    └─ PREDICTED from Images 1 & 2 only")
    print(f"        ROI  coords : row={result['predicted_from_12_row_roi']},  col={result['predicted_from_12_col_roi']}")
    print(f"        FULL coords : row={result['predicted_from_12_row_full']}, col={result['predicted_from_12_col_full']}")

    # Distance between PSI flag and prediction
    if result['flag_row_roi'] is not None:
        dr = result['flag_row_roi']  - result['predicted_from_12_row_roi']
        dc = result['flag_col_roi']  - result['predicted_from_12_col_roi']
        dist = np.sqrt(dr**2 + dc**2)
        print(f"    Prediction error: Δrow={dr}, Δcol={dc}, dist={dist:.1f} px")

    # ── Save visualisations
    print(f"\n  Saving visualisations...")
    save_visualisations(result, rois, prefix=f"analysis_folder{folder_name}")

    return result


def main():
    print("=" * 60)
    print("  INTERFERENCE FRINGE ANALYSIS")
    print(f"  ROI: [{ROI_ROWS.start}:{ROI_ROWS.stop}, {ROI_COLS.start}:{ROI_COLS.stop}]")
    print("=" * 60)

    print(f"\nFolder 1 images: {[p.name for p in FOLDER1_IMGS]}")
    print(f"Folder 2 images: {[p.name for p in FOLDER2_IMGS]}")

    results = {}

    r1 = analyse_folder(FOLDER1_IMGS, "1")
    r2 = analyse_folder(FOLDER2_IMGS, "2")

    results["folder1"] = {k: v for k, v in r1.items()
                          if not isinstance(v, np.ndarray)} if r1 else None
    results["folder2"] = {k: v for k, v in r2.items()
                          if not isinstance(v, np.ndarray)} if r2 else None

    # ── Cross-folder comparison
    if r1 and r2:
        print(f"\n{'='*60}")
        print("  CROSS-FOLDER FLAG COMPARISON")
        print(f"{'='*60}")
        for key in ("flag_row_full", "flag_col_full",
                    "predicted_from_12_row_full", "predicted_from_12_col_full"):
            v1 = r1.get(key)
            v2 = r2.get(key)
            delta = (v2 - v1) if (v1 is not None and v2 is not None) else "N/A"
            print(f"  {key:40s}: F1={v1}  F2={v2}  Δ={delta}")

    # Save summary JSON
    with open("fringe_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  Results saved to fringe_analysis_results.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
