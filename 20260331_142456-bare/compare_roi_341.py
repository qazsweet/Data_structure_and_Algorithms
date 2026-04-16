"""
Compare the ROI of *_341*.png against every earlier *.png in the same directory.

ROI matches contact_line_info.py: img[1100:2100, 2000:3100] (row, col).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

# Default folder and ROI (NumPy indexing: rows, cols)
DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "20260330_215619")
ROI_ROW = slice(1100, 2100)
ROI_COL = slice(2000, 3100)
THRESHOLDS = (5, 10)


def find_341_basename(files: list[str]) -> str | None:
    """Return basename of first file whose name contains _341 (before extension)."""
    for f in files:
        base = os.path.basename(f)
        if re.search(r"_341(?:\.|$)", base):
            return base
    return None


def enumerate_earlier_pngs(folder: str) -> tuple[str, list[str]]:
    """
    List all *.png in folder sorted by name; find *341* target; return (target_path, [earlier paths]).
    """
    pattern = os.path.join(folder, "*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PNG files in {folder}")

    target_name = find_341_basename(files)
    if not target_name:
        raise FileNotFoundError(f"No file matching *_341*.png in {folder}")

    target_path = os.path.join(folder, target_name)
    earlier = [f for f in files if os.path.basename(f) < target_name]
    return target_path, earlier


def load_roi(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img[ROI_ROW, ROI_COL].astype(np.float64)


def metrics(ref: np.ndarray, other: np.ndarray, taus: tuple[int, ...]) -> dict:
    d = ref - other
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    mx = float(np.max(np.abs(d)))
    corr = float(np.corrcoef(ref.ravel(), other.ravel())[0, 1])
    frac = {}
    for t in taus:
        frac[t] = float(np.mean(np.abs(d) > t))
    return {"mae": mae, "rmse": rmse, "max_abs": mx, "corr": corr, "frac_gt": frac}


def save_abs_diff(
    ref: np.ndarray,
    other: np.ndarray,
    out_path: str,
) -> None:
    d = np.abs(ref - other)
    # scale to 0-255 for visibility
    d8 = np.clip(d, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, d8)


def main() -> int:
    p = argparse.ArgumentParser(description="ROI compare *_341*.png vs earlier PNGs in folder.")
    p.add_argument(
        "--dir",
        dest="folder",
        default=DEFAULT_DIR,
        help=f"Directory with PNGs (default: {DEFAULT_DIR})",
    )
    p.add_argument(
        "--save-diff",
        metavar="SUBDIR",
        nargs="?",
        const="roi_diff_341",
        default=None,
        help="Write abs-diff PNGs under DIR/SUBDIR (default subdir: roi_diff_341)",
    )
    args = p.parse_args()
    folder = os.path.abspath(args.folder)

    target_path, earlier = enumerate_earlier_pngs(folder)
    ref = load_roi(target_path)
    target_base = os.path.basename(target_path)

    print(f"Target: {target_base}")
    print(f"ROI: rows [1100:2100], cols [2000:3100] -> shape {ref.shape}")
    print(f"Earlier PNGs ({len(earlier)}):")
    for f in earlier:
        print(f"  {os.path.basename(f)}")
    print()

    diff_dir = None
    if args.save_diff is not None:
        diff_dir = os.path.join(folder, args.save_diff)
        os.makedirs(diff_dir, exist_ok=True)

    # table header (ASCII labels for Windows consoles)
    tau_cols = "  ".join(f"P(abs_d>{t})" for t in THRESHOLDS)
    print(f"{'vs':<40} {'MAE':>8} {'RMSE':>8} {'max_abs':>8} {'corr':>8}  {tau_cols}")
    print("-" * 100)

    stem341 = os.path.splitext(target_base)[0]
    earlier.append(target_path)
    for path in earlier:
        name = os.path.basename(path)
        o = load_roi(path)
        if o.shape != ref.shape:
            print(f"{name}: shape mismatch {o.shape} vs {ref.shape}", file=sys.stderr)
            continue
        m = metrics(ref, o, THRESHOLDS)
        frac_str = "  ".join(f"{m['frac_gt'][t]:>10.4f}" for t in THRESHOLDS)
        print(
            f"{name:<40} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['max_abs']:>8.1f} {m['corr']:>8.4f}  {frac_str}"
        )
        if diff_dir is not None:
            stem = os.path.splitext(name)[0]
            out_png = os.path.join(diff_dir, f"absdiff_{stem341}__vs__{stem}.png")
            save_abs_diff(ref, o, out_png)

    if diff_dir:
        print(f"\nAbs-diff images: {diff_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
