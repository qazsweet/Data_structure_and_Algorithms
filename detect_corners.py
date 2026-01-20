from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SubpixCriteria:
    max_iter: int = 40
    eps: float = 1e-3

    def as_cv2(self) -> Tuple[int, int, float]:
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.eps)


def _read_gray(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray_u8


def _saddle_response(gray_u8: np.ndarray, blur_sigma: float = 0.8) -> np.ndarray:
    """
    Compute a saddle-like corner response using the Hessian determinant.

    For a saddle point, det(H) = Ixx*Iyy - Ixy^2 is negative (one direction concave, the other convex).
    We use response = max(0, -det(H)) so stronger saddles have higher score.
    """
    gray = gray_u8.astype(np.float32) / 255.0
    if blur_sigma > 0:
        k = int(round(blur_sigma * 6 + 1)) | 1
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)

    # First derivatives (for edge suppression).
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Second derivatives.
    ixx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)

    det_h = ixx * iyy - ixy * ixy
    # Curvature balance: chessboard corners tend to have opposite-sign principal curvatures
    # with relatively small |trace| compared to |Ixx|+|Iyy|.
    trace = ixx + iyy
    denom = (np.abs(ixx) + np.abs(iyy) + 1e-6)
    balance = 1.0 - (np.abs(trace) / denom)
    balance = np.clip(balance, 0.0, 1.0)

    # Edge suppression: downweight strong edges where gradient magnitude is large.
    g2 = dx * dx + dy * dy
    # alpha chosen empirically; higher = more aggressive edge suppression.
    alpha = 8.0
    edge_weight = 1.0 / (1.0 + alpha * g2)

    resp = np.maximum(0.0, -det_h) * balance * edge_weight
    return resp


def _nms_peaks(
    response: np.ndarray,
    max_points: int = 5000,
    radius: int = 4,
    *,
    thr_mode: str = "quantile",
    thr_quantile: float = 0.99985,
    rel_thresh: float = 0.2,
) -> np.ndarray:
    """
    Non-maximum suppression on a response map.

    Returns Nx2 array of (x, y) float coordinates at integer pixel locations.
    """
    if response.ndim != 2:
        raise ValueError("response must be 2D")
    if radius < 1:
        raise ValueError("radius must be >= 1")

    rmax = float(np.max(response))
    if not np.isfinite(rmax) or rmax <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    flat = response[response > 0].ravel()
    if flat.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if thr_mode == "rel":
        thr = rmax * float(rel_thresh)
    elif thr_mode == "quantile":
        q = float(thr_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("thr_quantile must be in (0, 1)")
        thr = float(np.quantile(flat, q))
    else:
        raise ValueError("thr_mode must be 'quantile' or 'rel'")

    k = 2 * radius + 1
    kernel = np.ones((k, k), np.uint8)
    # Dilation-based NMS.
    dil = cv2.dilate(response, kernel)
    is_peak = (response == dil) & (response >= thr)

    ys, xs = np.where(is_peak)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    scores = response[ys, xs]
    order = np.argsort(scores)[::-1]
    if order.size > max_points:
        order = order[:max_points]

    pts = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
    return pts


def _refine_subpixel_quadratic(response: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    Refine integer-pixel peak locations to subpixel using a local quadratic fit on the response map.

    Uses a single Newton step on a 2D quadratic approximation over a 3x3 neighborhood.
    """
    if pts_xy.size == 0:
        return pts_xy.astype(np.float32)

    h, w = response.shape[:2]
    pts = pts_xy.astype(np.float32).copy()

    for i in range(pts.shape[0]):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        xi, yi = int(round(x)), int(round(y))
        if xi < 1 or xi >= (w - 1) or yi < 1 or yi >= (h - 1):
            continue

        patch = response[yi - 1 : yi + 2, xi - 1 : xi + 2].astype(np.float64)
        # Finite-difference gradient.
        gx = 0.5 * (patch[1, 2] - patch[1, 0])
        gy = 0.5 * (patch[2, 1] - patch[0, 1])
        # Finite-difference Hessian.
        gxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
        gyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
        gxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])

        hess = np.array([[gxx, gxy], [gxy, gyy]], dtype=np.float64)
        grad = np.array([gx, gy], dtype=np.float64)

        det = float(np.linalg.det(hess))
        if not np.isfinite(det) or abs(det) < 1e-12:
            continue

        try:
            delta = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            continue

        # Clamp to a reasonable subpixel range within the 3x3 patch.
        dx, dy = float(delta[0]), float(delta[1])
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            continue

        pts[i, 0] = xi + dx
        pts[i, 1] = yi + dy

    return pts


def detect_saddle_points(
    gray_u8: np.ndarray,
    *,
    blur_sigma: float = 0.8,
    nms_radius: int = 4,
    max_points: int = 5000,
    thr_mode: str = "quantile",
    thr_quantile: float = 0.99985,
    rel_thresh: float = 0.2,
    subpixel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect saddle-point corner candidates on a chessboard.

    Returns:
        points_xy: Nx2 float32 array of (x, y) locations (subpixel if enabled).
        response:  HxW float32 response image.
    """
    resp = _saddle_response(gray_u8, blur_sigma=blur_sigma).astype(np.float32)
    pts = _nms_peaks(
        resp,
        max_points=max_points,
        radius=nms_radius,
        thr_mode=thr_mode,
        thr_quantile=thr_quantile,
        rel_thresh=rel_thresh,
    )
    if subpixel and pts.size:
        pts = _refine_subpixel_quadratic(resp, pts)
    return pts, resp


def detect_chessboard_corners_sb(
    gray_u8: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, Optional[np.ndarray]]:
    """
    OpenCV's saddle-based chessboard detector (subpixel output).
    """
    flags = (
        cv2.CALIB_CB_EXHAUSTIVE
        + cv2.CALIB_CB_ACCURACY
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCornersSB(gray_u8, pattern_size, flags=flags)
    if not found:
        return False, None
    return True, corners


def _draw_points(bgr: np.ndarray, pts_xy: np.ndarray, color=(0, 0, 255), radius: int = 2) -> np.ndarray:
    out = bgr.copy()
    for x, y in pts_xy.reshape(-1, 2):
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chessboard corner detection using saddle points + subpixel refinement."
    )
    parser.add_argument("image", help="Path to image (png/jpg/...)")

    parser.add_argument("--method", choices=["saddle", "sb", "classic"], default="sb")
    parser.add_argument("--rows", type=int, default=None, help="Inner corners per row")
    parser.add_argument("--cols", type=int, default=None, help="Inner corners per column")

    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument(
        "--thr-mode",
        choices=["quantile", "rel"],
        default="quantile",
        help="Thresholding mode for saddle response before NMS",
    )
    parser.add_argument(
        "--thr-quantile",
        type=float,
        default=0.99985,
        help="Quantile in (0,1) used when --thr-mode=quantile",
    )
    parser.add_argument(
        "--rel-thresh",
        type=float,
        default=0.2,
        help="Relative threshold (fraction of max) used when --thr-mode=rel",
    )
    parser.add_argument("--no-subpixel", action="store_true", help="Disable subpixel refinement")

    parser.add_argument("--out", default=None, help="Output image path (defaults to auto name)")
    args = parser.parse_args()

    bgr, gray = _read_gray(args.image)

    pattern_size: Optional[tuple[int, int]] = None
    if args.rows is not None and args.cols is not None:
        pattern_size = (int(args.rows), int(args.cols))

    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = f"corners_{args.method}_{stem}.png"

    if args.method == "saddle":
        pts, resp = detect_saddle_points(
            gray,
            blur_sigma=args.blur_sigma,
            nms_radius=args.nms_radius,
            max_points=args.max_points,
            thr_mode=args.thr_mode,
            thr_quantile=args.thr_quantile,
            rel_thresh=args.rel_thresh,
            subpixel=not args.no_subpixel,
        )
        print(f"Detected {pts.shape[0]} saddle-point candidates.")
        vis = _draw_points(bgr, pts, color=(0, 0, 255), radius=2)
        cv2.imwrite(out_path, vis)
        print(f"Saved visualization: {out_path}")
        return 0

    if pattern_size is None:
        print("Error: --rows and --cols are required for --method sb/classic.")
        return 2

    if args.method == "sb":
        found, corners = detect_chessboard_corners_sb(gray, pattern_size)
        if not found or corners is None:
            print(f"Chessboard not found with pattern_size={pattern_size} using SB.")
            return 1
        cv2.drawChessboardCorners(bgr, pattern_size, corners, True)
        cv2.imwrite(out_path, bgr)
        print(f"Found {corners.shape[0]} corners (subpixel). Saved: {out_path}")
        return 0

    # classic OpenCV method + cornerSubPix
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=(cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
    )
    if not found:
        print(f"Chessboard not found with pattern_size={pattern_size} using classic.")
        return 1

    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=SubpixCriteria().as_cv2(),
    )
    cv2.drawChessboardCorners(bgr, pattern_size, corners, True)
    cv2.imwrite(out_path, bgr)
    print(f"Found {corners.shape[0]} corners (subpixel). Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
