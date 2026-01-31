from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    center_x: float
    center_y: float
    score: float
    scale: float
    template_wh: tuple[int, int]


def _read_gray_u8(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _make_plus_template_u8(*, arm_len: int, thickness: int, margin: int) -> np.ndarray:
    """
    Create a simple '+' cross template (white on black background) as uint8.
    """
    arm_len = int(arm_len)
    thickness = int(thickness)
    margin = int(margin)
    if arm_len < 2:
        raise ValueError("arm_len must be >= 2")
    if thickness < 1:
        raise ValueError("thickness must be >= 1")
    if margin < 0:
        raise ValueError("margin must be >= 0")

    size = 2 * arm_len + 1 + 2 * margin
    tpl = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    half_t = thickness // 2

    # Vertical bar.
    tpl[max(0, c - arm_len) : min(size, c + arm_len + 1), max(0, c - half_t) : min(size, c + half_t + 1)] = 255
    # Horizontal bar.
    tpl[max(0, c - half_t) : min(size, c + half_t + 1), max(0, c - arm_len) : min(size, c + arm_len + 1)] = 255
    return tpl


def _refine_subpixel_quadratic_max(response: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """
    Subpixel refinement of a local maximum using a 3x3 quadratic approximation.

    Returns refined (x, y) in response-map coordinates.
    """
    h, w = response.shape[:2]
    xi, yi = int(round(float(x))), int(round(float(y)))
    if xi < 1 or xi >= (w - 1) or yi < 1 or yi >= (h - 1):
        return float(x), float(y)

    patch = response[yi - 1 : yi + 2, xi - 1 : xi + 2].astype(np.float64)
    gx = 0.5 * (patch[1, 2] - patch[1, 0])
    gy = 0.5 * (patch[2, 1] - patch[0, 1])
    gxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
    gyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
    gxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])

    hess = np.array([[gxx, gxy], [gxy, gyy]], dtype=np.float64)
    grad = np.array([gx, gy], dtype=np.float64)

    det = float(np.linalg.det(hess))
    if not np.isfinite(det) or abs(det) < 1e-12:
        return float(x), float(y)

    try:
        delta = -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        return float(x), float(y)

    dx, dy = float(delta[0]), float(delta[1])
    if abs(dx) > 1.0 or abs(dy) > 1.0:
        return float(x), float(y)

    return float(xi + dx), float(yi + dy)


def _nms_peaks_max(
    response: np.ndarray,
    *,
    topk: int,
    radius: int,
    min_score: float,
) -> np.ndarray:
    """
    NMS on a max-response map. Returns Nx3 array of (x, y, score).
    """
    if topk <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if response.ndim != 2:
        raise ValueError("response must be 2D")

    resp = response.astype(np.float32, copy=False)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dil = cv2.dilate(resp, kernel)
    is_peak = (resp == dil) & (resp >= float(min_score))
    ys, xs = np.where(is_peak)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    scores = resp[ys, xs]
    order = np.argsort(scores)[::-1]
    order = order[: min(order.size, int(topk))]

    out = np.stack([xs[order], ys[order], scores[order]], axis=1).astype(np.float32)
    return out


def _parse_scales(scales_csv: str) -> list[float]:
    items = [s.strip() for s in str(scales_csv).split(",") if s.strip()]
    if not items:
        raise ValueError("scales must be a non-empty comma-separated list")
    scales: list[float] = []
    for s in items:
        v = float(s)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid scale: {s}")
        scales.append(v)
    return scales


def _match_one(
    gray_u8: np.ndarray,
    template_u8: np.ndarray,
    *,
    method: int,
    topk: int,
    nms_radius: int,
    min_score: float,
    prior_xy: Optional[tuple[float, float]] = None,
    prior_sigma_px: float = 0.0,
) -> tuple[Optional[MatchResult], Optional[np.ndarray]]:
    h, w = gray_u8.shape[:2]
    th, tw = template_u8.shape[:2]
    if th < 3 or tw < 3 or th > h or tw > w:
        return None, None

    resp = cv2.matchTemplate(gray_u8, template_u8, method=method).astype(np.float32)

    # Find candidate peaks in response-map coordinates (top-left of template).
    peaks = _nms_peaks_max(resp, topk=topk, radius=nms_radius, min_score=min_score)
    if peaks.size == 0:
        return None, resp

    # Convert candidates to image-space centers, and (optionally) choose closest to prior.
    cx_off = (tw - 1) * 0.5
    cy_off = (th - 1) * 0.5

    if prior_xy is not None and float(prior_sigma_px) > 0:
        px, py = float(prior_xy[0]), float(prior_xy[1])
        d2 = (peaks[:, 0] + cx_off - px) ** 2 + (peaks[:, 1] + cy_off - py) ** 2
        # Gaussian prior weighting.
        wts = np.exp(-0.5 * d2 / (float(prior_sigma_px) ** 2))
        pick = int(np.argmax(peaks[:, 2] * wts))
    elif prior_xy is not None:
        px, py = float(prior_xy[0]), float(prior_xy[1])
        d2 = (peaks[:, 0] + cx_off - px) ** 2 + (peaks[:, 1] + cy_off - py) ** 2
        pick = int(np.argmin(d2))
    else:
        pick = 0

    x0, y0, s0 = float(peaks[pick, 0]), float(peaks[pick, 1]), float(peaks[pick, 2])
    x1, y1 = _refine_subpixel_quadratic_max(resp, x0, y0)

    return (
        MatchResult(
            center_x=float(x1 + cx_off),
            center_y=float(y1 + cy_off),
            score=float(s0),
            scale=1.0,
            template_wh=(tw, th),
        ),
        resp,
    )


def find_cross_center_subpixel(
    gray_u8: np.ndarray,
    *,
    template_u8: np.ndarray,
    scales: list[float],
    topk: int = 50,
    nms_radius: int = 6,
    min_score: float = 0.55,
    prior_xy: Optional[tuple[float, float]] = None,
    prior_sigma_px: float = 0.0,
    try_invert: bool = True,
) -> MatchResult:
    """
    Find a cross by template matching and return its center at subpixel precision.

    The returned (x, y) are in image pixel coordinates with origin at top-left.
    """
    method = cv2.TM_CCOEFF_NORMED
    best: Optional[MatchResult] = None

    for scale in scales:
        if scale <= 0:
            continue
        th0, tw0 = template_u8.shape[:2]
        tw = int(round(tw0 * float(scale)))
        th = int(round(th0 * float(scale)))
        if tw < 3 or th < 3:
            continue

        tpl = cv2.resize(template_u8, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        candidates = [tpl]
        if try_invert:
            candidates.append(cv2.bitwise_not(tpl))

        for cand in candidates:
            r, _ = _match_one(
                gray_u8,
                cand,
                method=method,
                topk=topk,
                nms_radius=nms_radius,
                min_score=min_score,
                prior_xy=prior_xy,
                prior_sigma_px=prior_sigma_px,
            )
            if r is None:
                continue
            r = MatchResult(
                center_x=r.center_x,
                center_y=r.center_y,
                score=r.score,
                scale=float(scale),
                template_wh=(tw, th),
            )
            if best is None or r.score > best.score:
                best = r

    if best is None:
        raise RuntimeError("No cross match found (try lowering --min-score, adjusting template, or adding scales).")
    return best


def _draw_result(bgr: np.ndarray, r: MatchResult) -> np.ndarray:
    out = bgr.copy()
    x, y = float(r.center_x), float(r.center_y)
    cv2.drawMarker(
        out,
        (int(round(x)), int(round(y))),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"({x:.3f}, {y:.3f}) score={r.score:.3f} scale={r.scale:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Find a cross template center with subpixel accuracy.")
    parser.add_argument("image", help="Path to image (png/jpg/...)")
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template image path. If omitted, a synthetic '+' template is used.",
    )
    parser.add_argument("--arm-len", type=int, default=18, help="Synthetic '+' arm length (pixels)")
    parser.add_argument("--thickness", type=int, default=5, help="Synthetic '+' bar thickness (pixels)")
    parser.add_argument("--margin", type=int, default=6, help="Synthetic '+' margin (pixels)")
    parser.add_argument(
        "--scales",
        default="0.75,1.0,1.25",
        help="Comma-separated template scales to search (e.g. 0.5,0.75,1.0,1.25)",
    )
    parser.add_argument("--topk", type=int, default=80, help="How many NMS peaks to consider per scale")
    parser.add_argument("--nms-radius", type=int, default=6, help="NMS radius on the response map (pixels)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.55,
        help="Minimum match score (TM_CCOEFF_NORMED). Lower if no match is found.",
    )
    parser.add_argument("--no-invert", action="store_true", help="Disable inverted-template attempt")
    parser.add_argument("--prior-x", type=float, default=None, help="Optional prior center x (pixels)")
    parser.add_argument("--prior-y", type=float, default=None, help="Optional prior center y (pixels)")
    parser.add_argument(
        "--prior-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian prior sigma (pixels). 0 means nearest-only if prior is set.",
    )
    parser.add_argument("--out", default=None, help="Optional output visualization path")
    args = parser.parse_args()

    bgr, gray = _read_gray_u8(args.image)
    scales = _parse_scales(args.scales)

    if args.template is None:
        template = _make_plus_template_u8(arm_len=args.arm_len, thickness=args.thickness, margin=args.margin)
    else:
        _, template = _read_gray_u8(args.template)

    prior_xy: Optional[tuple[float, float]] = None
    if args.prior_x is not None and args.prior_y is not None:
        prior_xy = (float(args.prior_x), float(args.prior_y))

    r = find_cross_center_subpixel(
        gray,
        template_u8=template,
        scales=scales,
        topk=int(args.topk),
        nms_radius=int(args.nms_radius),
        min_score=float(args.min_score),
        prior_xy=prior_xy,
        prior_sigma_px=float(args.prior_sigma),
        try_invert=not bool(args.no_invert),
    )

    # Main requested output: subpixel center coordinate.
    print(f"{r.center_x:.6f} {r.center_y:.6f}  score={r.score:.4f}  scale={r.scale:.4f}  tpl={r.template_wh[0]}x{r.template_wh[1]}")

    if args.out is not None:
        vis = _draw_result(bgr, r)
        cv2.imwrite(args.out, vis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

