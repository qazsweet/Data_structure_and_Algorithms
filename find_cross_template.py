#!/usr/bin/env python3
"""
Cross Template Detection with Subpixel Accuracy

This script finds cross patterns (the "+" or "×" shaped intersections where 
4 checkerboard squares meet) using multiple detection methods with subpixel refinement.

Methods:
1. Template matching with normalized cross-correlation
2. Hessian-based saddle point detection
3. Combined approach for best coverage

Usage:
    python find_cross_template.py <image_path> [--method METHOD] [--out OUTPUT]
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, Optional, List

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter


def create_cross_template(size: int = 31, line_width: int = 1) -> np.ndarray:
    """
    Create a cross (+) template mimicking the checkerboard corner pattern.
    
    The template represents the local pattern at a checkerboard corner where
    4 squares meet: alternating black/white in a cross configuration.
    
    Args:
        size: Template size (should be odd for symmetry)
        line_width: Width of transition zone
    
    Returns:
        Cross template as float32 array normalized to [0, 1]
    """
    if size % 2 == 0:
        size += 1
    
    template = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    # Create checkerboard-like cross pattern:
    # Top-left and bottom-right are white (1), top-right and bottom-left are black (0)
    for y in range(size):
        for x in range(size):
            # Determine quadrant
            if (y < center and x < center) or (y >= center and x >= center):
                template[y, x] = 1.0  # White quadrant
            else:
                template[y, x] = 0.0  # Black quadrant
    
    # Apply Gaussian blur to simulate realistic edge transitions
    template = cv2.GaussianBlur(template, (0, 0), sigmaX=1.5)
    
    return template


def create_saddle_cross_template(size: int = 41) -> np.ndarray:
    """
    Create a template that matches the saddle-point pattern at checkerboard corners.
    
    This is the characteristic pattern where 4 squares meet: two diagonal 
    opposites are dark, two are light, creating a saddle in intensity.
    
    Args:
        size: Template size (should be odd)
    
    Returns:
        Template as float32 array
    """
    if size % 2 == 0:
        size += 1
    
    center = size // 2
    template = np.zeros((size, size), dtype=np.float32)
    
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    y = y - center
    x = x - center
    
    # Pattern: sign(x*y) gives us the checkerboard cross pattern
    # Positive for top-right and bottom-left, negative for top-left and bottom-right
    pattern = np.sign(x * y).astype(np.float32)
    
    # Handle the center lines (where x=0 or y=0)
    pattern[center, :] = 0
    pattern[:, center] = 0
    
    # Normalize to [0, 1]
    template = (pattern + 1) / 2
    
    # Apply Gaussian blur for realistic transitions
    sigma = size / 15.0
    template = cv2.GaussianBlur(template, (0, 0), sigmaX=sigma)
    
    return template


def find_cross_centers_template_matching(
    gray: np.ndarray,
    template_size: int = 41,
    threshold: float = 0.5,
    min_distance: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find cross template centers using normalized cross-correlation template matching.
    
    Args:
        gray: Grayscale image (uint8 or float)
        template_size: Size of the cross template
        threshold: Correlation threshold for detection
        min_distance: Minimum distance between detected crosses
    
    Returns:
        Tuple of (points_xy, correlation_map)
        - points_xy: Nx2 array of (x, y) coordinates at integer precision
        - correlation_map: The template matching response map
    """
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0
    
    # Create the cross template
    template = create_saddle_cross_template(template_size)
    
    # Perform normalized cross-correlation template matching
    # TM_CCOEFF_NORMED gives values in [-1, 1], where 1 is perfect match
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    
    # Also try the inverted template (in case checkerboard polarity is opposite)
    template_inv = 1.0 - template
    result_inv = cv2.matchTemplate(gray, template_inv, cv2.TM_CCOEFF_NORMED)
    
    # Take the maximum response from both templates
    result = np.maximum(result, result_inv)
    
    # Non-maximum suppression using local maximum filter
    footprint_size = 2 * min_distance + 1
    local_max = maximum_filter(result, size=footprint_size)
    
    # Find peaks: where result equals local max and exceeds threshold
    peaks = (result == local_max) & (result >= threshold)
    
    # Get coordinates of peaks
    ys, xs = np.where(peaks)
    
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32), result
    
    # Adjust coordinates to account for template offset
    offset = template_size // 2
    xs = xs + offset
    ys = ys + offset
    
    # Sort by correlation strength (descending)
    scores = result[ys - offset, xs - offset]
    order = np.argsort(scores)[::-1]
    
    points = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
    
    return points, result


def refine_subpixel_quadratic(
    correlation_map: np.ndarray,
    points_xy: np.ndarray,
    template_size: int,
) -> np.ndarray:
    """
    Refine integer-pixel locations to subpixel using quadratic fitting on correlation map.
    
    Uses Taylor expansion / Newton step on a 3x3 neighborhood.
    
    Args:
        correlation_map: The template matching correlation response
        points_xy: Nx2 array of (x, y) integer coordinates
        template_size: Size of template used (to compute offset)
    
    Returns:
        Refined Nx2 array of (x, y) subpixel coordinates
    """
    if points_xy.size == 0:
        return points_xy
    
    h, w = correlation_map.shape
    offset = template_size // 2
    pts = points_xy.astype(np.float64).copy()
    
    for i in range(pts.shape[0]):
        x, y = pts[i, 0], pts[i, 1]
        
        # Convert to correlation map coordinates
        cx = int(round(x)) - offset
        cy = int(round(y)) - offset
        
        # Check bounds for 3x3 neighborhood
        if cx < 1 or cx >= w - 1 or cy < 1 or cy >= h - 1:
            continue
        
        # Extract 3x3 patch
        patch = correlation_map[cy - 1 : cy + 2, cx - 1 : cx + 2].astype(np.float64)
        
        # Compute gradient using central differences
        gx = 0.5 * (patch[1, 2] - patch[1, 0])
        gy = 0.5 * (patch[2, 1] - patch[0, 1])
        
        # Compute Hessian
        gxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
        gyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
        gxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])
        
        # Solve for subpixel offset: H * delta = -grad
        hess = np.array([[gxx, gxy], [gxy, gyy]])
        grad = np.array([gx, gy])
        
        det = gxx * gyy - gxy * gxy
        if abs(det) < 1e-12:
            continue
        
        try:
            delta = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            continue
        
        # Clamp to reasonable range (within pixel)
        dx, dy = delta[0], delta[1]
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            continue
        
        # Apply offset (remember to add back the template offset)
        pts[i, 0] = (cx + dx) + offset
        pts[i, 1] = (cy + dy) + offset
    
    return pts.astype(np.float32)


def refine_with_corner_subpix(
    gray_u8: np.ndarray,
    points_xy: np.ndarray,
    win_size: int = 11,
) -> np.ndarray:
    """
    Refine points using OpenCV's cornerSubPix for accurate subpixel localization.
    
    Args:
        gray_u8: Grayscale image (uint8)
        points_xy: Nx2 array of (x, y) coordinates
        win_size: Half-size of search window
    
    Returns:
        Refined Nx2 array of subpixel coordinates
    """
    if points_xy.size == 0:
        return points_xy
    
    # Ensure uint8
    if gray_u8.dtype != np.uint8:
        gray_u8 = (gray_u8 * 255).astype(np.uint8) if gray_u8.max() <= 1.0 else gray_u8.astype(np.uint8)
    
    # OpenCV cornerSubPix expects Nx1x2 float32
    corners = points_xy.reshape(-1, 1, 2).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    refined = cv2.cornerSubPix(
        gray_u8,
        corners,
        winSize=(win_size, win_size),
        zeroZone=(-1, -1),
        criteria=criteria
    )
    
    return refined.reshape(-1, 2)


def detect_saddle_points_hessian(
    gray_u8: np.ndarray,
    blur_sigma: float = 1.0,
    nms_radius: int = 8,
    rel_threshold: float = 0.1,
    max_points: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect saddle points using Hessian determinant (det(H) < 0 for saddle).
    
    This is effective for finding checkerboard cross patterns where 
    intensity has opposite curvatures in orthogonal directions.
    
    Args:
        gray_u8: Grayscale image (uint8)
        blur_sigma: Gaussian blur sigma for noise reduction
        nms_radius: Non-maximum suppression radius
        rel_threshold: Relative threshold (fraction of max response)
        max_points: Maximum number of points to return
    
    Returns:
        Tuple of (points_xy, saddle_response)
    """
    # Convert to float
    gray = gray_u8.astype(np.float32) / 255.0
    
    # Apply Gaussian smoothing
    if blur_sigma > 0:
        gray = gaussian_filter(gray, sigma=blur_sigma)
    
    # Compute second derivatives using Sobel
    Ixx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    
    # Hessian determinant: det(H) = Ixx*Iyy - Ixy^2
    # For saddle points, det(H) < 0
    det_H = Ixx * Iyy - Ixy * Ixy
    
    # Saddle response: stronger negative determinant = stronger saddle
    saddle_response = np.maximum(0.0, -det_H)
    
    # Threshold based on relative to maximum
    max_response = float(np.max(saddle_response))
    if max_response <= 0:
        return np.zeros((0, 2), dtype=np.float32), saddle_response
    
    threshold = max_response * rel_threshold
    
    # Non-maximum suppression
    footprint_size = 2 * nms_radius + 1
    local_max = maximum_filter(saddle_response, size=footprint_size)
    
    # Find peaks: where response equals local max and exceeds threshold
    peaks = (saddle_response == local_max) & (saddle_response >= threshold)
    ys, xs = np.where(peaks)
    
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32), saddle_response
    
    # Sort by response strength
    scores = saddle_response[ys, xs]
    order = np.argsort(scores)[::-1]
    
    if len(order) > max_points:
        order = order[:max_points]
    
    points = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
    
    return points, saddle_response


def detect_cross_templates(
    image_path: str,
    method: str = "combined",
    template_size: int = 41,
    threshold: float = 0.5,
    min_distance: int = 20,
    use_corner_subpix: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main function to detect cross templates in an image with subpixel accuracy.
    
    Args:
        image_path: Path to the input image
        method: Detection method - 'template', 'saddle', or 'combined'
        template_size: Size of the cross template
        threshold: Detection threshold
        min_distance: Minimum distance between detected crosses
        use_corner_subpix: Use OpenCV cornerSubPix for refinement
    
    Returns:
        Tuple of (subpixel_points, bgr_image, response_map)
    """
    # Read image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    if method == "template":
        # Template matching only
        points_xy, response_map = find_cross_centers_template_matching(
            gray,
            template_size=template_size,
            threshold=threshold,
            min_distance=min_distance,
        )
    elif method == "saddle":
        # Saddle point detection only
        points_xy, response_map = detect_saddle_points_hessian(
            gray,
            blur_sigma=1.0,
            nms_radius=min_distance // 2,
            rel_threshold=threshold,
        )
    else:  # combined
        # Use saddle point detection (more robust for checkerboard)
        points_xy, response_map = detect_saddle_points_hessian(
            gray,
            blur_sigma=0.8,
            nms_radius=min_distance // 2,
            rel_threshold=0.15,
        )
    
    if points_xy.size == 0:
        print("No cross templates found.")
        return np.zeros((0, 2), dtype=np.float32), bgr, response_map
    
    # Filter points that are too close to image edges
    h, w = gray.shape
    margin = 5
    mask = (points_xy[:, 0] >= margin) & (points_xy[:, 0] < w - margin) & \
           (points_xy[:, 1] >= margin) & (points_xy[:, 1] < h - margin)
    points_xy = points_xy[mask]
    
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), bgr, response_map
    
    # Refine to subpixel using OpenCV's cornerSubPix
    if use_corner_subpix:
        subpixel_points = refine_with_corner_subpix(gray, points_xy, win_size=11)
    else:
        subpixel_points = points_xy.copy()
    
    return subpixel_points, bgr, response_map


def draw_points(
    bgr: np.ndarray,
    points_xy: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 5,
    show_coords: bool = False,
) -> np.ndarray:
    """Draw detected points on the image."""
    out = bgr.copy()
    
    for idx, (x, y) in enumerate(points_xy.reshape(-1, 2)):
        # Draw crosshair
        cx, cy = int(round(x)), int(round(y))
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, radius * 2, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), radius, color, 1, cv2.LINE_AA)
        
        if show_coords and idx < 50:  # Only show first 50 to avoid clutter
            text = f"({x:.2f},{y:.2f})"
            cv2.putText(out, text, (cx + 10, cy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find cross templates in image with subpixel accuracy"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--method", choices=["template", "saddle", "combined"], 
                       default="saddle",
                       help="Detection method (default: saddle)")
    parser.add_argument("--template-size", type=int, default=41,
                       help="Size of cross template for template method (default: 41)")
    parser.add_argument("--threshold", type=float, default=0.01,
                       help="Detection threshold - relative to max response (default: 0.01)")
    parser.add_argument("--min-distance", type=int, default=15,
                       help="Minimum distance between detections (default: 15)")
    parser.add_argument("--out", default=None,
                       help="Output image path (default: auto)")
    parser.add_argument("--csv", default=None,
                       help="Output CSV file for coordinates (default: auto)")
    parser.add_argument("--no-corner-subpix", action="store_true",
                       help="Disable OpenCV cornerSubPix refinement")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print all detected coordinates")
    
    args = parser.parse_args()
    
    # Detect cross templates
    print(f"Processing: {args.image}")
    print(f"Method: {args.method}")
    points, bgr, response_map = detect_cross_templates(
        args.image,
        method=args.method,
        template_size=args.template_size,
        threshold=args.threshold,
        min_distance=args.min_distance,
        use_corner_subpix=not args.no_corner_subpix,
    )
    
    print(f"Found {len(points)} cross template centers with subpixel coordinates:")
    
    # Print coordinates
    if args.verbose or len(points) <= 20:
        for i, (x, y) in enumerate(points):
            print(f"  Point {i+1}: ({x:.4f}, {y:.4f})")
    else:
        # Print first and last few
        for i, (x, y) in enumerate(points[:5]):
            print(f"  Point {i+1}: ({x:.4f}, {y:.4f})")
        print(f"  ... ({len(points) - 10} more points) ...")
        for i, (x, y) in enumerate(points[-5:], len(points) - 4):
            print(f"  Point {i}: ({x:.4f}, {y:.4f})")
    
    # Save CSV
    csv_path = args.csv
    if csv_path is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        csv_path = f"cross_centers_{stem}.csv"
    
    with open(csv_path, 'w') as f:
        f.write("index,x,y\n")
        for i, (x, y) in enumerate(points):
            f.write(f"{i},{x:.6f},{y:.6f}\n")
    print(f"Coordinates saved to: {csv_path}")
    
    # Save visualization
    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = f"cross_detected_{stem}.png"
    
    vis = draw_points(bgr, points, color=(0, 255, 0), radius=8)
    cv2.imwrite(out_path, vis)
    print(f"Visualization saved to: {out_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
