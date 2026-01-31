"""
Cross template detection with subpixel accuracy.

This script finds cross markers in images using template matching
and refines the center coordinates to subpixel precision.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class CrossDetection:
    """Represents a detected cross marker with subpixel coordinates."""
    x: float  # x coordinate (subpixel)
    y: float  # y coordinate (subpixel)
    score: float  # correlation score
    
    def __str__(self) -> str:
        return f"Cross at ({self.x:.4f}, {self.y:.4f}), score={self.score:.4f}"


def create_cross_template(
    size: int = 21,
    line_width: int = 3,
    bg_val: float = 180.0,
    cross_val: float = 40.0,
) -> np.ndarray:
    """
    Create a cross template for template matching.
    
    Args:
        size: Template size (should be odd)
        line_width: Width of cross lines
        bg_val: Background intensity value
        cross_val: Cross line intensity value
        
    Returns:
        Template image as float32 array
    """
    if size % 2 == 0:
        size += 1
    
    template = np.full((size, size), bg_val, dtype=np.float32)
    center = size // 2
    half_lw = line_width // 2
    
    # Draw horizontal line
    template[center - half_lw:center + half_lw + 1, :] = cross_val
    # Draw vertical line  
    template[:, center - half_lw:center + half_lw + 1] = cross_val
    
    return template


def multi_scale_template_match(
    gray: np.ndarray,
    template: np.ndarray,
    scales: List[float] = [0.8, 0.9, 1.0, 1.1, 1.2],
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Tuple[np.ndarray, float]:
    """
    Perform template matching at multiple scales.
    
    Args:
        gray: Grayscale input image
        template: Template image
        scales: List of scale factors to try
        method: OpenCV template matching method
        
    Returns:
        Best result map and corresponding scale factor
    """
    best_result = None
    best_scale = 1.0
    best_max = -np.inf
    
    for scale in scales:
        if scale != 1.0:
            scaled_template = cv2.resize(
                template, None, fx=scale, fy=scale, 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_template = template
            
        if scaled_template.shape[0] >= gray.shape[0] or scaled_template.shape[1] >= gray.shape[1]:
            continue
            
        result = cv2.matchTemplate(gray.astype(np.float32), scaled_template, method)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_max:
            best_max = max_val
            best_result = result
            best_scale = scale
    
    return best_result, best_scale


def non_maximum_suppression(
    detections: List[Tuple[float, float, float]],
    min_distance: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """
    Apply non-maximum suppression to detections.
    
    Args:
        detections: List of (x, y, score) tuples
        min_distance: Minimum distance between detections
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort by score (descending)
    sorted_dets = sorted(detections, key=lambda x: -x[2])
    
    kept = []
    for x, y, score in sorted_dets:
        # Check if too close to any kept detection
        is_duplicate = False
        for kx, ky, _ in kept:
            dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
            if dist < min_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append((x, y, score))
    
    return kept


def refine_subpixel_quadratic(
    correlation_map: np.ndarray,
    x: int,
    y: int,
) -> Tuple[float, float]:
    """
    Refine integer coordinates to subpixel using quadratic interpolation.
    
    Uses a 3x3 neighborhood to fit a quadratic surface and find the peak.
    
    Args:
        correlation_map: The correlation/response map
        x, y: Integer coordinates of the peak
        
    Returns:
        Subpixel (x, y) coordinates
    """
    h, w = correlation_map.shape
    
    # Ensure we have valid neighbors
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return float(x), float(y)
    
    # Extract 3x3 patch
    patch = correlation_map[y - 1:y + 2, x - 1:x + 2].astype(np.float64)
    
    # Compute gradient using central differences
    gx = 0.5 * (patch[1, 2] - patch[1, 0])
    gy = 0.5 * (patch[2, 1] - patch[0, 1])
    
    # Compute Hessian
    gxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
    gyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
    gxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])
    
    # Solve for subpixel offset: H * delta = -g
    hess = np.array([[gxx, gxy], [gxy, gyy]], dtype=np.float64)
    grad = np.array([gx, gy], dtype=np.float64)
    
    det = np.linalg.det(hess)
    if not np.isfinite(det) or abs(det) < 1e-10:
        return float(x), float(y)
    
    try:
        delta = -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        return float(x), float(y)
    
    # Clamp to reasonable range
    dx, dy = float(delta[0]), float(delta[1])
    if abs(dx) > 1.0 or abs(dy) > 1.0:
        return float(x), float(y)
    
    return x + dx, y + dy


def refine_subpixel_gaussian(
    gray: np.ndarray,
    x: int,
    y: int,
    window_size: int = 11,
) -> Tuple[float, float]:
    """
    Refine coordinates using Gaussian fitting on intensity values.
    
    This is useful when the cross center is at a local extremum.
    
    Args:
        gray: Grayscale image
        x, y: Integer coordinates
        window_size: Size of fitting window
        
    Returns:
        Subpixel (x, y) coordinates
    """
    h, w = gray.shape
    half = window_size // 2
    
    if x < half or x >= w - half or y < half or y >= h - half:
        return float(x), float(y)
    
    # Extract window
    window = gray[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
    
    # Compute centroid weighted by intensity
    # For dark cross, invert the intensities
    weights = window.max() - window
    weights = weights ** 2  # Square to emphasize
    
    ys, xs = np.mgrid[0:window_size, 0:window_size]
    total_weight = weights.sum()
    
    if total_weight < 1e-10:
        return float(x), float(y)
    
    cx = (xs * weights).sum() / total_weight
    cy = (ys * weights).sum() / total_weight
    
    # Convert back to image coordinates
    sub_x = x - half + cx
    sub_y = y - half + cy
    
    return sub_x, sub_y


def detect_cross_markers(
    gray: np.ndarray,
    template_sizes: List[int] = [15, 19, 23],
    line_widths: List[int] = [2, 3],
    threshold: float = 0.4,
    nms_distance: float = 30.0,
    subpixel_method: str = "quadratic",
) -> List[CrossDetection]:
    """
    Detect cross markers in a grayscale image.
    
    Args:
        gray: Grayscale input image
        template_sizes: List of template sizes to try
        line_widths: List of line widths to try
        threshold: Detection threshold (correlation coefficient)
        nms_distance: Minimum distance between detections for NMS
        subpixel_method: "quadratic" or "gaussian"
        
    Returns:
        List of CrossDetection objects with subpixel coordinates
    """
    gray_float = gray.astype(np.float32)
    all_detections = []
    
    for size in template_sizes:
        for lw in line_widths:
            if lw >= size // 3:
                continue
            
            # Try both dark-on-light and light-on-dark crosses
            for bg_val, cross_val in [(180.0, 40.0), (40.0, 180.0)]:
                template = create_cross_template(size, lw, bg_val, cross_val)
                
                # Template matching
                result = cv2.matchTemplate(gray_float, template, cv2.TM_CCOEFF_NORMED)
                
                # Find local maxima above threshold
                local_max = cv2.dilate(result, np.ones((5, 5)))
                peaks = (result == local_max) & (result >= threshold)
                ys, xs = np.where(peaks)
                
                half = size // 2
                for px, py in zip(xs, ys):
                    score = result[py, px]
                    # Convert to center coordinates
                    center_x = px + half
                    center_y = py + half
                    all_detections.append((center_x, center_y, score))
    
    # Apply NMS
    filtered = non_maximum_suppression(all_detections, nms_distance)
    
    # Refine to subpixel
    detections = []
    for x, y, score in filtered:
        # Get best correlation map for refinement
        best_size = template_sizes[len(template_sizes) // 2]
        template = create_cross_template(best_size, line_widths[0])
        result = cv2.matchTemplate(gray_float, template, cv2.TM_CCOEFF_NORMED)
        
        half = best_size // 2
        map_x = int(x - half)
        map_y = int(y - half)
        
        if subpixel_method == "quadratic":
            sub_x, sub_y = refine_subpixel_quadratic(result, map_x, map_y)
            sub_x += half
            sub_y += half
        else:
            sub_x, sub_y = refine_subpixel_gaussian(gray, int(x), int(y))
        
        detections.append(CrossDetection(sub_x, sub_y, score))
    
    return detections


def detect_cross_markers_multiscale(
    image_path: str,
    downsample_factor: float = 0.25,
    template_size: int = 11,
    line_width: int = 2,
    threshold: float = 0.4,
    refine_at_full_res: bool = True,
) -> List[CrossDetection]:
    """
    Detect cross markers using multi-scale approach for efficiency.
    
    First detects at low resolution, then refines at full resolution.
    
    Args:
        image_path: Path to input image
        downsample_factor: Factor for initial downsampling
        template_size: Size of cross template (at downsampled resolution)
        line_width: Width of cross lines
        threshold: Detection threshold
        refine_at_full_res: Whether to refine at full resolution
        
    Returns:
        List of CrossDetection objects with subpixel coordinates
    """
    # Load image
    gray_full = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_full is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Downsample for initial detection
    gray_small = cv2.resize(gray_full, None, fx=downsample_factor, fy=downsample_factor)
    
    # Create templates for both polarities
    detections = []
    
    for bg_val, cross_val in [(180.0, 40.0), (40.0, 180.0)]:
        template = create_cross_template(template_size, line_width, bg_val, cross_val)
        
        # Template matching
        result = cv2.matchTemplate(
            gray_small.astype(np.float32), 
            template, 
            cv2.TM_CCOEFF_NORMED
        )
        
        # Find peaks
        local_max = cv2.dilate(result, np.ones((5, 5)))
        peaks = (result == local_max) & (result >= threshold)
        ys, xs = np.where(peaks)
        
        half = template_size // 2
        for px, py in zip(xs, ys):
            score = result[py, px]
            # Convert to center at downsampled resolution
            small_x = px + half
            small_y = py + half
            # Scale to full resolution
            full_x = small_x / downsample_factor
            full_y = small_y / downsample_factor
            detections.append((full_x, full_y, score))
    
    # Apply NMS at full resolution scale
    nms_distance = template_size / downsample_factor
    filtered = non_maximum_suppression(detections, nms_distance)
    
    # Refine at full resolution
    final_detections = []
    full_template_size = int(template_size / downsample_factor)
    if full_template_size % 2 == 0:
        full_template_size += 1
    full_line_width = max(2, int(line_width / downsample_factor))
    
    for x, y, score in filtered:
        if refine_at_full_res:
            # Create template at full resolution
            template = create_cross_template(full_template_size, full_line_width, 180.0, 40.0)
            
            # Extract ROI for local template matching
            roi_size = full_template_size * 3
            roi_half = roi_size // 2
            
            xi, yi = int(x), int(y)
            y1 = max(0, yi - roi_half)
            y2 = min(gray_full.shape[0], yi + roi_half)
            x1 = max(0, xi - roi_half)
            x2 = min(gray_full.shape[1], xi + roi_half)
            
            roi = gray_full[y1:y2, x1:x2].astype(np.float32)
            
            if roi.shape[0] > template.shape[0] and roi.shape[1] > template.shape[1]:
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # Subpixel refinement on correlation map
                half = full_template_size // 2
                local_x = max_loc[0] + half
                local_y = max_loc[1] + half
                
                sub_x, sub_y = refine_subpixel_quadratic(result, max_loc[0], max_loc[1])
                sub_x += half
                sub_y += half
                
                # Convert back to image coordinates
                final_x = x1 + sub_x
                final_y = y1 + sub_y
                
                final_detections.append(CrossDetection(final_x, final_y, max_val))
            else:
                final_detections.append(CrossDetection(x, y, score))
        else:
            final_detections.append(CrossDetection(x, y, score))
    
    return final_detections


def visualize_detections(
    image_path: str,
    detections: List[CrossDetection],
    output_path: str,
    marker_size: int = 10,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> None:
    """
    Visualize detected cross markers on the image.
    
    Args:
        image_path: Path to input image
        detections: List of CrossDetection objects
        output_path: Path to save visualization
        marker_size: Size of visualization markers
        color: BGR color for markers
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    for det in detections:
        x, y = int(round(det.x)), int(round(det.y))
        # Draw crosshair
        cv2.line(img, (x - marker_size, y), (x + marker_size, y), color, 2)
        cv2.line(img, (x, y - marker_size), (x, y + marker_size), color, 2)
        # Draw circle
        cv2.circle(img, (x, y), marker_size, color, 2)
    
    cv2.imwrite(output_path, img)


def refine_cross_center_subpixel(
    gray: np.ndarray,
    x: int,
    y: int,
    window_size: int = 31,
) -> Tuple[float, float]:
    """
    Refine cross center to subpixel accuracy using intensity-weighted centroid
    along the cross arms.
    
    This method fits the center by analyzing the horizontal and vertical
    intensity profiles through the cross.
    
    Args:
        gray: Grayscale image
        x, y: Initial integer coordinates
        window_size: Size of analysis window
        
    Returns:
        Subpixel (x, y) coordinates
    """
    h, w = gray.shape
    half = window_size // 2
    
    if x < half or x >= w - half or y < half or y >= h - half:
        return float(x), float(y)
    
    # Extract window
    window = gray[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
    
    # For a dark cross on light background, find the minimum along profiles
    # Extract horizontal profile through center
    h_profile = window[half, :]
    # Extract vertical profile through center
    v_profile = window[:, half]
    
    # Compute weighted centroid for each profile
    # Use inverted intensity as weights (dark = high weight)
    h_weights = h_profile.max() - h_profile
    v_weights = v_profile.max() - v_profile
    
    # Apply a threshold to focus on the cross
    h_thresh = np.percentile(h_weights, 80)
    v_thresh = np.percentile(v_weights, 80)
    h_weights[h_weights < h_thresh] = 0
    v_weights[v_weights < v_thresh] = 0
    
    positions = np.arange(window_size)
    
    # Compute centroid
    h_sum = h_weights.sum()
    v_sum = v_weights.sum()
    
    if h_sum > 0 and v_sum > 0:
        h_centroid = (positions * h_weights).sum() / h_sum
        v_centroid = (positions * v_weights).sum() / v_sum
        
        # Convert to image coordinates
        sub_x = x - half + h_centroid
        sub_y = y - half + v_centroid
        
        # Sanity check - should be close to original
        if abs(sub_x - x) > 2 or abs(sub_y - y) > 2:
            return float(x), float(y)
            
        return sub_x, sub_y
    
    return float(x), float(y)


def find_cross_center_precise(
    gray: np.ndarray,
    approx_x: int,
    approx_y: int,
    search_radius: int = 30,
) -> Tuple[float, float]:
    """
    Find the precise subpixel center of a cross marker.
    
    The cross center is detected as the darkest point, then refined
    using parabolic fitting on horizontal and vertical intensity profiles.
    
    Args:
        gray: Grayscale image
        approx_x, approx_y: Approximate center coordinates
        search_radius: Search radius around approximate location
        
    Returns:
        (x, y) subpixel coordinates of cross center
    """
    h, w = gray.shape
    
    # Define search region
    y1 = max(0, approx_y - search_radius)
    y2 = min(h, approx_y + search_radius)
    x1 = max(0, approx_x - search_radius)
    x2 = min(w, approx_x + search_radius)
    
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(patch, (3, 3), 0.8)
    
    # Find the darkest point as initial estimate
    _, _, min_loc, _ = cv2.minMaxLoc(blurred)
    init_x, init_y = min_loc
    
    # Ensure we have enough margin for parabolic fit
    window = 8
    if init_x < window or init_x >= patch.shape[1] - window:
        init_x = patch.shape[1] // 2
    if init_y < window or init_y >= patch.shape[0] - window:
        init_y = patch.shape[0] // 2
    
    # Extract profiles through the center
    h_profile = blurred[init_y, init_x - window:init_x + window + 1]
    v_profile = blurred[init_y - window:init_y + window + 1, init_x]
    
    def fit_parabola(profile: np.ndarray) -> float:
        """Fit parabola to profile and find minimum location."""
        n = len(profile)
        min_idx = int(np.argmin(profile))
        
        if min_idx < 2 or min_idx > n - 3:
            return float(n // 2)
        
        # Fit using 5 points around minimum
        indices = np.arange(min_idx - 2, min_idx + 3, dtype=np.float64)
        values = profile[min_idx - 2:min_idx + 3].astype(np.float64)
        
        # Fit y = a*x^2 + b*x + c
        A = np.column_stack([indices ** 2, indices, np.ones(5)])
        try:
            coeffs = np.linalg.lstsq(A, values, rcond=None)[0]
            a, b, _ = coeffs
            if a > 1e-6:  # Valid parabola with minimum
                subpix = -b / (2 * a)
                if abs(subpix - min_idx) < 2:
                    return subpix
        except np.linalg.LinAlgError:
            pass
        return float(min_idx)
    
    sub_x_local = fit_parabola(h_profile)
    sub_y_local = fit_parabola(v_profile)
    
    # Convert to full image coordinates
    sub_x = x1 + init_x - window + sub_x_local
    sub_y = y1 + init_y - window + sub_y_local
    
    return sub_x, sub_y


def detect_cross_high_precision(
    image_path: str,
    min_score: float = 0.5,
) -> List[CrossDetection]:
    """
    Detect cross markers with high precision and subpixel accuracy.
    
    Uses multi-scale template matching followed by precise subpixel refinement
    based on parabolic fitting of intensity profiles.
    
    Args:
        image_path: Path to input image
        min_score: Minimum correlation score for valid detection
        
    Returns:
        List of CrossDetection objects with subpixel coordinates
    """
    # Load image
    gray_full = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_full is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    h, w = gray_full.shape
    
    # Multi-scale detection
    all_detections = []
    
    for scale in [0.125, 0.25, 0.5]:
        gray_small = cv2.resize(gray_full, None, fx=scale, fy=scale)
        
        # Determine template size based on scale
        if scale <= 0.125:
            template_size = 7
            line_width = 1
        elif scale <= 0.25:
            template_size = 11
            line_width = 2
        else:
            template_size = 21
            line_width = 3
        
        # Try both polarities (dark cross on light, light cross on dark)
        for bg_val, cross_val in [(180.0, 40.0), (60.0, 160.0)]:
            template = create_cross_template(template_size, line_width, bg_val, cross_val)
            
            result = cv2.matchTemplate(
                gray_small.astype(np.float32), 
                template, 
                cv2.TM_CCOEFF_NORMED
            )
            
            # Find peaks
            local_max = cv2.dilate(result, np.ones((5, 5)))
            peaks = (result == local_max) & (result >= min_score * 0.8)
            ys, xs = np.where(peaks)
            
            half = template_size // 2
            for px, py in zip(xs, ys):
                score = result[py, px]
                full_x = (px + half) / scale
                full_y = (py + half) / scale
                all_detections.append((full_x, full_y, score, scale))
    
    # Apply NMS
    nms_distance = 50.0
    all_detections.sort(key=lambda x: -x[2])
    
    kept = []
    for x, y, score, scale in all_detections:
        is_duplicate = False
        for kx, ky, _, _ in kept:
            if np.sqrt((x - kx) ** 2 + (y - ky) ** 2) < nms_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((x, y, score, scale))
    
    # Refine each detection at full resolution
    final_detections = []
    
    for x, y, score, _ in kept:
        if score < min_score:
            continue
            
        xi, yi = int(round(x)), int(round(y))
        
        # Local template matching at full resolution for accurate localization
        roi_size = 100
        roi_half = roi_size // 2
        
        y1 = max(0, yi - roi_half)
        y2 = min(h, yi + roi_half)
        x1 = max(0, xi - roi_half)
        x2 = min(w, xi + roi_half)
        
        roi = gray_full[y1:y2, x1:x2].astype(np.float32)
        
        # Try multiple template sizes at full resolution
        best_score = 0
        best_loc = (xi, yi)
        
        for ts in [31, 41, 51]:
            for lw in [3, 5, 7]:
                template = create_cross_template(ts, lw, 180.0, 40.0)
                if roi.shape[0] <= ts or roi.shape[1] <= ts:
                    continue
                    
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    half = ts // 2
                    best_loc = (x1 + max_loc[0] + half, y1 + max_loc[1] + half)
        
        # Precise subpixel refinement using parabolic fitting
        sub_x, sub_y = find_cross_center_precise(
            gray_full, 
            int(round(best_loc[0])), 
            int(round(best_loc[1])),
            search_radius=20
        )
        
        final_detections.append(CrossDetection(sub_x, sub_y, best_score))
    
    return final_detections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect cross markers in images with subpixel accuracy."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (0-1)")
    parser.add_argument("--downsample", type=float, default=0.25,
                        help="Downsample factor for initial detection")
    parser.add_argument("--template-size", type=int, default=11,
                        help="Cross template size (at downsampled resolution)")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Cross line width")
    parser.add_argument("--out", default=None,
                        help="Output visualization path")
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip full-resolution refinement")
    parser.add_argument("--high-precision", action="store_true",
                        help="Use high-precision multi-scale detection")
    
    args = parser.parse_args()
    
    print(f"Processing: {args.image}")
    
    if args.high_precision:
        print(f"Using high-precision mode with threshold={args.threshold}")
        detections = detect_cross_high_precision(args.image, min_score=args.threshold)
    else:
        print(f"Parameters: threshold={args.threshold}, downsample={args.downsample}")
        detections = detect_cross_markers_multiscale(
            args.image,
            downsample_factor=args.downsample,
            template_size=args.template_size,
            line_width=args.line_width,
            threshold=args.threshold,
            refine_at_full_res=not args.no_refine,
        )
    
    print(f"\nFound {len(detections)} cross marker(s):")
    for i, det in enumerate(detections):
        print(f"  {i + 1}. {det}")
    
    # Save visualization
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = f"cross_detections_{stem}.png"
    else:
        out_path = args.out
    
    visualize_detections(args.image, detections, out_path)
    print(f"\nVisualization saved: {out_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
