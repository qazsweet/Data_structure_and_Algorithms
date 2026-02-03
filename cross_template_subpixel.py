"""
Cross Template Detection with Subpixel Center Estimation

This module provides functions to detect cross-shaped fiducial markers in
microscopy or industrial images and compute their center coordinates with
subpixel accuracy.

The detection algorithm:
1. Uses adaptive thresholding to segment the image
2. Identifies cross-shaped contours based on geometric properties (solidity, convexity defects)
3. Refines the center location using edge-weighted centroid computation

Usage:
    python cross_template_subpixel.py <image_path> [--debug]
"""

import cv2
import numpy as np
import argparse
import sys
import os
from typing import Optional, Tuple, List, Dict, Any


def find_cross_candidates(
    gray: np.ndarray,
    min_area: int = 5000,
    max_area: int = 1000000,
    min_solidity: float = 0.30,
    max_solidity: float = 0.65,
    min_defects: int = 3,
    max_defects: int = 6
) -> List[Dict[str, Any]]:
    """
    Find all cross-shaped contour candidates in the image.
    
    Args:
        gray: Grayscale input image
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        min_solidity: Minimum solidity (area/convex_hull_area) ratio
        max_solidity: Maximum solidity ratio
        min_defects: Minimum number of significant convexity defects
        max_defects: Maximum number of significant convexity defects
    
    Returns:
        List of candidate dictionaries with keys:
            - 'contour': The contour points
            - 'center': (x, y) tuple of centroid
            - 'area': Contour area
            - 'solidity': Solidity ratio
            - 'defects': Number of significant convexity defects
            - 'score': Quality score
    """
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)
    
    candidates = []
    
    # Try multiple thresholding parameters
    for block_size in [51, 101, 201, 301]:
        for c_val in [2, 5, 10, 15, 20]:
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_val
            )
            
            # Try both binary and inverted
            for test_img in [binary, 255 - binary]:
                contours, _ = cv2.findContours(
                    test_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area or area > max_area:
                        continue
                    
                    # Check aspect ratio (cross should be roughly square)
                    x, y, bw, bh = cv2.boundingRect(contour)
                    aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
                    if aspect < 0.5:
                        continue
                    
                    # Check solidity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area == 0:
                        continue
                    
                    solidity = area / hull_area
                    if not (min_solidity < solidity < max_solidity):
                        continue
                    
                    # Check convexity defects
                    hull_indices = cv2.convexHull(contour, returnPoints=False)
                    if hull_indices is None or len(hull_indices) < 4:
                        continue
                    
                    try:
                        defects = cv2.convexityDefects(contour, hull_indices)
                        if defects is None:
                            continue
                        
                        # Count significant defects (depth > 10 pixels)
                        sig_defects = sum(1 for d in defects if d[0, 3] / 256.0 > 10)
                        
                        if min_defects <= sig_defects <= max_defects:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]
                                
                                # Score based on area, solidity closeness to 0.45, and defect count
                                score = area * (1 - abs(solidity - 0.45) * 3) * min(sig_defects / 4, 1.2)
                                
                                candidates.append({
                                    'contour': contour,
                                    'center': (cx, cy),
                                    'area': area,
                                    'solidity': solidity,
                                    'defects': sig_defects,
                                    'score': score
                                })
                    except cv2.error:
                        continue
    
    # Remove duplicates (keep highest scoring for each location)
    unique_candidates = []
    for c in candidates:
        is_duplicate = False
        for i, u in enumerate(unique_candidates):
            dist = np.sqrt(
                (c['center'][0] - u['center'][0])**2 + 
                (c['center'][1] - u['center'][1])**2
            )
            if dist < 100:  # Within 100 pixels = same cross
                if c['score'] > u['score']:
                    unique_candidates[i] = c
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(c)
    
    return sorted(unique_candidates, key=lambda x: x['score'], reverse=True)


def refine_center_subpixel(
    gray: np.ndarray,
    center: Tuple[float, float],
    window_size: int = 150
) -> Tuple[float, float]:
    """
    Refine cross center to subpixel accuracy using edge-weighted centroid.
    
    Args:
        gray: Grayscale input image
        center: Initial (x, y) center estimate
        window_size: Size of the window around center to analyze
    
    Returns:
        Refined (x, y) center coordinates with subpixel precision
    """
    h, w = gray.shape
    cx, cy = center
    
    # Extract ROI
    x1 = max(0, int(cx - window_size))
    x2 = min(w, int(cx + window_size))
    y1 = max(0, int(cy - window_size))
    y2 = min(h, int(cy + window_size))
    
    local = gray[y1:y2, x1:x2].astype(np.float64)
    
    # Compute gradient magnitude
    gy, gx = np.gradient(local)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Edge detection
    edges = cv2.Canny(local.astype(np.uint8), 20, 80)
    edge_pts = np.where(edges > 0)
    
    if len(edge_pts[0]) < 10:
        return center
    
    # Compute edge-weighted centroid
    # Weight by squared gradient magnitude for stronger weighting of prominent edges
    weights = grad_mag[edge_pts[0], edge_pts[1]] ** 2
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        refined_x = np.sum(edge_pts[1] * weights) / total_weight + x1
        refined_y = np.sum(edge_pts[0] * weights) / total_weight + y1
        return (refined_x, refined_y)
    
    return center


def detect_cross_center(
    image: np.ndarray,
    return_contour: bool = False
) -> Optional[Tuple[float, float]]:
    """
    Detect cross template and compute subpixel center coordinates.
    
    Args:
        image: Input image (grayscale or BGR)
        return_contour: If True, also return the detected contour
    
    Returns:
        (x, y) tuple of subpixel center coordinates, or None if no cross found
        If return_contour is True, returns ((x, y), contour) or (None, None)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find candidates
    candidates = find_cross_candidates(gray)
    
    if not candidates:
        if return_contour:
            return None, None
        return None
    
    # Get best candidate
    best = candidates[0]
    
    # Refine center to subpixel
    refined_center = refine_center_subpixel(gray, best['center'])
    
    if return_contour:
        return refined_center, best['contour']
    return refined_center


def visualize_detection(
    image: np.ndarray,
    center: Tuple[float, float],
    contour: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of the detected cross center.
    
    Args:
        image: Input image
        center: (x, y) center coordinates
        contour: Optional contour to draw
        output_path: Optional path to save visualization
    
    Returns:
        Visualization image
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    cx, cy = center
    int_cx, int_cy = int(round(cx)), int(round(cy))
    
    # Draw contour if provided
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    
    # Draw crosshair at detected center
    cv2.line(vis, (int_cx - 50, int_cy), (int_cx + 50, int_cy), (0, 0, 255), 2)
    cv2.line(vis, (int_cx, int_cy - 50), (int_cx, int_cy + 50), (0, 0, 255), 2)
    cv2.circle(vis, (int_cx, int_cy), 5, (255, 0, 255), -1)
    
    # Add text with coordinates
    text = f"({cx:.3f}, {cy:.3f})"
    cv2.putText(vis, text, (int_cx + 10, int_cy - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Detect cross template center with subpixel accuracy"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--debug", action="store_true", help="Save debug visualization")
    args = parser.parse_args()
    
    # Read image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot read image '{args.image}'")
        return 1
    
    print(f"Image: {args.image}")
    print(f"Dimensions: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Detect cross
    result = detect_cross_center(image, return_contour=True)
    
    if result[0] is None:
        print("Error: No cross template detected in image")
        return 1
    
    center, contour = result
    
    print(f"\n{'='*60}")
    print(f"CROSS CENTER (SUBPIXEL COORDINATES):")
    print(f"  x = {center[0]:.6f}")
    print(f"  y = {center[1]:.6f}")
    print(f"{'='*60}")
    
    if args.debug:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = f"{base_name}_cross_center.png"
        visualize_detection(image, center, contour, output_path)
        print(f"\nVisualization saved: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
