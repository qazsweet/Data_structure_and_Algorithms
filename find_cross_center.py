"""
Cross template center detection with subpixel accuracy.

This script finds the center of a cross/plus shaped fiducial marker
in an image using multiple methods for robust subpixel localization.

Usage:
    python3 find_cross_center.py image.png
    python3 find_cross_center.py --test  # Run with synthetic test image
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple, List
import numpy as np
import cv2
from scipy import ndimage


def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and return BGR and grayscale versions."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def preprocess_image(gray: np.ndarray, blur_sigma: float = 1.5) -> np.ndarray:
    """Preprocess image with Gaussian blur."""
    if blur_sigma > 0:
        k = int(round(blur_sigma * 6 + 1)) | 1
        gray_smooth = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    else:
        gray_smooth = gray.copy()
    return gray_smooth


def create_cross_template(size: int, arm_width_ratio: float = 0.35, 
                          border_width: int = 2, filled: bool = False) -> np.ndarray:
    """Create a synthetic cross/plus template."""
    template = np.zeros((size, size), dtype=np.uint8)
    arm_width = int(size * arm_width_ratio)
    center = size // 2
    half_arm = arm_width // 2
    
    if filled:
        # Filled cross
        template[:, center - half_arm:center + half_arm + 1] = 255
        template[center - half_arm:center + half_arm + 1, :] = 255
    else:
        # Outline cross
        cv2.rectangle(template, 
                     (center - half_arm, 0), 
                     (center + half_arm, size - 1), 
                     255, border_width)
        cv2.rectangle(template, 
                     (0, center - half_arm), 
                     (size - 1, center + half_arm), 
                     255, border_width)
    
    return template


def detect_cross_by_template_matching(gray: np.ndarray,
                                       cross_sizes: List[int] = None,
                                       arm_ratios: List[float] = None) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """
    Detect cross using normalized cross-correlation template matching.
    Returns (x, y, best_score, correlation_map) or None if not found.
    """
    if cross_sizes is None:
        cross_sizes = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140]
    if arm_ratios is None:
        arm_ratios = [0.25, 0.30, 0.35, 0.40, 0.45]
    
    best_result = None
    best_score = -1
    best_corr_map = None
    best_template_size = 0
    
    for size in cross_sizes:
        for arm_ratio in arm_ratios:
            for filled in [True, False]:
                for border_w in [1, 2, 3]:
                    template = create_cross_template(size, arm_ratio, border_width=border_w, filled=filled)
                    
                    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                        continue
                    
                    for templ in [template, 255 - template]:
                        result = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > best_score:
                            best_score = max_val
                            cx = max_loc[0] + template.shape[1] / 2.0
                            cy = max_loc[1] + template.shape[0] / 2.0
                            best_result = (cx, cy, max_val)
                            best_corr_map = result.copy()
                            best_template_size = size
    
    if best_result is None:
        return None
    
    return best_result[0], best_result[1], best_result[2], best_corr_map


def detect_cross_by_contour(gray: np.ndarray, 
                            min_area: int = 200,
                            max_area_ratio: float = 0.3) -> Optional[Tuple[float, float]]:
    """Detect cross using contour analysis."""
    h, w = gray.shape
    max_area = h * w * max_area_ratio
    
    methods = []
    binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 51, 5)
    methods.extend([binary_adapt, 255 - binary_adapt])
    
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.extend([binary_otsu, 255 - binary_otsu])
    
    for thresh_val in [80, 100, 120, 140, 160, 180]:
        _, binary_fixed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        methods.extend([binary_fixed, 255 - binary_fixed])
    
    best_cross = None
    best_score = 0
    
    for bin_img in methods:
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is None or len(hull) < 4 or len(contour) < 5:
                continue
            
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is None:
                    continue
                
                significant_defects = sum(1 for i in range(defects.shape[0]) if defects[i, 0, 3] > 1000)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = area / hull_area if hull_area > 0 else 0
                cross_score = significant_defects * (1.0 - abs(solidity - 0.6))
                
                if cross_score > best_score and significant_defects >= 4:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        best_cross = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                        best_score = cross_score
            except cv2.error:
                continue
    
    return best_cross


def detect_cross_by_edge_intersection(gray: np.ndarray) -> Optional[Tuple[float, float]]:
    """Detect cross center by finding edge intersections using Hough lines."""
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(edges_dilated, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    
    if lines is None or len(lines) < 2:
        return None
    
    h_lines, v_lines = [], []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if abs(angle) < 20 or abs(angle) > 160:
            h_lines.append((x1, y1, x2, y2, length))
        elif 70 < abs(angle) < 110:
            v_lines.append((x1, y1, x2, y2, length))
    
    if not h_lines or not v_lines:
        return None
    
    h_weights = np.array([l[4] for l in h_lines])
    v_weights = np.array([l[4] for l in v_lines])
    h_weights = h_weights / h_weights.sum()
    v_weights = v_weights / v_weights.sum()
    
    h_y = sum(((y1 + y2) / 2) * w for (x1, y1, x2, y2, _), w in zip(h_lines, h_weights))
    v_x = sum(((x1 + x2) / 2) * w for (x1, y1, x2, y2, _), w in zip(v_lines, v_weights))
    
    return (v_x, h_y)


def refine_by_symmetry(gray: np.ndarray, cx: float, cy: float,
                       window_size: int = 51, search_range: float = 5.0,
                       step: float = 0.1) -> Tuple[float, float]:
    """Refine center by finding point of maximum rotational symmetry."""
    h, w = gray.shape
    half_win = window_size // 2
    gray_f = gray.astype(np.float64)
    
    best_x, best_y = cx, cy
    best_symmetry = -float('inf')
    
    for dx in np.arange(-search_range, search_range + step, step):
        for dy in np.arange(-search_range, search_range + step, step):
            test_x, test_y = cx + dx, cy + dy
            x_start = int(test_x - half_win)
            y_start = int(test_y - half_win)
            x_end = x_start + 2 * half_win + 1
            y_end = y_start + 2 * half_win + 1
            
            if x_start < 0 or y_start < 0 or x_end > w or y_end > h:
                continue
            
            roi = gray_f[y_start:y_end, x_start:x_end]
            roi_rot = np.rot90(roi, 2)
            symmetry = -np.sum((roi - roi_rot) ** 2)
            
            if symmetry > best_symmetry:
                best_symmetry = symmetry
                best_x, best_y = test_x, test_y
    
    return best_x, best_y


def refine_subpixel_quadratic(corr_map: np.ndarray, peak_x: int, peak_y: int) -> Tuple[float, float]:
    """Refine peak location using 2D quadratic (parabolic) fitting."""
    h, w = corr_map.shape
    
    if not (1 <= peak_x < w - 1 and 1 <= peak_y < h - 1):
        return float(peak_x), float(peak_y)
    
    patch = corr_map[peak_y - 1:peak_y + 2, peak_x - 1:peak_x + 2].astype(np.float64)
    
    denom_x = 2 * patch[1, 1] - patch[1, 0] - patch[1, 2]
    dx = 0.5 * (patch[1, 0] - patch[1, 2]) / denom_x if abs(denom_x) > 1e-10 else 0.0
    
    denom_y = 2 * patch[1, 1] - patch[0, 1] - patch[2, 1]
    dy = 0.5 * (patch[0, 1] - patch[2, 1]) / denom_y if abs(denom_y) > 1e-10 else 0.0
    
    return peak_x + np.clip(dx, -0.5, 0.5), peak_y + np.clip(dy, -0.5, 0.5)


def refine_center_by_gradient(gray: np.ndarray, cx: float, cy: float,
                               window_size: int = 31) -> Tuple[float, float]:
    """Refine center using gradient-based centroid method."""
    h, w = gray.shape
    half_win = window_size // 2
    
    x_start = max(0, int(cx) - half_win)
    x_end = min(w, int(cx) + half_win + 1)
    y_start = max(0, int(cy) - half_win)
    y_end = min(h, int(cy) + half_win + 1)
    
    roi = gray[y_start:y_end, x_start:x_end].astype(np.float64)
    gy, gx = np.gradient(roi)
    mag = np.sqrt(gx**2 + gy**2)
    
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    total_weight = np.sum(mag) + 1e-10
    
    return x_start + np.sum(xx * mag) / total_weight, y_start + np.sum(yy * mag) / total_weight


def detect_cross_center(image_path: str, visualize: bool = True,
                        output_path: Optional[str] = None) -> Tuple[float, float]:
    """
    Main function to detect cross center with subpixel accuracy.
    
    Uses multiple detection and refinement methods:
    1. Template matching with synthetic crosses
    2. Contour analysis  
    3. Edge-based line detection
    4. Subpixel refinement using symmetry and quadratic fitting
    """
    bgr, gray = load_image(image_path)
    gray_smooth = preprocess_image(gray, blur_sigma=1.0)
    
    print(f"Image size: {gray.shape[1]} x {gray.shape[0]} pixels")
    print("=" * 60)
    
    results = []
    corr_map = None
    
    # Method 1: Template matching
    print("\n[Method 1] Template matching...")
    template_result = detect_cross_by_template_matching(gray_smooth)
    if template_result is not None:
        cx, cy, score, corr_map = template_result
        print(f"  Detection: ({cx:.2f}, {cy:.2f}), correlation: {score:.4f}")
        results.append((cx, cy, score, "template"))
    else:
        print("  No cross found")
    
    # Method 2: Contour-based
    print("\n[Method 2] Contour analysis...")
    contour_result = detect_cross_by_contour(gray_smooth)
    if contour_result is not None:
        cx, cy = contour_result
        print(f"  Detection: ({cx:.2f}, {cy:.2f})")
        results.append((cx, cy, 0.5, "contour"))
    else:
        print("  No cross found")
    
    # Method 3: Edge intersection
    print("\n[Method 3] Edge intersection...")
    edge_result = detect_cross_by_edge_intersection(gray_smooth)
    if edge_result is not None:
        cx, cy = edge_result
        print(f"  Detection: ({cx:.2f}, {cy:.2f})")
        results.append((cx, cy, 0.3, "edge"))
    else:
        print("  No cross found")
    
    if not results:
        raise ValueError("Could not detect cross in the image")
    
    results.sort(key=lambda x: x[2], reverse=True)
    best_cx, best_cy, best_score, best_method = results[0]
    print(f"\nBest initial: ({best_cx:.2f}, {best_cy:.2f}) via {best_method}")
    
    # Subpixel refinement
    print("\n" + "=" * 60)
    print("SUBPIXEL REFINEMENT")
    print("=" * 60)
    
    refinements = []
    
    if corr_map is not None:
        _, _, _, max_loc = cv2.minMaxLoc(corr_map)
        template_size = gray.shape[0] - corr_map.shape[0] + 1
        ref_x, ref_y = refine_subpixel_quadratic(corr_map, max_loc[0], max_loc[1])
        ref_x += template_size / 2.0
        ref_y += template_size / 2.0
        print(f"\n[A] Quadratic fit: ({ref_x:.6f}, {ref_y:.6f})")
        refinements.append((ref_x, ref_y, "quadratic"))
    
    ref_x_sym, ref_y_sym = refine_by_symmetry(gray_smooth, best_cx, best_cy,
                                               window_size=61, search_range=3.0, step=0.05)
    print(f"[B] Symmetry: ({ref_x_sym:.6f}, {ref_y_sym:.6f})")
    refinements.append((ref_x_sym, ref_y_sym, "symmetry"))
    
    ref_x_grad, ref_y_grad = refine_center_by_gradient(gray_smooth, best_cx, best_cy, window_size=41)
    print(f"[C] Gradient: ({ref_x_grad:.6f}, {ref_y_grad:.6f})")
    refinements.append((ref_x_grad, ref_y_grad, "gradient"))
    
    # Weighted combination
    weights = {"quadratic": 2.0, "symmetry": 1.5, "gradient": 1.0}
    total_weight = sum(weights[m] for _, _, m in refinements)
    final_x = sum(x * weights[m] for x, _, m in refinements) / total_weight
    final_y = sum(y * weights[m] for _, y, m in refinements) / total_weight
    
    print("\n" + "=" * 60)
    print(f"FINAL CENTER (SUBPIXEL): ({final_x:.6f}, {final_y:.6f})")
    print("=" * 60)
    
    if visualize:
        vis = bgr.copy()
        colors = {"template": (0, 255, 0), "contour": (255, 0, 0), "edge": (0, 255, 255)}
        
        for cx, cy, _, method in results:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 8, colors.get(method, (128, 128, 128)), 1)
        
        for rx, ry, _ in refinements:
            cv2.circle(vis, (int(round(rx)), int(round(ry))), 5, (255, 0, 255), 1)
        
        final_ix, final_iy = int(round(final_x)), int(round(final_y))
        cv2.drawMarker(vis, (final_ix, final_iy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(vis, (final_ix, final_iy), 15, (0, 0, 255), 2)
        
        cv2.putText(vis, f"Center: ({final_x:.4f}, {final_y:.4f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if output_path is None:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"cross_center_{stem}.png"
        
        cv2.imwrite(output_path, vis)
        print(f"\nVisualization saved to: {output_path}")
    
    return final_x, final_y


def create_test_image(output_path: str = "test_cross.png",
                      size: Tuple[int, int] = (1232, 959),
                      cross_center: Tuple[float, float] = (616.37, 478.82),
                      cross_size: int = 80) -> Tuple[str, float, float]:
    """Create a synthetic test image mimicking the user's attached image."""
    w, h = size
    cx, cy = cross_center
    
    np.random.seed(42)
    img = np.random.randint(110, 140, (h, w), dtype=np.uint8)
    
    # Add diagonal stripes
    for i in range(0, w + h, 50):
        cv2.line(img, (i, 0), (i - h, h), 120, 25)
    
    # Dark top region
    img[0:75, :] = np.random.randint(15, 35, (75, w), dtype=np.uint8)
    
    img = cv2.GaussianBlur(img, (15, 15), 3)
    
    # Draw cross
    arm_half_width = cross_size // 6
    arm_length = cross_size // 2
    
    cv2.rectangle(img, (int(cx - arm_half_width), int(cy - arm_length)),
                  (int(cx + arm_half_width), int(cy + arm_length)), 175, -1)
    cv2.rectangle(img, (int(cx - arm_length), int(cy - arm_half_width)),
                  (int(cx + arm_length), int(cy + arm_half_width)), 175, -1)
    cv2.rectangle(img, (int(cx - arm_half_width), int(cy - arm_length)),
                  (int(cx + arm_half_width), int(cy + arm_length)), 85, 2)
    cv2.rectangle(img, (int(cx - arm_length), int(cy - arm_half_width)),
                  (int(cx + arm_length), int(cy + arm_half_width)), 85, 2)
    
    noise = np.random.normal(0, 2, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    cv2.imwrite(output_path, img)
    print(f"Test image created: {output_path}")
    print(f"True cross center: ({cx}, {cy})")
    
    return output_path, cx, cy


def main():
    parser = argparse.ArgumentParser(description="Detect cross template center with subpixel accuracy")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image")
    parser.add_argument("--output", "-o", default=None, help="Output visualization path")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--test", action="store_true", help="Run test with synthetic image")
    
    args = parser.parse_args()
    
    try:
        if args.test or args.image is None:
            print("Running test with synthetic cross image...\n")
            test_path, true_cx, true_cy = create_test_image()
            
            print()
            detected_cx, detected_cy = detect_cross_center(
                test_path, visualize=not args.no_visualize, output_path=args.output)
            
            error_x = detected_cx - true_cx
            error_y = detected_cy - true_cy
            error_dist = np.sqrt(error_x**2 + error_y**2)
            
            print(f"\n{'='*60}")
            print("TEST RESULTS")
            print(f"{'='*60}")
            print(f"True center:     ({true_cx:.4f}, {true_cy:.4f})")
            print(f"Detected:        ({detected_cx:.4f}, {detected_cy:.4f})")
            print(f"Error:           ({error_x:.4f}, {error_y:.4f})")
            print(f"Distance error:  {error_dist:.4f} pixels")
            print(f"{'='*60}")
        else:
            detect_cross_center(args.image, visualize=not args.no_visualize, output_path=args.output)
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
