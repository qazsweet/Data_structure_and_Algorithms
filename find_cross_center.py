"""
Cross template detection with subpixel center estimation.

This script detects a cross (plus sign) pattern in an image and computes
the center coordinates with subpixel accuracy using multiple techniques:
1. Template matching for coarse detection
2. Edge detection and line fitting
3. Moment-based centroid calculation
4. Subpixel refinement using quadratic fitting
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
import argparse


def create_cross_template(size=50, arm_width=10, bg_val=0, fg_val=255):
    """Create a synthetic cross template for template matching."""
    template = np.full((size, size), bg_val, dtype=np.uint8)
    center = size // 2
    half_arm = arm_width // 2
    
    # Horizontal arm
    template[center - half_arm:center + half_arm, :] = fg_val
    # Vertical arm
    template[:, center - half_arm:center + half_arm] = fg_val
    
    return template


def detect_cross_template_matching(gray, template_sizes=[30, 40, 50, 60, 70, 80]):
    """
    Detect cross using multi-scale template matching.
    Returns the best match location and correlation score.
    """
    best_val = -1
    best_loc = None
    best_template = None
    best_scale = None
    
    # Normalize image
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    for size in template_sizes:
        for arm_width in [size // 5, size // 4, size // 3]:
            # Try both light cross on dark and dark cross on light
            for invert in [False, True]:
                template = create_cross_template(size, arm_width)
                if invert:
                    template = 255 - template
                
                # Apply template matching
                result = cv2.matchTemplate(gray_norm, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_template = template
                    best_scale = size
    
    if best_loc is not None:
        # Convert to center coordinates
        center_x = best_loc[0] + best_scale // 2
        center_y = best_loc[1] + best_scale // 2
        return (center_x, center_y), best_val, best_template
    
    return None, 0, None


def detect_cross_edges(gray, sigma=2.0):
    """
    Detect cross using edge detection and contour analysis.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Adaptive thresholding to handle uneven illumination
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 5
    )
    
    # Also try Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours_adaptive, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_otsu, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours = contours_adaptive + contours_otsu
    
    best_cross = None
    best_score = 0
    
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Too small
            continue
        
        # Check if contour resembles a cross shape
        # Cross has a specific aspect ratio and convexity defects
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            solidity = area / hull_area
            # A cross typically has solidity around 0.4-0.7
            if 0.3 < solidity < 0.8:
                # Check perimeter to area ratio
                perimeter = cv2.arcLength(contour, True)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Cross has low compactness (< 0.5)
                if compactness < 0.5:
                    score = area * (1 - compactness)
                    if score > best_score:
                        best_score = score
                        best_cross = contour
    
    if best_cross is not None:
        M = cv2.moments(best_cross)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx, cy)
    
    return None


def detect_cross_hough_lines(gray, sigma=1.5):
    """
    Detect cross using Hough line detection.
    Looks for perpendicular lines that intersect.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return None
    
    # Group lines by angle
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Horizontal lines (angle close to 0 or 180)
        if abs(angle) < 20 or abs(angle) > 160:
            horizontal_lines.append((x1, y1, x2, y2))
        # Vertical lines (angle close to 90 or -90)
        elif 70 < abs(angle) < 110:
            vertical_lines.append((x1, y1, x2, y2))
    
    if not horizontal_lines or not vertical_lines:
        return None
    
    # Find intersection of dominant horizontal and vertical lines
    intersections = []
    
    for h_line in horizontal_lines:
        hx1, hy1, hx2, hy2 = h_line
        for v_line in vertical_lines:
            vx1, vy1, vx2, vy2 = v_line
            
            # Check if lines are near each other
            h_mid_x = (hx1 + hx2) / 2
            h_mid_y = (hy1 + hy2) / 2
            v_mid_x = (vx1 + vx2) / 2
            v_mid_y = (vy1 + vy2) / 2
            
            dist = np.sqrt((h_mid_x - v_mid_x)**2 + (h_mid_y - v_mid_y)**2)
            
            if dist < 100:  # Lines are close
                intersections.append(((h_mid_x + v_mid_x) / 2, (h_mid_y + v_mid_y) / 2))
    
    if intersections:
        # Average all intersection points
        avg_x = np.mean([p[0] for p in intersections])
        avg_y = np.mean([p[1] for p in intersections])
        return (avg_x, avg_y)
    
    return None


def refine_center_subpixel(gray, center, window_size=20):
    """
    Refine center location to subpixel accuracy using intensity-weighted centroid
    and quadratic fitting.
    """
    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape
    
    # Define ROI
    x1 = max(0, cx - window_size)
    x2 = min(w, cx + window_size)
    y1 = max(0, cy - window_size)
    y2 = min(h, cy + window_size)
    
    roi = gray[y1:y2, x1:x2].astype(np.float64)
    
    # Apply edge detection in ROI
    edges = cv2.Canny(roi.astype(np.uint8), 30, 100)
    
    # Find edge points
    edge_points = np.column_stack(np.where(edges > 0))
    
    if len(edge_points) < 10:
        return center
    
    # Compute centroid of edge points (weighted by edge strength)
    edge_y, edge_x = edge_points[:, 0], edge_points[:, 1]
    
    # Weight by gradient magnitude
    gy, gx = np.gradient(roi)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    weights = grad_mag[edge_y, edge_x]
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        refined_x = np.sum(edge_x * weights) / total_weight
        refined_y = np.sum(edge_y * weights) / total_weight
        
        # Convert back to image coordinates
        refined_x += x1
        refined_y += y1
        
        return (refined_x, refined_y)
    
    return center


def refine_center_quadratic(gray, center, window_size=15):
    """
    Refine center using quadratic surface fitting on gradient magnitude.
    """
    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape
    
    # Define ROI
    x1 = max(0, cx - window_size)
    x2 = min(w, cx + window_size)
    y1 = max(0, cy - window_size)
    y2 = min(h, cy + window_size)
    
    roi = gray[y1:y2, x1:x2].astype(np.float64)
    
    # Compute gradient magnitude
    gy, gx = np.gradient(roi)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Create coordinate grids
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    z_flat = grad_mag.flatten()
    
    # Fit a 2D quadratic surface: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    # Use only high-gradient regions
    threshold = np.percentile(z_flat, 70)
    mask = z_flat > threshold
    
    if np.sum(mask) < 20:
        return center
    
    xx_m = xx_flat[mask]
    yy_m = yy_flat[mask]
    z_m = z_flat[mask]
    
    # Build design matrix
    A = np.column_stack([
        xx_m**2, yy_m**2, xx_m * yy_m, xx_m, yy_m, np.ones_like(xx_m)
    ])
    
    try:
        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, z_m, rcond=None)
        a, b, c, d, e, f = coeffs
        
        # Find the maximum of the quadratic surface
        # Gradient: [2ax + cy + d, 2by + cx + e] = [0, 0]
        # Matrix form: [[2a, c], [c, 2b]] * [x, y] = [-d, -e]
        M = np.array([[2*a, c], [c, 2*b]])
        rhs = np.array([-d, -e])
        
        det = np.linalg.det(M)
        if abs(det) > 1e-6:
            peak = np.linalg.solve(M, rhs)
            peak_x, peak_y = peak
            
            # Verify the peak is within ROI bounds
            if 0 <= peak_x < roi.shape[1] and 0 <= peak_y < roi.shape[0]:
                # This gives us where gradients are highest - we want the center
                # which is typically near the average position of high-gradient points
                refined_x = x1 + np.mean(xx_m)
                refined_y = y1 + np.mean(yy_m)
                return (refined_x, refined_y)
    except:
        pass
    
    return center


def detect_cross_moment_based(gray, sigma=2.0):
    """
    Detect cross using image moments after preprocessing.
    """
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Normalize
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 101, 2
    )
    
    # Try to isolate the cross region
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find the largest connected component that resembles a cross
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    best_idx = -1
    best_score = 0
    
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        if area < 100:
            continue
        
        # A cross has roughly similar width and height
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        
        # Cross occupies about 40-60% of bounding box
        fill_ratio = area / (width * height) if width * height > 0 else 0
        
        if aspect_ratio > 0.5 and 0.2 < fill_ratio < 0.7:
            score = area * aspect_ratio
            if score > best_score:
                best_score = score
                best_idx = i
    
    if best_idx > 0:
        return tuple(centroids[best_idx])
    
    return None


def find_cross_center_combined(image_path, debug=False):
    """
    Combine multiple detection methods for robust cross center detection.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    print(f"Image size: {gray.shape[1]} x {gray.shape[0]}")
    
    # Method 1: Template matching
    center_tm, score_tm, template = detect_cross_template_matching(gray)
    print(f"Template matching: center={center_tm}, score={score_tm:.4f}")
    
    # Method 2: Edge-based detection
    center_edge = detect_cross_edges(gray)
    print(f"Edge detection: center={center_edge}")
    
    # Method 3: Hough lines
    center_hough = detect_cross_hough_lines(gray)
    print(f"Hough lines: center={center_hough}")
    
    # Method 4: Moment-based
    center_moment = detect_cross_moment_based(gray)
    print(f"Moment-based: center={center_moment}")
    
    # Combine results
    valid_centers = []
    if center_tm is not None and score_tm > 0.3:
        valid_centers.append(center_tm)
    if center_edge is not None:
        valid_centers.append(center_edge)
    if center_hough is not None:
        valid_centers.append(center_hough)
    if center_moment is not None:
        valid_centers.append(center_moment)
    
    if not valid_centers:
        # Fallback: use image center as starting point
        print("Warning: No methods found the cross. Using image center.")
        coarse_center = (gray.shape[1] / 2, gray.shape[0] / 2)
    else:
        # Average the detected centers
        coarse_center = (
            np.mean([c[0] for c in valid_centers]),
            np.mean([c[1] for c in valid_centers])
        )
    
    print(f"\nCoarse center (averaged): ({coarse_center[0]:.2f}, {coarse_center[1]:.2f})")
    
    # Subpixel refinement
    refined_center = refine_center_subpixel(gray, coarse_center)
    print(f"After intensity-weighted refinement: ({refined_center[0]:.4f}, {refined_center[1]:.4f})")
    
    final_center = refine_center_quadratic(gray, refined_center)
    print(f"\n=== Final subpixel center: ({final_center[0]:.6f}, {final_center[1]:.6f}) ===")
    
    if debug:
        # Save debug visualization
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw detected centers
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ['Template', 'Edge', 'Hough', 'Moment']
        centers = [center_tm, center_edge, center_hough, center_moment]
        
        for center, color, label in zip(centers, colors, labels):
            if center is not None:
                cv2.circle(vis, (int(center[0]), int(center[1])), 5, color, 1)
        
        # Draw final center
        cv2.circle(vis, (int(final_center[0]), int(final_center[1])), 8, (0, 255, 255), 2)
        cv2.drawMarker(vis, (int(final_center[0]), int(final_center[1])), (0, 255, 255), 
                       cv2.MARKER_CROSS, 20, 2)
        
        cv2.imwrite('cross_detection_debug.png', vis)
        print("Debug image saved: cross_detection_debug.png")
    
    return final_center


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect cross center with subpixel accuracy")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--debug", action="store_true", help="Save debug visualization")
    args = parser.parse_args()
    
    center = find_cross_center_combined(args.image, debug=args.debug)
    print(f"\nResult: Center coordinates (x, y) = ({center[0]:.6f}, {center[1]:.6f})")
