"""
Cross template detection with subpixel accuracy.

Specifically designed to detect cross fiducial markers in microscopy/industrial images.
Uses template correlation and subpixel refinement.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize, least_squares
import sys
import os


def create_cross_templates(sizes=[60, 80, 100, 120, 150], arm_widths_ratio=[0.15, 0.2, 0.25]):
    """
    Create a set of cross templates at various sizes and arm widths.
    Templates are created as both filled and outline versions.
    """
    templates = []
    
    for size in sizes:
        for ratio in arm_widths_ratio:
            arm_width = max(4, int(size * ratio))
            
            # Filled cross template
            filled = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            hw = arm_width // 2
            
            # Horizontal arm
            filled[center - hw:center + hw + 1, :] = 1.0
            # Vertical arm
            filled[:, center - hw:center + hw + 1] = 1.0
            
            # Apply slight blur for better matching
            filled = cv2.GaussianBlur(filled, (3, 3), 0.5)
            templates.append(('filled', size, arm_width, filled))
            
            # Outline cross template (edges only)
            outline = cv2.Canny((filled * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
            outline = cv2.GaussianBlur(outline, (3, 3), 0.5)
            templates.append(('outline', size, arm_width, outline))
    
    return templates


def match_cross_template(gray, template, method=cv2.TM_CCOEFF_NORMED):
    """
    Perform template matching and return the best match location and score.
    """
    # Normalize image
    img = cv2.normalize(gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    
    # Template matching
    result = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return max_loc, max_val, result


def find_cross_by_template_matching(gray, debug_path=None):
    """
    Find cross using multi-scale template matching.
    """
    # Normalize and enhance
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Try both original and inverted (cross might be lighter or darker than background)
    images_to_try = [
        ('normal', gray_norm),
        ('inverted', 255 - gray_norm),
    ]
    
    # Also try edge-enhanced version
    edges = cv2.Canny(gray_norm, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    images_to_try.append(('edges', edges_dilated))
    
    templates = create_cross_templates()
    
    best_match = {
        'score': -1,
        'center': None,
        'template_info': None,
        'image_type': None
    }
    
    for img_type, img in images_to_try:
        for template_type, size, arm_width, template in templates:
            loc, score, result_map = match_cross_template(img, template)
            
            if score > best_match['score']:
                center_x = loc[0] + size // 2
                center_y = loc[1] + size // 2
                best_match = {
                    'score': score,
                    'center': (center_x, center_y),
                    'template_info': (template_type, size, arm_width),
                    'image_type': img_type,
                    'result_map': result_map,
                    'top_left': loc
                }
    
    return best_match


def detect_cross_by_edge_geometry(gray, expected_region=None):
    """
    Detect cross by analyzing edge geometry.
    Looks for perpendicular line segments forming a cross pattern.
    """
    # Preprocessing
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Edge detection
    edges = cv2.Canny(blur, 30, 100)
    
    # Hough line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return None
    
    # Classify lines
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < 20:
            continue
            
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        if angle < 20:  # Horizontal
            horizontal.append((mid_x, mid_y, x1, y1, x2, y2, length))
        elif angle > 70:  # Vertical
            vertical.append((mid_x, mid_y, x1, y1, x2, y2, length))
    
    # Find potential cross centers (intersections of H and V lines)
    cross_candidates = []
    
    for h in horizontal:
        hx, hy = h[0], h[1]
        for v in vertical:
            vx, vy = v[0], v[1]
            
            # Check if lines are close to each other
            dist = np.sqrt((hx - vx)**2 + (hy - vy)**2)
            
            if dist < 50:  # Lines are close
                # Compute intersection point more precisely
                h_x1, h_y1, h_x2, h_y2 = h[2:6]
                v_x1, v_y1, v_x2, v_y2 = v[2:6]
                
                # Line equations
                # Horizontal: y = h_y1 (approximately)
                # Vertical: x = v_x1 (approximately)
                
                avg_h_y = (h_y1 + h_y2) / 2
                avg_v_x = (v_x1 + v_x2) / 2
                
                cross_candidates.append({
                    'center': (avg_v_x, avg_h_y),
                    'h_length': h[6],
                    'v_length': v[6],
                    'score': h[6] * v[6]  # Score by combined line lengths
                })
    
    if not cross_candidates:
        return None
    
    # Return the best candidate
    best = max(cross_candidates, key=lambda x: x['score'])
    return best['center']


def refine_center_subpixel_correlation(gray, coarse_center, template_size=80, search_window=5):
    """
    Refine center using subpixel template correlation.
    Uses parabolic fitting on correlation surface.
    """
    cx, cy = int(round(coarse_center[0])), int(round(coarse_center[1]))
    h, w = gray.shape
    
    # Create optimal cross template based on local analysis
    half_size = template_size // 2
    
    # Extract local region
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)
    
    local = gray[y1:y2, x1:x2].astype(np.float32)
    
    # Create template from local region (assuming cross is roughly centered)
    template_half = min(template_size // 2, min(local.shape) // 2 - 2)
    
    if template_half < 10:
        return coarse_center
    
    local_center = (local.shape[1] // 2, local.shape[0] // 2)
    template = local[
        local_center[1] - template_half:local_center[1] + template_half,
        local_center[0] - template_half:local_center[0] + template_half
    ]
    
    # Compute correlation in search window
    search_half = search_window
    corr_values = np.zeros((2 * search_half + 1, 2 * search_half + 1))
    
    for dy in range(-search_half, search_half + 1):
        for dx in range(-search_half, search_half + 1):
            shifted_cy = local_center[1] + dy
            shifted_cx = local_center[0] + dx
            
            shifted_region = local[
                shifted_cy - template_half:shifted_cy + template_half,
                shifted_cx - template_half:shifted_cx + template_half
            ]
            
            if shifted_region.shape == template.shape:
                corr = np.corrcoef(template.flatten(), shifted_region.flatten())[0, 1]
                corr_values[dy + search_half, dx + search_half] = corr
    
    # Find peak in correlation
    peak_y, peak_x = np.unravel_index(np.argmax(corr_values), corr_values.shape)
    
    # Subpixel refinement using parabolic fit
    if 1 <= peak_x < corr_values.shape[1] - 1 and 1 <= peak_y < corr_values.shape[0] - 1:
        # Fit parabola in x direction
        v_left = corr_values[peak_y, peak_x - 1]
        v_center = corr_values[peak_y, peak_x]
        v_right = corr_values[peak_y, peak_x + 1]
        
        denom_x = v_left - 2 * v_center + v_right
        if abs(denom_x) > 1e-6:
            subpix_dx = 0.5 * (v_left - v_right) / denom_x
        else:
            subpix_dx = 0
        
        # Fit parabola in y direction
        v_top = corr_values[peak_y - 1, peak_x]
        v_bottom = corr_values[peak_y + 1, peak_x]
        
        denom_y = v_top - 2 * v_center + v_bottom
        if abs(denom_y) > 1e-6:
            subpix_dy = 0.5 * (v_top - v_bottom) / denom_y
        else:
            subpix_dy = 0
        
        # Clamp to reasonable range
        subpix_dx = np.clip(subpix_dx, -0.5, 0.5)
        subpix_dy = np.clip(subpix_dy, -0.5, 0.5)
        
        # Compute final refined center
        refined_x = coarse_center[0] + (peak_x - search_half + subpix_dx)
        refined_y = coarse_center[1] + (peak_y - search_half + subpix_dy)
        
        return (refined_x, refined_y)
    
    return coarse_center


def refine_center_gradient_descent(gray, coarse_center, window_size=40):
    """
    Refine center using gradient-based optimization.
    Minimizes the asymmetry of gradients around the center.
    """
    cx, cy = coarse_center
    h, w = gray.shape
    
    # Compute gradients
    gy, gx = np.gradient(gray.astype(np.float64))
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    def asymmetry_cost(center):
        """
        Cost function: measures asymmetry around the given center.
        For a perfect cross, the gradient pattern should be symmetric.
        """
        x, y = center
        
        # Define region
        x1 = max(0, int(x - window_size))
        x2 = min(w, int(x + window_size))
        y1 = max(0, int(y - window_size))
        y2 = min(h, int(y + window_size))
        
        local_grad = grad_mag[y1:y2, x1:x2]
        
        if local_grad.size == 0:
            return float('inf')
        
        # Local center in ROI coordinates
        lx = x - x1
        ly = y - y1
        
        # Create coordinate grids
        yy, xx = np.mgrid[0:local_grad.shape[0], 0:local_grad.shape[1]]
        
        # Compute weighted centroid of gradient magnitude
        weights = local_grad ** 2
        total_weight = np.sum(weights)
        
        if total_weight < 1e-6:
            return float('inf')
        
        centroid_x = np.sum(xx * weights) / total_weight
        centroid_y = np.sum(yy * weights) / total_weight
        
        # Cost is distance from current center to weighted centroid
        cost = (centroid_x - lx)**2 + (centroid_y - ly)**2
        
        return cost
    
    # Optimize
    result = minimize(
        asymmetry_cost,
        [cx, cy],
        method='Nelder-Mead',
        options={'xatol': 0.001, 'fatol': 0.001, 'maxiter': 100}
    )
    
    if result.success:
        return (result.x[0], result.x[1])
    
    return coarse_center


def refine_center_edge_intersection(gray, coarse_center, window_size=60):
    """
    Refine center by finding intersection of cross arms using edge fitting.
    """
    cx, cy = int(round(coarse_center[0])), int(round(coarse_center[1]))
    h, w = gray.shape
    
    # Extract ROI
    x1 = max(0, cx - window_size)
    x2 = min(w, cx + window_size)
    y1 = max(0, cy - window_size)
    y2 = min(h, cy + window_size)
    
    roi = gray[y1:y2, x1:x2].astype(np.uint8)
    
    # Edge detection
    edges = cv2.Canny(roi, 30, 100)
    
    # Find edge points
    edge_points = np.column_stack(np.where(edges > 0))  # (y, x) format
    
    if len(edge_points) < 20:
        return coarse_center
    
    # Local center
    lcx = cx - x1
    lcy = cy - y1
    
    # Separate edge points into 4 quadrants relative to center
    # Then fit lines to horizontal and vertical arms
    
    # Points above center (horizontal arm, upper edge)
    upper = edge_points[edge_points[:, 0] < lcy - 5]
    # Points below center (horizontal arm, lower edge)
    lower = edge_points[edge_points[:, 0] > lcy + 5]
    # Points left of center (vertical arm, left edge)
    left = edge_points[edge_points[:, 1] < lcx - 5]
    # Points right of center (vertical arm, right edge)
    right = edge_points[edge_points[:, 1] > lcx + 5]
    
    h_lines = []  # y values of horizontal lines
    v_lines = []  # x values of vertical lines
    
    # Fit horizontal lines (y = const) for upper and lower edges
    for pts in [upper, lower]:
        if len(pts) > 5:
            # Filter points that are part of horizontal arm
            horiz_pts = pts[np.abs(pts[:, 1] - lcx) < window_size * 0.8]
            if len(horiz_pts) > 5:
                h_lines.append(np.mean(horiz_pts[:, 0]))
    
    # Fit vertical lines (x = const) for left and right edges
    for pts in [left, right]:
        if len(pts) > 5:
            # Filter points that are part of vertical arm
            vert_pts = pts[np.abs(pts[:, 0] - lcy) < window_size * 0.8]
            if len(vert_pts) > 5:
                v_lines.append(np.mean(vert_pts[:, 1]))
    
    # Compute center from fitted lines
    refined_y = np.mean(h_lines) if h_lines else lcy
    refined_x = np.mean(v_lines) if v_lines else lcx
    
    # Convert back to image coordinates
    return (refined_x + x1, refined_y + y1)


def find_cross_center_subpixel(image_path, debug=False):
    """
    Main function to find cross center with subpixel accuracy.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    print(f"Image: {w} x {h} pixels")
    
    # Step 1: Template matching for coarse detection
    print("\n--- Stage 1: Template Matching ---")
    match_result = find_cross_by_template_matching(gray)
    
    if match_result['score'] > 0.3:
        print(f"Template match: score={match_result['score']:.4f}, center={match_result['center']}")
        print(f"  Template: {match_result['template_info']}, Image type: {match_result['image_type']}")
        coarse_center = match_result['center']
    else:
        print("Template matching failed, trying edge geometry...")
        coarse_center = detect_cross_by_edge_geometry(gray)
        if coarse_center:
            print(f"Edge geometry: center={coarse_center}")
        else:
            print("Warning: Using image center as fallback")
            coarse_center = (w / 2, h / 2)
    
    # Step 2: Subpixel refinement using multiple methods
    print("\n--- Stage 2: Subpixel Refinement ---")
    
    # Method 1: Correlation-based refinement
    refined1 = refine_center_subpixel_correlation(gray, coarse_center)
    print(f"Correlation refinement: ({refined1[0]:.6f}, {refined1[1]:.6f})")
    
    # Method 2: Gradient-based refinement
    refined2 = refine_center_gradient_descent(gray, refined1)
    print(f"Gradient refinement: ({refined2[0]:.6f}, {refined2[1]:.6f})")
    
    # Method 3: Edge intersection refinement
    refined3 = refine_center_edge_intersection(gray, refined2)
    print(f"Edge intersection: ({refined3[0]:.6f}, {refined3[1]:.6f})")
    
    # Final center: weighted average of refinement methods
    # (give more weight to more reliable methods)
    final_x = (refined1[0] + refined2[0] + 2 * refined3[0]) / 4
    final_y = (refined1[1] + refined2[1] + 2 * refined3[1]) / 4
    
    final_center = (final_x, final_y)
    
    print(f"\n{'='*60}")
    print(f"FINAL SUBPIXEL CENTER: x = {final_center[0]:.6f}, y = {final_center[1]:.6f}")
    print(f"{'='*60}")
    
    if debug:
        # Save visualization
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw coarse center (blue)
        cv2.circle(vis, (int(coarse_center[0]), int(coarse_center[1])), 10, (255, 0, 0), 2)
        
        # Draw final center (green crosshair)
        fcx, fcy = int(round(final_center[0])), int(round(final_center[1]))
        cv2.line(vis, (fcx - 30, fcy), (fcx + 30, fcy), (0, 255, 0), 2)
        cv2.line(vis, (fcx, fcy - 30), (fcx, fcy + 30), (0, 255, 0), 2)
        cv2.circle(vis, (fcx, fcy), 5, (0, 0, 255), -1)
        
        # Add text
        text = f"Center: ({final_center[0]:.3f}, {final_center[1]:.3f})"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_cross_center.png"
        cv2.imwrite(output_path, vis)
        print(f"Debug image: {output_path}")
    
    return final_center


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_cross_template_subpixel.py <image_path> [--debug]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    try:
        center = find_cross_center_subpixel(image_path, debug=debug)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
