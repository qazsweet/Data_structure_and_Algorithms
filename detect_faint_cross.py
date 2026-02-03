"""
Detect faint cross pattern in microscopy/industrial images with subpixel accuracy.

This script is optimized for detecting cross patterns that appear as faint
outline shapes (like fiducial markers) in grayscale images.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
import sys


def enhance_cross_features(gray, sigma_blur=1.0):
    """
    Enhance cross features in the image using various filters.
    """
    # Normalize to float
    img = gray.astype(np.float64)
    
    # Apply mild blur to reduce noise
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_blur)
    
    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Compute gradient magnitude for edge detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    return enhanced, grad_mag, sobelx, sobely


def detect_cross_lines(gray, edge_thresh_low=20, edge_thresh_high=80):
    """
    Detect the cross by finding its constituent lines.
    """
    # Enhance contrast
    enhanced, grad_mag, _, _ = enhance_cross_features(gray)
    
    # Edge detection with various thresholds
    edges = cv2.Canny(enhanced, edge_thresh_low, edge_thresh_high)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=15)
    
    if lines is None:
        return None, None
    
    # Classify lines as horizontal or vertical
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 10:
            continue
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if angle < 30:  # Horizontal
            horizontal.append(line[0])
        elif angle > 60:  # Vertical
            vertical.append(line[0])
    
    return horizontal, vertical


def find_cross_region(gray, min_area=500, max_area=50000):
    """
    Find the region containing the cross using contour analysis.
    """
    # Multiple preprocessing approaches
    results = []
    
    # Approach 1: Adaptive thresholding
    for block_size in [51, 101, 151]:
        for c in [2, 5, 10]:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c
            )
            results.append(binary)
            results.append(255 - binary)  # Inverted
    
    # Approach 2: Otsu thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(otsu)
    results.append(255 - otsu)
    
    best_cross = None
    best_score = 0
    best_center = None
    
    for binary in results:
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Cross should have similar width and height
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if aspect < 0.5:
                continue
            
            # Check hull vs contour area (solidity)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            
            solidity = area / hull_area
            
            # A cross typically has solidity around 0.3-0.7
            if 0.25 < solidity < 0.75:
                # Check number of convexity defects
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) >= 3:
                    try:
                        defects = cv2.convexityDefects(contour, hull_indices)
                        if defects is not None:
                            # Cross has 4 prominent defects (indentations)
                            large_defects = sum(1 for d in defects if d[0, 3] > 500)
                            
                            score = area * (1 - abs(solidity - 0.5)) * min(large_defects + 1, 5)
                            
                            if score > best_score:
                                best_score = score
                                best_cross = contour
                                # Compute centroid
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = M["m10"] / M["m00"]
                                    cy = M["m01"] / M["m00"]
                                    best_center = (cx, cy)
                    except:
                        pass
    
    return best_cross, best_center


def refine_cross_center_subpixel(gray, coarse_center, window_size=30):
    """
    Refine the cross center to subpixel accuracy.
    Uses edge-weighted centroid and iterative refinement.
    """
    if coarse_center is None:
        return None
    
    cx, cy = coarse_center
    h, w = gray.shape
    
    # Extract ROI
    x1 = max(0, int(cx - window_size))
    x2 = min(w, int(cx + window_size))
    y1 = max(0, int(cy - window_size))
    y2 = min(h, int(cy + window_size))
    
    roi = gray[y1:y2, x1:x2].astype(np.float64)
    
    if roi.size == 0:
        return coarse_center
    
    # Compute gradients
    gy, gx = np.gradient(roi)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Enhance edges
    edges = cv2.Canny(roi.astype(np.uint8), 20, 60)
    
    # Create coordinate grids
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    
    # Method 1: Edge-weighted centroid
    edge_mask = edges > 0
    if np.any(edge_mask):
        weights = grad_mag * edge_mask
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            refined_x = np.sum(xx * weights) / total_weight
            refined_y = np.sum(yy * weights) / total_weight
            
            # Convert back to image coordinates
            refined_x += x1
            refined_y += y1
            
            # Method 2: Iterative refinement using local symmetry
            # For a cross, the center should have similar gradient patterns
            # in opposite directions
            
            for _ in range(3):  # Iterative refinement
                local_x = refined_x - x1
                local_y = refined_y - y1
                
                # Small window around current estimate
                ws = 10
                lx1 = max(0, int(local_x - ws))
                lx2 = min(roi.shape[1], int(local_x + ws))
                ly1 = max(0, int(local_y - ws))
                ly2 = min(roi.shape[0], int(local_y + ws))
                
                local_roi = roi[ly1:ly2, lx1:lx2]
                local_grad = grad_mag[ly1:ly2, lx1:lx2]
                local_yy, local_xx = np.mgrid[0:local_roi.shape[0], 0:local_roi.shape[1]]
                
                # Weighted centroid in local window
                weights = local_grad ** 2  # Square for stronger weighting
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    new_local_x = np.sum(local_xx * weights) / total_weight
                    new_local_y = np.sum(local_yy * weights) / total_weight
                    
                    refined_x = x1 + lx1 + new_local_x
                    refined_y = y1 + ly1 + new_local_y
            
            return (refined_x, refined_y)
    
    return coarse_center


def fit_cross_model(gray, coarse_center, search_radius=20):
    """
    Fit a cross model to the image for precise center localization.
    Uses optimization to find the center that best aligns with cross edges.
    """
    if coarse_center is None:
        return None
    
    cx, cy = coarse_center
    h, w = gray.shape
    
    # Compute edge strength
    edges = cv2.Canny(gray, 20, 60)
    edge_points = np.column_stack(np.where(edges > 0))  # (y, x) format
    
    if len(edge_points) < 10:
        return coarse_center
    
    # Define region of interest
    roi_mask = (
        (edge_points[:, 1] > cx - search_radius) & 
        (edge_points[:, 1] < cx + search_radius) &
        (edge_points[:, 0] > cy - search_radius) & 
        (edge_points[:, 0] < cy + search_radius)
    )
    local_edges = edge_points[roi_mask]
    
    if len(local_edges) < 10:
        return coarse_center
    
    def cross_distance(center):
        """
        Compute how well edge points align with cross arms from a given center.
        For a cross, edge points should align with horizontal or vertical lines.
        """
        cx, cy = center
        points = local_edges
        
        # Distance to horizontal arm (y = cy)
        dy = np.abs(points[:, 0] - cy)
        # Distance to vertical arm (x = cx)
        dx = np.abs(points[:, 1] - cx)
        
        # For each point, take minimum distance to either arm
        min_dist = np.minimum(dx, dy)
        
        # Weight by distance from center (closer points matter more)
        dist_from_center = np.sqrt((points[:, 0] - cy)**2 + (points[:, 1] - cx)**2)
        weights = np.exp(-dist_from_center / search_radius)
        
        return np.sum(min_dist * weights)
    
    # Optimize
    result = minimize(
        cross_distance, 
        [cx, cy], 
        method='Nelder-Mead',
        options={'xatol': 0.01, 'fatol': 0.01}
    )
    
    if result.success:
        return (result.x[0], result.x[1])
    
    return coarse_center


def detect_cross_center(image_path, debug=False):
    """
    Main function to detect cross center with subpixel accuracy.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    print(f"Image dimensions: {gray.shape[1]} x {gray.shape[0]} pixels")
    
    # Step 1: Find cross region
    cross_contour, coarse_center = find_cross_region(gray)
    
    if coarse_center is not None:
        print(f"Coarse detection (contour): ({coarse_center[0]:.2f}, {coarse_center[1]:.2f})")
    else:
        # Fallback: try gradient-based detection
        enhanced, grad_mag, _, _ = enhance_cross_features(gray)
        
        # Find region with high edge density that might be the cross
        # Apply threshold to gradient magnitude
        grad_thresh = np.percentile(grad_mag, 90)
        high_grad = grad_mag > grad_thresh
        
        # Find connected components
        labeled, num_features = ndimage.label(high_grad)
        if num_features > 0:
            # Find the component with characteristics of a cross
            for i in range(1, num_features + 1):
                component = labeled == i
                ys, xs = np.where(component)
                if len(xs) > 50:  # Minimum points
                    cx = np.mean(xs)
                    cy = np.mean(ys)
                    
                    # Check if this region spans both directions (cross-like)
                    x_spread = np.std(xs)
                    y_spread = np.std(ys)
                    
                    if x_spread > 5 and y_spread > 5:  # Has extent in both directions
                        coarse_center = (cx, cy)
                        print(f"Coarse detection (gradient): ({cx:.2f}, {cy:.2f})")
                        break
    
    if coarse_center is None:
        print("Warning: Could not detect cross. Using image center.")
        coarse_center = (gray.shape[1] / 2, gray.shape[0] / 2)
    
    # Step 2: Subpixel refinement using edge-weighted centroid
    refined_center = refine_cross_center_subpixel(gray, coarse_center)
    if refined_center:
        print(f"Refined (edge-weighted): ({refined_center[0]:.4f}, {refined_center[1]:.4f})")
    else:
        refined_center = coarse_center
    
    # Step 3: Model-based optimization
    final_center = fit_cross_model(gray, refined_center)
    if final_center:
        print(f"Final (model-fit): ({final_center[0]:.6f}, {final_center[1]:.6f})")
    else:
        final_center = refined_center
    
    print(f"\n{'='*50}")
    print(f"SUBPIXEL CENTER: x = {final_center[0]:.6f}, y = {final_center[1]:.6f}")
    print(f"{'='*50}")
    
    if debug:
        # Create visualization
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw cross contour if found
        if cross_contour is not None:
            cv2.drawContours(vis, [cross_contour], -1, (0, 255, 0), 1)
        
        # Draw center
        cx, cy = int(round(final_center[0])), int(round(final_center[1]))
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
        
        # Add text with coordinates
        text = f"({final_center[0]:.3f}, {final_center[1]:.3f})"
        cv2.putText(vis, text, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        output_path = image_path.rsplit('.', 1)[0] + '_cross_detected.png'
        cv2.imwrite(output_path, vis)
        print(f"Debug image saved: {output_path}")
    
    return final_center


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_faint_cross.py <image_path> [--debug]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    try:
        center = detect_cross_center(image_path, debug=debug)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
