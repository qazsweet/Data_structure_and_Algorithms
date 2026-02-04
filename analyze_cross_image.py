#!/usr/bin/env python3
"""
Analyze a cross marker image to measure half-arm lengths and ratios.
Designed for the specific cross image provided by the user.
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def load_and_preprocess(image_path):
    """Load image and convert to grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Handle different image types
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:  # BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return img, gray


def find_cross_by_edge_detection(gray):
    """
    Detect the cross shape using edge detection.
    Returns contour and edge information.
    """
    h, w = gray.shape
    
    # Multiple edge detection approaches for robustness
    # 1. Canny with adaptive thresholds
    median_val = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    
    edges_canny = cv2.Canny(gray, lower, upper)
    
    # 2. Gradient-based edge detection
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 1)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad_mag / (grad_mag.max() + 1e-6) * 255).astype(np.uint8)
    _, edges_grad = cv2.threshold(grad_norm, 15, 255, cv2.THRESH_BINARY)
    
    # Combine edges
    edges_combined = cv2.bitwise_or(edges_canny, edges_grad)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    edges_clean = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
    
    return edges_clean, edges_canny, edges_grad


def find_cross_contour(edges, gray_shape):
    """Find the contour most likely to be the cross."""
    h, w = gray_shape
    center_x, center_y = w / 2, h / 2
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Score contours based on area, shape, and position
    best_contour = None
    best_score = -1
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area < 500:  # Too small
            continue
        
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # Check bounding box aspect ratio (cross should be roughly square)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh)
        
        # Score: larger area, closer to center, more square-ish bounding box
        dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        normalized_dist = dist_to_center / max(h, w)
        
        score = area * aspect / (1 + normalized_dist * 2)
        
        if score > best_score:
            best_score = score
            best_contour = cnt
    
    return best_contour


def measure_cross_arms(contour, center=None):
    """
    Measure the four half-arms of a cross from its contour.
    
    Args:
        contour: OpenCV contour of the cross
        center: Optional (cx, cy) tuple; if None, calculated from contour moments
    
    Returns:
        Dictionary with arm measurements and analysis
    """
    pts = contour.reshape(-1, 2)
    
    # Calculate center if not provided
    if center is None:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
    else:
        cx, cy = center
    
    # Find extreme points - these are the tips of the arms
    top_idx = np.argmin(pts[:, 1])
    bottom_idx = np.argmax(pts[:, 1])
    left_idx = np.argmin(pts[:, 0])
    right_idx = np.argmax(pts[:, 0])
    
    top_pt = pts[top_idx]
    bottom_pt = pts[bottom_idx]
    left_pt = pts[left_idx]
    right_pt = pts[right_idx]
    
    # Calculate half-arm lengths (from center to tip)
    top_arm = cy - top_pt[1]
    bottom_arm = bottom_pt[1] - cy
    left_arm = cx - left_pt[0]
    right_arm = right_pt[0] - cx
    
    results = {
        'center': (float(cx), float(cy)),
        'half_arms': {
            'top': float(top_arm),
            'bottom': float(bottom_arm),
            'left': float(left_arm),
            'right': float(right_arm)
        },
        'extreme_points': {
            'top': (int(top_pt[0]), int(top_pt[1])),
            'bottom': (int(bottom_pt[0]), int(bottom_pt[1])),
            'left': (int(left_pt[0]), int(left_pt[1])),
            'right': (int(right_pt[0]), int(right_pt[1]))
        }
    }
    
    # Calculate full arm lengths
    h_total = left_arm + right_arm
    v_total = top_arm + bottom_arm
    
    results['full_arms'] = {
        'horizontal': float(h_total),
        'vertical': float(v_total)
    }
    
    # Calculate ratios
    results['ratios'] = {
        'left_to_right': float(left_arm / right_arm) if right_arm > 0 else float('inf'),
        'right_to_left': float(right_arm / left_arm) if left_arm > 0 else float('inf'),
        'top_to_bottom': float(top_arm / bottom_arm) if bottom_arm > 0 else float('inf'),
        'bottom_to_top': float(bottom_arm / top_arm) if top_arm > 0 else float('inf'),
        'horizontal_to_vertical': float(h_total / v_total) if v_total > 0 else float('inf'),
        'vertical_to_horizontal': float(v_total / h_total) if h_total > 0 else float('inf'),
    }
    
    return results


def visualize_results(original_img, gray, contour, results, output_path):
    """Create visualization of the cross analysis."""
    if len(original_img.shape) == 2:
        vis = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 4:
        vis = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
    else:
        vis = original_img.copy()
    
    # Draw contour
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    
    # Get measurements
    cx, cy = results['center']
    arms = results['half_arms']
    extremes = results['extreme_points']
    
    # Draw center point
    cv2.circle(vis, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cv2.circle(vis, (int(cx), int(cy)), 10, (255, 255, 255), 2)
    
    # Draw arm lines with different colors
    colors = {
        'top': (255, 0, 0),      # Blue
        'bottom': (255, 0, 255), # Magenta
        'left': (0, 255, 255),   # Yellow
        'right': (0, 255, 0)     # Green
    }
    
    for direction, pt in extremes.items():
        color = colors[direction]
        cv2.line(vis, (int(cx), int(cy)), pt, color, 2)
        cv2.circle(vis, pt, 5, color, -1)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Top label
    cv2.putText(vis, f"Top: {arms['top']:.1f}px", 
                (extremes['top'][0] + 10, extremes['top'][1] + 20),
                font, font_scale, (255, 255, 255), thickness)
    
    # Bottom label  
    cv2.putText(vis, f"Bottom: {arms['bottom']:.1f}px",
                (extremes['bottom'][0] + 10, extremes['bottom'][1] - 10),
                font, font_scale, (255, 255, 255), thickness)
    
    # Left label
    cv2.putText(vis, f"Left: {arms['left']:.1f}px",
                (extremes['left'][0] + 10, extremes['left'][1] - 10),
                font, font_scale, (255, 255, 255), thickness)
    
    # Right label
    cv2.putText(vis, f"Right: {arms['right']:.1f}px",
                (extremes['right'][0] - 120, extremes['right'][1] - 10),
                font, font_scale, (255, 255, 255), thickness)
    
    # Save visualization
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to: {output_path}")
    
    return vis


def print_results(results, image_shape):
    """Print formatted results."""
    print()
    print("=" * 70)
    print("CROSS ARM MEASUREMENT RESULTS")
    print("=" * 70)
    print()
    print(f"Image dimensions: {image_shape[1]} x {image_shape[0]} pixels")
    print(f"Cross center: ({results['center'][0]:.2f}, {results['center'][1]:.2f})")
    print()
    
    print("-" * 70)
    print("HALF-ARM LENGTHS (from center to arm tip)")
    print("-" * 70)
    arms = results['half_arms']
    for direction in ['top', 'bottom', 'left', 'right']:
        length = arms[direction]
        print(f"  {direction.upper():8s}: {length:8.2f} pixels")
    
    print()
    print("-" * 70)
    print("FULL ARM LENGTHS")
    print("-" * 70)
    full = results['full_arms']
    print(f"  Horizontal (left + right): {full['horizontal']:.2f} pixels")
    print(f"  Vertical (top + bottom):   {full['vertical']:.2f} pixels")
    
    print()
    print("-" * 70)
    print("ARM RATIOS")
    print("-" * 70)
    ratios = results['ratios']
    print(f"  Left / Right:              {ratios['left_to_right']:.4f}")
    print(f"  Right / Left:              {ratios['right_to_left']:.4f}")
    print(f"  Top / Bottom:              {ratios['top_to_bottom']:.4f}")
    print(f"  Bottom / Top:              {ratios['bottom_to_top']:.4f}")
    print(f"  Horizontal / Vertical:     {ratios['horizontal_to_vertical']:.4f}")
    print(f"  Vertical / Horizontal:     {ratios['vertical_to_horizontal']:.4f}")
    
    print()
    print("-" * 70)
    print("EXTREME POINTS (arm tips)")
    print("-" * 70)
    for direction, pt in results['extreme_points'].items():
        print(f"  {direction.upper():8s}: ({pt[0]:4d}, {pt[1]:4d})")
    
    print()
    print("=" * 70)


def analyze_cross(image_path, output_viz=True):
    """Main analysis function."""
    # Load image
    img, gray = load_and_preprocess(image_path)
    print(f"Loaded image: {image_path}")
    print(f"  Dimensions: {gray.shape[1]} x {gray.shape[0]}")
    
    # Detect edges
    edges, edges_canny, edges_grad = find_cross_by_edge_detection(gray)
    
    # Find cross contour
    contour = find_cross_contour(edges, gray.shape)
    
    if contour is None:
        print("ERROR: Could not detect cross contour in image")
        # Save debug images
        cv2.imwrite('/workspace/debug_edges.png', edges)
        cv2.imwrite('/workspace/debug_canny.png', edges_canny)
        cv2.imwrite('/workspace/debug_grad.png', edges_grad)
        print("Saved debug edge images")
        return None
    
    # Measure arms
    results = measure_cross_arms(contour)
    
    # Print results
    print_results(results, gray.shape)
    
    # Create visualization
    if output_viz:
        stem = Path(image_path).stem
        viz_path = f'/workspace/cross_measurement_{stem}.png'
        visualize_results(img, gray, contour, results, viz_path)
        
        # Also save edges for reference
        cv2.imwrite(f'/workspace/edges_{stem}.png', edges)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_cross_image.py <image_path>")
        print()
        print("Attempting to find cross images in workspace...")
        
        # Look for images in workspace
        workspace = Path('/workspace')
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            for img_path in workspace.glob(ext):
                if 'cross' in img_path.name.lower() or 'measurement' not in img_path.name.lower():
                    print(f"\nAnalyzing: {img_path}")
                    analyze_cross(str(img_path))
                    break
    else:
        analyze_cross(sys.argv[1])
