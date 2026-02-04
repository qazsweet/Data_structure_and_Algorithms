#!/usr/bin/env python3
"""
Measure the half-arm lengths of a cross shape in an image.

This script detects a cross (+) shape in a grayscale image and measures:
- The length of each half-arm (top, bottom, left, right from center)
- Arm ratios between different arms
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def find_cross_center_and_arms(gray_img):
    """
    Detect the cross shape and measure its arms.
    
    Returns:
        center: (x, y) tuple of center coordinates
        arms: dict with 'top', 'bottom', 'left', 'right' half-arm lengths
    """
    # Threshold to find the cross region (cross is lighter than background on edges)
    # First, let's detect edges
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1)
    
    # Use adaptive thresholding or Canny to find edges
    edges = cv2.Canny(blurred, 30, 100)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Try different approach - look for the cross shape directly
        # The cross appears lighter with distinct boundaries
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (should be the cross)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rect and moments for center
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2
        
        return (cx, cy), largest_contour
    
    return None, None


def detect_cross_edges(gray_img):
    """
    Detect the cross shape using edge detection and find its boundaries.
    """
    # Apply multiple preprocessing steps
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 1)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Edge magnitude
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_mag = (edge_mag / edge_mag.max() * 255).astype(np.uint8)
    
    # Threshold edges
    _, edge_binary = cv2.threshold(edge_mag, 30, 255, cv2.THRESH_BINARY)
    
    return edge_binary, sobelx, sobely


def measure_cross_from_edges(gray_img):
    """
    Measure the cross arms by detecting edges and tracing from center.
    """
    h, w = gray_img.shape
    
    # Detect edges
    edge_binary, sobelx, sobely = detect_cross_edges(gray_img)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edge_binary, 1, np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return None
    
    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 20 or angle > 160:  # Horizontal
            h_lines.append(line[0])
        elif 70 < angle < 110:  # Vertical
            v_lines.append(line[0])
    
    return h_lines, v_lines, edge_binary


def find_cross_shape_refined(gray_img):
    """
    Find the cross shape using contour analysis with refined edge detection.
    """
    h, w = gray_img.shape
    
    # Use Canny with automatic thresholds
    median_val = np.median(gray_img)
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))
    
    edges = cv2.Canny(gray_img, lower, upper)
    
    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, edges


def analyze_cross_contour(contour, img_shape):
    """
    Analyze a cross-shaped contour to extract arm measurements.
    """
    # Get contour bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Get contour moments for centroid
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = x + w // 2
        cy = y + h // 2
    
    # For a cross shape, measure from centroid to edges
    # Trace along horizontal and vertical lines
    
    return cx, cy, x, y, w, h


def measure_arms_from_contour(gray_img, contour):
    """
    Given a contour approximating the cross, measure arm lengths from the centroid.
    """
    h, w = gray_img.shape
    
    # Get centroid
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw / 2, y + bh / 2
    
    # Convert contour points to array
    pts = contour.reshape(-1, 2)
    
    # Find extreme points in each direction from centroid
    # Right arm: max x where y is close to cy
    tolerance = 20  # pixels tolerance for arm width
    
    # Points near horizontal center line
    h_pts = pts[np.abs(pts[:, 1] - cy) < tolerance]
    # Points near vertical center line  
    v_pts = pts[np.abs(pts[:, 0] - cx) < tolerance]
    
    results = {
        'center': (cx, cy),
        'arms': {}
    }
    
    if len(h_pts) > 0:
        right_pts = h_pts[h_pts[:, 0] > cx]
        left_pts = h_pts[h_pts[:, 0] < cx]
        
        if len(right_pts) > 0:
            results['arms']['right'] = float(np.max(right_pts[:, 0]) - cx)
        if len(left_pts) > 0:
            results['arms']['left'] = float(cx - np.min(left_pts[:, 0]))
    
    if len(v_pts) > 0:
        bottom_pts = v_pts[v_pts[:, 1] > cy]
        top_pts = v_pts[v_pts[:, 1] < cy]
        
        if len(bottom_pts) > 0:
            results['arms']['bottom'] = float(np.max(bottom_pts[:, 1]) - cy)
        if len(top_pts) > 0:
            results['arms']['top'] = float(cy - np.min(top_pts[:, 1]))
    
    return results


def detect_cross_by_intensity(gray_img):
    """
    Detect the cross by analyzing intensity profiles and finding transitions.
    This is more robust for subtle crosses.
    """
    h, w = gray_img.shape
    
    # Find approximate center by looking for the cross pattern
    # The cross center should be where horizontal and vertical edge patterns intersect
    
    # Apply edge detection
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1)
    edges = cv2.Canny(blurred, 20, 80)
    
    # Find all edge points
    edge_points = np.column_stack(np.where(edges > 0))
    
    if len(edge_points) == 0:
        return None
    
    # Cluster edge points to find the cross structure
    # For a cross, we expect vertical and horizontal clusters
    
    # Find horizontal and vertical projections of edges
    h_proj = np.sum(edges, axis=1)  # sum along rows
    v_proj = np.sum(edges, axis=0)  # sum along columns
    
    # Find peaks in projections (these indicate arm positions)
    # For a cross, there should be strong edge responses at the arm boundaries
    
    return edges, h_proj, v_proj


def refined_cross_measurement(gray_img):
    """
    Main function to measure cross arms with refined analysis.
    """
    h, w = gray_img.shape
    
    # Step 1: Detect edges
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 1)
    edges = cv2.Canny(blurred, 15, 60)
    
    # Step 2: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("No contours found")
        return None
    
    # Step 3: Find the cross contour (should be roughly in the center and cross-shaped)
    # Filter for contours that could be the cross
    center_x, center_y = w / 2, h / 2
    cross_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Too small
            continue
            
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw / 2, y + bh / 2
        
        # Check if near center of image
        dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        if dist_to_center < min(w, h) * 0.4:  # Within 40% of image center
            cross_candidates.append((cnt, area, dist_to_center))
    
    if not cross_candidates:
        # Return all contours for debugging
        return contours, edges, None
    
    # Sort by area (largest first)
    cross_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Take the largest contour near center
    cross_contour = cross_candidates[0][0]
    
    return contours, edges, cross_contour


def measure_by_edge_tracing(gray_img, visualize=False):
    """
    Measure cross by tracing edges from detected center.
    """
    h, w = gray_img.shape
    
    # Edge detection with multiple methods
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.5)
    
    # Canny edges
    edges = cv2.Canny(blurred, 10, 50)
    
    # Find all edge pixels
    edge_y, edge_x = np.where(edges > 0)
    
    if len(edge_x) == 0:
        return None
    
    # Estimate cross center as centroid of all edge points
    cx = np.mean(edge_x)
    cy = np.mean(edge_y)
    
    # Create edge map
    edge_map = edges > 0
    
    # Trace from center outward in four directions
    def trace_direction(start_x, start_y, dx, dy, max_dist=500):
        """Trace until hitting an edge."""
        x, y = start_x, start_y
        dist = 0
        
        while 0 <= x < w and 0 <= y < h and dist < max_dist:
            if edge_map[int(y), int(x)]:
                return dist
            x += dx
            y += dy
            dist += 1
        
        return dist
    
    # Measure in each direction
    right_dist = trace_direction(cx, cy, 1, 0)
    left_dist = trace_direction(cx, cy, -1, 0)
    down_dist = trace_direction(cx, cy, 0, 1)
    up_dist = trace_direction(cx, cy, 0, -1)
    
    results = {
        'center': (cx, cy),
        'half_arms': {
            'right': right_dist,
            'left': left_dist,
            'bottom': down_dist,
            'top': up_dist
        },
        'full_arms': {
            'horizontal': left_dist + right_dist,
            'vertical': up_dist + down_dist
        }
    }
    
    if visualize:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        
        # Draw arm extents
        cv2.line(vis, (int(cx), int(cy)), (int(cx + right_dist), int(cy)), (0, 0, 255), 2)
        cv2.line(vis, (int(cx), int(cy)), (int(cx - left_dist), int(cy)), (255, 0, 0), 2)
        cv2.line(vis, (int(cx), int(cy)), (int(cx), int(cy + down_dist)), (0, 255, 255), 2)
        cv2.line(vis, (int(cx), int(cy)), (int(cx), int(cy - up_dist)), (255, 255, 0), 2)
        
        cv2.imwrite('/workspace/cross_measurement_viz.png', vis)
        cv2.imwrite('/workspace/cross_edges.png', edges)
    
    return results


def find_cross_boundary_refined(gray_img, visualize=True):
    """
    Find the cross boundary using gradient analysis.
    More robust for faint crosses.
    """
    h, w = gray_img.shape
    
    # Compute gradients
    blurred = cv2.GaussianBlur(gray_img.astype(np.float32), (3, 3), 1)
    
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize
    grad_mag_norm = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
    
    # Threshold to find significant gradients
    _, grad_binary = cv2.threshold(grad_mag_norm, 20, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    contours, _ = cv2.findContours(grad_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the contour most likely to be the cross (largest, roughly centered)
    best_contour = None
    best_score = -1
    center_x, center_y = w / 2, h / 2
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Too small
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            continue
        
        # Score based on area and proximity to center
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        score = area / (1 + dist)
        
        if score > best_score:
            best_score = score
            best_contour = cnt
    
    if best_contour is None:
        return None
    
    # Analyze the cross contour
    # For a cross shape, approximate to polygon
    epsilon = 0.02 * cv2.arcLength(best_contour, True)
    approx = cv2.approxPolyDP(best_contour, epsilon, True)
    
    # Get bounding box
    x, y, bw, bh = cv2.boundingRect(best_contour)
    
    # Get centroid
    M = cv2.moments(best_contour)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Find extremes of the cross in each direction
    pts = best_contour.reshape(-1, 2)
    
    # Find arm tips (extreme points in each direction)
    top_idx = np.argmin(pts[:, 1])
    bottom_idx = np.argmax(pts[:, 1])
    left_idx = np.argmin(pts[:, 0])
    right_idx = np.argmax(pts[:, 0])
    
    top_pt = pts[top_idx]
    bottom_pt = pts[bottom_idx]
    left_pt = pts[left_idx]
    right_pt = pts[right_idx]
    
    # Calculate half-arm lengths from center
    top_arm = cy - top_pt[1]
    bottom_arm = bottom_pt[1] - cy
    left_arm = cx - left_pt[0]
    right_arm = right_pt[0] - cx
    
    results = {
        'center': (cx, cy),
        'half_arms': {
            'top': float(top_arm),
            'bottom': float(bottom_arm),
            'left': float(left_arm),
            'right': float(right_arm)
        },
        'extreme_points': {
            'top': tuple(top_pt),
            'bottom': tuple(bottom_pt),
            'left': tuple(left_pt),
            'right': tuple(right_pt)
        },
        'bounding_box': (x, y, bw, bh),
        'contour': best_contour
    }
    
    # Calculate ratios
    arms = results['half_arms']
    h_total = arms['left'] + arms['right']
    v_total = arms['top'] + arms['bottom']
    
    results['ratios'] = {
        'horizontal_total': h_total,
        'vertical_total': v_total,
        'h_to_v': h_total / v_total if v_total > 0 else 0,
        'left_to_right': arms['left'] / arms['right'] if arms['right'] > 0 else 0,
        'top_to_bottom': arms['top'] / arms['bottom'] if arms['bottom'] > 0 else 0
    }
    
    if visualize:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        
        # Draw contour
        cv2.drawContours(vis, [best_contour], -1, (0, 255, 0), 2)
        
        # Draw center
        cv2.circle(vis, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        
        # Draw arm lines
        cv2.line(vis, (int(cx), int(cy)), tuple(top_pt), (255, 0, 0), 2)
        cv2.line(vis, (int(cx), int(cy)), tuple(bottom_pt), (255, 0, 0), 2)
        cv2.line(vis, (int(cx), int(cy)), tuple(left_pt), (0, 255, 255), 2)
        cv2.line(vis, (int(cx), int(cy)), tuple(right_pt), (0, 255, 255), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, f"T:{arms['top']:.1f}", (int(cx)+10, int(top_pt[1])+20), 
                    font, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, f"B:{arms['bottom']:.1f}", (int(cx)+10, int(bottom_pt[1])-10), 
                    font, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, f"L:{arms['left']:.1f}", (int(left_pt[0])+10, int(cy)-10), 
                    font, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, f"R:{arms['right']:.1f}", (int(right_pt[0])-80, int(cy)-10), 
                    font, 0.5, (255, 255, 255), 2)
        
        cv2.imwrite('/workspace/cross_analysis.png', vis)
        cv2.imwrite('/workspace/gradient_edges.png', grad_binary)
        
        print(f"Saved visualization to /workspace/cross_analysis.png")
    
    return results


def main(image_path):
    """Main function to analyze cross in image."""
    # Read image
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"Image size: {gray.shape[1]} x {gray.shape[0]} pixels")
    print()
    
    # Analyze the cross
    results = find_cross_boundary_refined(gray, visualize=True)
    
    if results is None:
        print("Could not detect cross in image")
        return
    
    # Print results
    print("=" * 60)
    print("CROSS ARM MEASUREMENTS")
    print("=" * 60)
    print()
    print(f"Cross center: ({results['center'][0]:.2f}, {results['center'][1]:.2f})")
    print()
    print("Half-arm lengths (from center to tip):")
    print("-" * 40)
    for arm, length in results['half_arms'].items():
        print(f"  {arm.capitalize():8s}: {length:.2f} pixels")
    
    print()
    print("Full arm lengths:")
    print("-" * 40)
    h_total = results['half_arms']['left'] + results['half_arms']['right']
    v_total = results['half_arms']['top'] + results['half_arms']['bottom']
    print(f"  Horizontal (left + right): {h_total:.2f} pixels")
    print(f"  Vertical (top + bottom):   {v_total:.2f} pixels")
    
    print()
    print("Arm ratios:")
    print("-" * 40)
    print(f"  Left / Right:     {results['ratios']['left_to_right']:.4f}")
    print(f"  Top / Bottom:     {results['ratios']['top_to_bottom']:.4f}")
    print(f"  Horizontal / Vertical: {results['ratios']['h_to_v']:.4f}")
    
    print()
    print("Extreme points:")
    print("-" * 40)
    for pos, pt in results['extreme_points'].items():
        print(f"  {pos.capitalize():8s}: ({pt[0]}, {pt[1]})")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to looking for cross image
        image_path = "/workspace/cross_image.png"
    else:
        image_path = sys.argv[1]
    
    main(image_path)
