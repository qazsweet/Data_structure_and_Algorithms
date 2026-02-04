#!/usr/bin/env python3
"""
Cross arm measurement script - designed to work with the attached cross image.

This script analyzes a grayscale image containing a cross marker and measures:
- Half-arm lengths from center to each tip (top, bottom, left, right)
- Full arm lengths (horizontal, vertical)
- Various arm ratios
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def measure_cross_image(gray):
    """
    Measure cross arms in a grayscale image.
    
    The algorithm:
    1. Apply edge detection to find the cross boundary
    2. Find contours and select the one most likely to be the cross
    3. Measure from center to extreme points in each direction
    """
    h, w = gray.shape
    print(f"Image size: {w} x {h} pixels")
    
    # Edge detection with tuned parameters for subtle cross edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Use multiple edge detection methods and combine
    # Method 1: Canny with adaptive thresholds
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges1 = cv2.Canny(blurred, lower, upper)
    
    # Method 2: Lower threshold Canny for subtle edges
    edges2 = cv2.Canny(blurred, 10, 50)
    
    # Method 3: Gradient magnitude
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
    _, edges3 = cv2.threshold(grad_norm, 10, 255, cv2.THRESH_BINARY)
    
    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)
    
    # Clean up with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("ERROR: No contours found")
        return None, edges
    
    print(f"Found {len(contours)} contours")
    
    # Find the cross contour - it should be:
    # - Near the center of the image
    # - Have a reasonable size
    # - Have roughly equal width and height (bounding box is square-ish)
    
    center_x, center_y = w / 2, h / 2
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            cx, cy = x + bw/2, y + bh/2
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        
        # Distance from image center
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        
        # Aspect ratio of bounding box
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        
        # Cross should have roughly square bounding box and be near center
        score = area * aspect / (1 + dist / 100)
        
        candidates.append((cnt, score, area, (cx, cy), (x, y, bw, bh)))
    
    if not candidates:
        print("ERROR: No valid cross contours found")
        return None, edges
    
    # Sort by score and take best candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_cnt = candidates[0][0]
    best_area = candidates[0][2]
    best_center = candidates[0][3]
    best_bbox = candidates[0][4]
    
    print(f"Best candidate: area={best_area}, center={best_center}, bbox={best_bbox}")
    
    # Measure the cross arms
    pts = best_cnt.reshape(-1, 2)
    cx, cy = best_center
    
    # Find extreme points
    top_idx = np.argmin(pts[:, 1])
    bottom_idx = np.argmax(pts[:, 1])
    left_idx = np.argmin(pts[:, 0])
    right_idx = np.argmax(pts[:, 0])
    
    top_pt = pts[top_idx]
    bottom_pt = pts[bottom_idx]
    left_pt = pts[left_idx]
    right_pt = pts[right_idx]
    
    # Calculate half-arm lengths
    top_arm = float(cy - top_pt[1])
    bottom_arm = float(bottom_pt[1] - cy)
    left_arm = float(cx - left_pt[0])
    right_arm = float(right_pt[0] - cx)
    
    results = {
        'center': (cx, cy),
        'half_arms': {
            'top': top_arm,
            'bottom': bottom_arm,
            'left': left_arm,
            'right': right_arm
        },
        'extreme_points': {
            'top': tuple(top_pt),
            'bottom': tuple(bottom_pt),
            'left': tuple(left_pt),
            'right': tuple(right_pt)
        },
        'full_arms': {
            'horizontal': left_arm + right_arm,
            'vertical': top_arm + bottom_arm
        },
        'contour': best_cnt
    }
    
    # Calculate ratios
    results['ratios'] = {}
    if right_arm > 0:
        results['ratios']['left/right'] = left_arm / right_arm
    if left_arm > 0:
        results['ratios']['right/left'] = right_arm / left_arm
    if bottom_arm > 0:
        results['ratios']['top/bottom'] = top_arm / bottom_arm
    if top_arm > 0:
        results['ratios']['bottom/top'] = bottom_arm / top_arm
    
    h_total = left_arm + right_arm
    v_total = top_arm + bottom_arm
    if v_total > 0:
        results['ratios']['horizontal/vertical'] = h_total / v_total
    if h_total > 0:
        results['ratios']['vertical/horizontal'] = v_total / h_total
    
    return results, edges


def print_results(results):
    """Print formatted measurement results."""
    if results is None:
        print("No results to display")
        return
    
    print()
    print("=" * 70)
    print("CROSS ARM MEASUREMENTS")
    print("=" * 70)
    
    cx, cy = results['center']
    print(f"\nCross center: ({cx:.2f}, {cy:.2f})")
    
    print("\n" + "-" * 70)
    print("HALF-ARM LENGTHS (center to tip)")
    print("-" * 70)
    arms = results['half_arms']
    print(f"  {'Top':12s}: {arms['top']:10.2f} pixels")
    print(f"  {'Bottom':12s}: {arms['bottom']:10.2f} pixels")
    print(f"  {'Left':12s}: {arms['left']:10.2f} pixels")
    print(f"  {'Right':12s}: {arms['right']:10.2f} pixels")
    
    print("\n" + "-" * 70)
    print("FULL ARM LENGTHS")
    print("-" * 70)
    full = results['full_arms']
    print(f"  Horizontal (L+R): {full['horizontal']:.2f} pixels")
    print(f"  Vertical (T+B):   {full['vertical']:.2f} pixels")
    
    print("\n" + "-" * 70)
    print("ARM RATIOS")
    print("-" * 70)
    for name, value in results['ratios'].items():
        print(f"  {name:25s}: {value:.4f}")
    
    print("\n" + "-" * 70)
    print("EXTREME POINTS (arm tips)")
    print("-" * 70)
    for pos, pt in results['extreme_points'].items():
        print(f"  {pos.capitalize():12s}: ({int(pt[0]):4d}, {int(pt[1]):4d})")
    
    print()
    print("=" * 70)


def save_visualization(gray, results, output_path):
    """Save visualization of measurements."""
    if results is None:
        return
    
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Draw contour
    cv2.drawContours(vis, [results['contour']], -1, (0, 255, 0), 2)
    
    cx, cy = results['center']
    
    # Draw center
    cv2.circle(vis, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    
    # Draw arm lines
    colors = {'top': (255, 0, 0), 'bottom': (255, 0, 255), 
              'left': (0, 255, 255), 'right': (0, 255, 0)}
    
    for direction, pt in results['extreme_points'].items():
        cv2.line(vis, (int(cx), int(cy)), (int(pt[0]), int(pt[1])), colors[direction], 2)
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, colors[direction], -1)
    
    # Add labels
    arms = results['half_arms']
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"T:{arms['top']:.1f}", (int(cx)+5, int(results['extreme_points']['top'][1])+15),
                font, 0.4, (255,255,255), 1)
    cv2.putText(vis, f"B:{arms['bottom']:.1f}", (int(cx)+5, int(results['extreme_points']['bottom'][1])-5),
                font, 0.4, (255,255,255), 1)
    cv2.putText(vis, f"L:{arms['left']:.1f}", (int(results['extreme_points']['left'][0])+5, int(cy)-5),
                font, 0.4, (255,255,255), 1)
    cv2.putText(vis, f"R:{arms['right']:.1f}", (int(results['extreme_points']['right'][0])-50, int(cy)-5),
                font, 0.4, (255,255,255), 1)
    
    cv2.imwrite(output_path, vis)
    print(f"Visualization saved to: {output_path}")


def main():
    """Main entry point."""
    # Try to find cross image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Look for cross image in workspace
        workspace = Path('/workspace')
        img_path = None
        
        for pattern in ['cross*.png', 'cross*.jpg', 'attached*.png', '*.png']:
            files = list(workspace.glob(pattern))
            for f in files:
                if 'measurement' not in f.name and 'edge' not in f.name:
                    img_path = str(f)
                    break
            if img_path:
                break
        
        if not img_path:
            print("Usage: python measure_attached_cross.py <image_path>")
            print("No cross image found in workspace")
            return 1
    
    print(f"Loading image: {img_path}")
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not read image: {img_path}")
        return 1
    
    results, edges = measure_cross_image(img)
    
    # Save edges for debugging
    cv2.imwrite('/workspace/detected_edges.png', edges)
    
    if results:
        print_results(results)
        save_visualization(img, results, '/workspace/cross_measurement_result.png')
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
