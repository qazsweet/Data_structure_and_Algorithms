"""
Measure the length of each half arm of a cross shape in an image.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and return both color and grayscale versions."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def find_cross_center_and_arms(gray: np.ndarray, debug: bool = False) -> Dict:
    """
    Detect the cross shape and measure each half arm.
    
    Returns a dictionary with:
    - center: (x, y) center coordinates
    - arms: dict with 'top', 'bottom', 'left', 'right' lengths in pixels
    - ratios: dict with arm ratios
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying illumination
    # The cross appears lighter than background based on the image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, -5
    )
    
    # Also try Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use edge detection to find the cross outline
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to connect them
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Try with different approach - look for the cross shape directly
        # Apply morphological operations
        kernel_close = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the cross contour (should be one of the larger contours with cross-like shape)
    cross_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > 100:  # Minimum area threshold
            # Check if the contour could be a cross shape
            # A cross has a specific perimeter to area ratio
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Cross shapes have lower circularity than circles
                if circularity < 0.7:
                    max_area = area
                    cross_contour = contour
    
    if cross_contour is None and contours:
        # Fall back to largest contour
        cross_contour = max(contours, key=cv2.contourArea)
    
    if cross_contour is None:
        raise ValueError("Could not find cross shape in image")
    
    # Get bounding rect and moments
    moments = cv2.moments(cross_contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # Use bounding rect center
        x, y, w, h = cv2.boundingRect(cross_contour)
        cx, cy = x + w // 2, y + h // 2
    
    # Create a mask from the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cross_contour], -1, 255, -1)
    
    # Measure arms by scanning from center in each direction
    h, w = gray.shape
    
    # For better accuracy, use the edge image
    # Find edges of the cross
    cross_edges = cv2.Canny(mask, 50, 150)
    
    # Measure each half arm by finding the distance from center to edge
    def measure_arm(direction: str) -> float:
        """Measure distance from center to edge in given direction."""
        if direction == "top":
            # Scan upward from center
            for i, y in enumerate(range(cy, -1, -1)):
                if mask[y, cx] == 0:
                    return float(i)
            return float(cy)
        elif direction == "bottom":
            # Scan downward from center
            for i, y in enumerate(range(cy, h)):
                if mask[y, cx] == 0:
                    return float(i)
            return float(h - cy)
        elif direction == "left":
            # Scan left from center
            for i, x in enumerate(range(cx, -1, -1)):
                if mask[cy, x] == 0:
                    return float(i)
            return float(cx)
        elif direction == "right":
            # Scan right from center
            for i, x in enumerate(range(cx, w)):
                if mask[cy, x] == 0:
                    return float(i)
            return float(w - cx)
        return 0.0
    
    # More robust measurement: average over multiple scan lines
    def measure_arm_robust(direction: str, num_lines: int = 5) -> float:
        """Measure arm length averaging over multiple parallel scan lines."""
        measurements = []
        
        if direction in ["top", "bottom"]:
            offsets = range(-num_lines // 2, num_lines // 2 + 1)
            for dx in offsets:
                x = cx + dx
                if 0 <= x < w:
                    if direction == "top":
                        for i, y in enumerate(range(cy, -1, -1)):
                            if mask[y, x] == 0:
                                measurements.append(i)
                                break
                        else:
                            measurements.append(cy)
                    else:  # bottom
                        for i, y in enumerate(range(cy, h)):
                            if mask[y, x] == 0:
                                measurements.append(i)
                                break
                        else:
                            measurements.append(h - cy)
        else:  # left or right
            offsets = range(-num_lines // 2, num_lines // 2 + 1)
            for dy in offsets:
                y = cy + dy
                if 0 <= y < h:
                    if direction == "left":
                        for i, x in enumerate(range(cx, -1, -1)):
                            if mask[y, x] == 0:
                                measurements.append(i)
                                break
                        else:
                            measurements.append(cx)
                    else:  # right
                        for i, x in enumerate(range(cx, w)):
                            if mask[y, x] == 0:
                                measurements.append(i)
                                break
                        else:
                            measurements.append(w - cx)
        
        if measurements:
            return float(np.median(measurements))
        return measure_arm(direction)
    
    arms = {
        "top": measure_arm_robust("top"),
        "bottom": measure_arm_robust("bottom"),
        "left": measure_arm_robust("left"),
        "right": measure_arm_robust("right"),
    }
    
    # Calculate ratios
    vertical_total = arms["top"] + arms["bottom"]
    horizontal_total = arms["left"] + arms["right"]
    
    ratios = {
        "top_to_bottom": arms["top"] / arms["bottom"] if arms["bottom"] > 0 else float("inf"),
        "left_to_right": arms["left"] / arms["right"] if arms["right"] > 0 else float("inf"),
        "vertical_to_horizontal": vertical_total / horizontal_total if horizontal_total > 0 else float("inf"),
        "top_to_vertical": arms["top"] / vertical_total if vertical_total > 0 else float("inf"),
        "bottom_to_vertical": arms["bottom"] / vertical_total if vertical_total > 0 else float("inf"),
        "left_to_horizontal": arms["left"] / horizontal_total if horizontal_total > 0 else float("inf"),
        "right_to_horizontal": arms["right"] / horizontal_total if horizontal_total > 0 else float("inf"),
    }
    
    return {
        "center": (cx, cy),
        "arms": arms,
        "ratios": ratios,
        "contour": cross_contour,
        "mask": mask,
    }


def measure_cross_from_edges(gray: np.ndarray) -> Dict:
    """
    Alternative approach: detect cross by finding its edges directly using 
    gradient analysis.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 20, 80)
    
    # Find the cross region by looking at the intensity profile
    h, w = gray.shape
    
    # The cross appears as a lighter region in the center of the image
    # Find the approximate center by looking for the region of interest
    
    # Use Hough lines to detect the cross edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
    
    if lines is None or len(lines) < 4:
        raise ValueError("Could not detect enough lines for cross measurement")
    
    # Separate lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:  # Horizontal
            horizontal_lines.append((x1, y1, x2, y2))
        elif 60 < angle < 120:  # Vertical
            vertical_lines.append((x1, y1, x2, y2))
    
    # Find the bounding lines of the cross
    # Group horizontal lines by their y-coordinate
    def get_cross_boundaries(gray: np.ndarray, edges: np.ndarray) -> Dict:
        """Find cross boundaries using profile analysis."""
        h, w = gray.shape
        
        # Find the approximate center of the cross
        # Sum columns and rows to find the cross location
        col_profile = np.sum(edges, axis=0)
        row_profile = np.sum(edges, axis=1)
        
        # Find peaks in the profiles (these correspond to cross edges)
        def find_peaks(profile, min_distance=20):
            peaks = []
            for i in range(1, len(profile) - 1):
                if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                    if profile[i] > np.mean(profile):
                        if not peaks or i - peaks[-1] > min_distance:
                            peaks.append(i)
            return peaks
        
        col_peaks = find_peaks(col_profile)
        row_peaks = find_peaks(row_profile)
        
        return col_peaks, row_peaks
    
    col_peaks, row_peaks = get_cross_boundaries(gray, edges)
    
    # Return available information
    return {
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "col_peaks": col_peaks,
        "row_peaks": row_peaks,
    }


def refined_cross_measurement(gray: np.ndarray) -> Dict:
    """
    Refined approach combining contour detection with edge analysis.
    """
    h, w = gray.shape
    
    # Apply bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use Otsu's thresholding
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # The cross is lighter than background, so we need to look at the bright regions
    # But with varying illumination, let's try both
    
    # Invert if needed (cross should be the foreground)
    # Check which version has more distinct cross shape
    
    # Find contours in both versions
    contours_normal, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    binary_inv = cv2.bitwise_not(binary)
    contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    def find_cross_contour(contours):
        """Find the contour most likely to be the cross."""
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small
                continue
            if area > h * w * 0.5:  # Too large (background)
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1:
                continue
                
            # Cross shape characteristics:
            # 1. Has 12 vertices (or close to it)
            # 2. Has specific aspect ratio
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # A cross typically has 12 corners
            num_vertices = len(approx)
            vertex_score = 1.0 - abs(num_vertices - 12) / 12.0
            
            # Check bounding box aspect ratio (should be roughly square for a symmetric cross)
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # A cross has lower solidity than a square (has the indentations)
            # Typical cross solidity is around 0.55-0.7
            solidity_score = 1.0 - abs(solidity - 0.6) / 0.4
            
            score = vertex_score * 0.3 + aspect * 0.3 + solidity_score * 0.4 + area / (h * w) * 0.1
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour, best_score
    
    contour_normal, score_normal = find_cross_contour(contours_normal)
    contour_inv, score_inv = find_cross_contour(contours_inv)
    
    if score_inv > score_normal:
        cross_contour = contour_inv
        mask = binary_inv
    else:
        cross_contour = contour_normal
        mask = binary
    
    if cross_contour is None:
        raise ValueError("Could not find cross shape")
    
    # Get the center
    moments = cv2.moments(cross_contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        x, y, bw, bh = cv2.boundingRect(cross_contour)
        cx, cy = x + bw // 2, y + bh // 2
    
    # Create filled mask
    filled_mask = np.zeros_like(gray)
    cv2.drawContours(filled_mask, [cross_contour], -1, 255, -1)
    
    # Measure arms from center
    def measure_arm_from_mask(mask, cx, cy, direction, num_samples=11):
        h, w = mask.shape
        measurements = []
        
        half_samples = num_samples // 2
        
        if direction == "top":
            for dx in range(-half_samples, half_samples + 1):
                x = cx + dx
                if 0 <= x < w:
                    for dist, y in enumerate(range(cy, -1, -1)):
                        if mask[y, x] == 0:
                            measurements.append(dist)
                            break
                    else:
                        measurements.append(cy)
        elif direction == "bottom":
            for dx in range(-half_samples, half_samples + 1):
                x = cx + dx
                if 0 <= x < w:
                    for dist, y in enumerate(range(cy, h)):
                        if mask[y, x] == 0:
                            measurements.append(dist)
                            break
                    else:
                        measurements.append(h - cy)
        elif direction == "left":
            for dy in range(-half_samples, half_samples + 1):
                y = cy + dy
                if 0 <= y < h:
                    for dist, x in enumerate(range(cx, -1, -1)):
                        if mask[y, x] == 0:
                            measurements.append(dist)
                            break
                    else:
                        measurements.append(cx)
        elif direction == "right":
            for dy in range(-half_samples, half_samples + 1):
                y = cy + dy
                if 0 <= y < h:
                    for dist, x in enumerate(range(cx, w)):
                        if mask[y, x] == 0:
                            measurements.append(dist)
                            break
                    else:
                        measurements.append(w - cx)
        
        return float(np.median(measurements)) if measurements else 0.0
    
    arms = {
        "top": measure_arm_from_mask(filled_mask, cx, cy, "top"),
        "bottom": measure_arm_from_mask(filled_mask, cx, cy, "bottom"),
        "left": measure_arm_from_mask(filled_mask, cx, cy, "left"),
        "right": measure_arm_from_mask(filled_mask, cx, cy, "right"),
    }
    
    # Calculate ratios
    vertical_total = arms["top"] + arms["bottom"]
    horizontal_total = arms["left"] + arms["right"]
    
    ratios = {
        "top_to_bottom": arms["top"] / arms["bottom"] if arms["bottom"] > 0 else float("inf"),
        "left_to_right": arms["left"] / arms["right"] if arms["right"] > 0 else float("inf"),
        "vertical_to_horizontal": vertical_total / horizontal_total if horizontal_total > 0 else float("inf"),
    }
    
    return {
        "center": (cx, cy),
        "arms": arms,
        "ratios": ratios,
        "contour": cross_contour,
        "mask": filled_mask,
    }


def visualize_measurements(bgr: np.ndarray, results: Dict, output_path: str = None) -> np.ndarray:
    """Create visualization of the measurements."""
    vis = bgr.copy()
    
    cx, cy = results["center"]
    arms = results["arms"]
    
    # Draw the center
    cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
    
    # Draw the arm measurements
    colors = {
        "top": (255, 0, 0),      # Blue
        "bottom": (0, 255, 255),  # Yellow
        "left": (0, 0, 255),      # Red
        "right": (255, 0, 255),   # Magenta
    }
    
    # Draw lines for each arm
    cv2.line(vis, (cx, cy), (cx, int(cy - arms["top"])), colors["top"], 2)
    cv2.line(vis, (cx, cy), (cx, int(cy + arms["bottom"])), colors["bottom"], 2)
    cv2.line(vis, (cx, cy), (int(cx - arms["left"]), cy), colors["left"], 2)
    cv2.line(vis, (cx, cy), (int(cx + arms["right"]), cy), colors["right"], 2)
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"Top: {arms['top']:.1f}px", (cx + 10, cy - int(arms['top']/2)), 
                font, 0.5, colors["top"], 1)
    cv2.putText(vis, f"Bottom: {arms['bottom']:.1f}px", (cx + 10, cy + int(arms['bottom']/2)), 
                font, 0.5, colors["bottom"], 1)
    cv2.putText(vis, f"Left: {arms['left']:.1f}px", (cx - int(arms['left']/2) - 60, cy - 10), 
                font, 0.5, colors["left"], 1)
    cv2.putText(vis, f"Right: {arms['right']:.1f}px", (cx + int(arms['right']/2) - 20, cy - 10), 
                font, 0.5, colors["right"], 1)
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Measure cross arm lengths in an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Path to output visualization")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    try:
        bgr, gray = load_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1
    
    try:
        results = refined_cross_measurement(gray)
    except ValueError as e:
        print(f"Error detecting cross: {e}")
        return 1
    
    print("\n" + "="*60)
    print("CROSS ARM MEASUREMENTS")
    print("="*60)
    
    print(f"\nCenter position: ({results['center'][0]}, {results['center'][1]})")
    
    print("\n--- Half Arm Lengths (pixels) ---")
    for arm, length in results["arms"].items():
        print(f"  {arm.capitalize():8s}: {length:8.2f} px")
    
    print("\n--- Total Arm Lengths ---")
    vertical = results["arms"]["top"] + results["arms"]["bottom"]
    horizontal = results["arms"]["left"] + results["arms"]["right"]
    print(f"  Vertical (top + bottom):    {vertical:.2f} px")
    print(f"  Horizontal (left + right):  {horizontal:.2f} px")
    
    print("\n--- Arm Ratios ---")
    for ratio_name, ratio_value in results["ratios"].items():
        if ratio_value != float("inf"):
            print(f"  {ratio_name:25s}: {ratio_value:.4f}")
        else:
            print(f"  {ratio_name:25s}: undefined (division by zero)")
    
    print("="*60 + "\n")
    
    # Create visualization
    output_path = args.output or "cross_measurements.png"
    visualize_measurements(bgr, results, output_path)
    print(f"Visualization saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
