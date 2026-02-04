"""
Refined cross arm measurement with subpixel accuracy.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional


def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and return both color and grayscale versions."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def find_cross_edges_hough(gray: np.ndarray) -> Dict:
    """
    Find cross edges using Hough line detection and edge analysis.
    """
    # Pre-process
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Detect edges with Canny
    edges = cv2.Canny(blurred, 30, 100)
    
    # Use Hough Transform to find lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=30, 
        maxLineGap=10
    )
    
    if lines is None:
        return None
    
    # Classify lines as horizontal or vertical
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if angle < 15 or angle > 165:  # Horizontal
            y_avg = (y1 + y2) / 2
            horizontal.append({'y': y_avg, 'x1': min(x1, x2), 'x2': max(x1, x2), 'len': length})
        elif 75 < angle < 105:  # Vertical
            x_avg = (x1 + x2) / 2
            vertical.append({'x': x_avg, 'y1': min(y1, y2), 'y2': max(y1, y2), 'len': length})
    
    return {
        'horizontal': horizontal,
        'vertical': vertical,
        'edges': edges
    }


def find_cross_contour(gray: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Find the cross contour using thresholding and morphological operations.
    """
    # Apply bilateral filter to preserve edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try both normal and inverted
    contours_normal, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    binary_inv = cv2.bitwise_not(binary)
    contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    def score_contour(contour, img_shape):
        """Score a contour based on cross-like characteristics."""
        h, w = img_shape[:2]
        area = cv2.contourArea(contour)
        
        if area < 100 or area > h * w * 0.5:
            return -1
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1:
            return -1
        
        # Approximate polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Cross has 12 vertices ideally
        vertex_score = max(0, 1.0 - abs(num_vertices - 12) / 12.0)
        
        # Bounding box aspect ratio
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        
        # Solidity (cross is less solid than a square)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Cross solidity is typically 0.55-0.7
        solidity_score = max(0, 1.0 - abs(solidity - 0.6) / 0.4)
        
        return vertex_score * 0.3 + aspect * 0.3 + solidity_score * 0.4
    
    all_contours = [(c, 'normal') for c in contours_normal] + [(c, 'inv') for c in contours_inv]
    
    best_contour = None
    best_score = -1
    best_type = None
    
    for contour, ctype in all_contours:
        score = score_contour(contour, gray.shape)
        if score > best_score:
            best_score = score
            best_contour = contour
            best_type = ctype
    
    mask = binary_inv if best_type == 'inv' else binary
    return best_contour, mask


def measure_arm_subpixel(mask: np.ndarray, cx: float, cy: float, direction: str, 
                         gray: np.ndarray = None) -> Tuple[float, float]:
    """
    Measure arm length with subpixel accuracy using gradient-based edge detection.
    
    Returns: (length, edge_position)
    """
    h, w = mask.shape
    
    # Get the profile along the arm direction
    if direction == "top":
        # Extract vertical profile going up
        profile_length = int(cy)
        if profile_length < 1:
            return 0.0, cy
        x = int(round(cx))
        profile = mask[0:int(cy)+1, x][::-1].astype(np.float64)
    elif direction == "bottom":
        profile_length = h - int(cy) - 1
        if profile_length < 1:
            return 0.0, cy
        x = int(round(cx))
        profile = mask[int(cy):, x].astype(np.float64)
    elif direction == "left":
        profile_length = int(cx)
        if profile_length < 1:
            return 0.0, cx
        y = int(round(cy))
        profile = mask[y, 0:int(cx)+1][::-1].astype(np.float64)
    elif direction == "right":
        profile_length = w - int(cx) - 1
        if profile_length < 1:
            return 0.0, cx
        y = int(round(cy))
        profile = mask[y, int(cx):].astype(np.float64)
    else:
        return 0.0, 0.0
    
    # Normalize profile
    profile = profile / 255.0
    
    # Find the edge using gradient
    if len(profile) < 3:
        return 0.0, 0.0
    
    gradient = np.gradient(profile)
    
    # Find where the profile transitions from 1 (inside) to 0 (outside)
    # This is where gradient is most negative
    edge_idx = np.argmin(gradient)
    
    # Subpixel refinement using parabolic fit
    if 1 <= edge_idx < len(gradient) - 1:
        g_minus = gradient[edge_idx - 1]
        g_center = gradient[edge_idx]
        g_plus = gradient[edge_idx + 1]
        
        # Parabolic interpolation for subpixel
        denom = g_minus - 2*g_center + g_plus
        if abs(denom) > 1e-6:
            offset = 0.5 * (g_minus - g_plus) / denom
            edge_idx_subpixel = edge_idx + offset
        else:
            edge_idx_subpixel = float(edge_idx)
    else:
        edge_idx_subpixel = float(edge_idx)
    
    # Calculate actual edge position
    if direction == "top":
        edge_pos = cy - edge_idx_subpixel
    elif direction == "bottom":
        edge_pos = cy + edge_idx_subpixel
    elif direction == "left":
        edge_pos = cx - edge_idx_subpixel
    elif direction == "right":
        edge_pos = cx + edge_idx_subpixel
    
    return edge_idx_subpixel, edge_pos


def measure_cross_arms(gray: np.ndarray, debug: bool = False) -> Dict:
    """
    Measure all cross arms with subpixel accuracy.
    """
    h, w = gray.shape
    
    # Find the cross contour
    contour, mask = find_cross_contour(gray)
    
    if contour is None:
        raise ValueError("Could not find cross contour")
    
    # Create filled mask
    filled_mask = np.zeros_like(gray)
    cv2.drawContours(filled_mask, [contour], -1, 255, -1)
    
    # Find center using moments
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw / 2, y + bh / 2
    
    # Measure arms using multiple parallel scan lines for robustness
    def measure_arm_robust(mask, cx, cy, direction, num_lines=21):
        """Measure arm using multiple parallel scan lines."""
        measurements = []
        half_lines = num_lines // 2
        
        for offset in range(-half_lines, half_lines + 1):
            if direction in ["top", "bottom"]:
                x_test = cx + offset
                if 0 <= x_test < mask.shape[1]:
                    length, _ = measure_arm_subpixel(mask, x_test, cy, direction, gray)
                    if length > 0:
                        measurements.append(length)
            else:  # left, right
                y_test = cy + offset
                if 0 <= y_test < mask.shape[0]:
                    length, _ = measure_arm_subpixel(mask, cx, y_test, direction, gray)
                    if length > 0:
                        measurements.append(length)
        
        if measurements:
            # Use median for robustness
            return float(np.median(measurements))
        return 0.0
    
    # Simple measurement using binary mask transition
    def simple_arm_measure(mask, cx, cy, direction):
        """Simple measurement by finding first transition point."""
        h, w = mask.shape
        cx_int, cy_int = int(round(cx)), int(round(cy))
        
        if direction == "top":
            for i in range(cy_int, -1, -1):
                if mask[i, cx_int] == 0:
                    return cy - i
            return float(cy)
        elif direction == "bottom":
            for i in range(cy_int, h):
                if mask[i, cx_int] == 0:
                    return i - cy
            return float(h - cy)
        elif direction == "left":
            for i in range(cx_int, -1, -1):
                if mask[cy_int, i] == 0:
                    return cx - i
            return float(cx)
        elif direction == "right":
            for i in range(cx_int, w):
                if mask[cy_int, i] == 0:
                    return i - cx
            return float(w - cx)
        return 0.0
    
    # Measure with multiple scan lines
    def measure_multi_line(mask, cx, cy, direction, num_lines=21):
        """Measure using multiple parallel lines."""
        measurements = []
        half = num_lines // 2
        h, w = mask.shape
        
        for offset in range(-half, half + 1):
            if direction == "top":
                x = int(round(cx)) + offset
                if 0 <= x < w:
                    for i in range(int(round(cy)), -1, -1):
                        if mask[i, x] == 0:
                            measurements.append(cy - i)
                            break
                    else:
                        measurements.append(cy)
            elif direction == "bottom":
                x = int(round(cx)) + offset
                if 0 <= x < w:
                    for i in range(int(round(cy)), h):
                        if mask[i, x] == 0:
                            measurements.append(i - cy)
                            break
                    else:
                        measurements.append(h - cy)
            elif direction == "left":
                y = int(round(cy)) + offset
                if 0 <= y < h:
                    for i in range(int(round(cx)), -1, -1):
                        if mask[y, i] == 0:
                            measurements.append(cx - i)
                            break
                    else:
                        measurements.append(cx)
            elif direction == "right":
                y = int(round(cy)) + offset
                if 0 <= y < h:
                    for i in range(int(round(cx)), w):
                        if mask[y, i] == 0:
                            measurements.append(i - cx)
                            break
                    else:
                        measurements.append(w - cx)
        
        if measurements:
            return float(np.median(measurements))
        return 0.0
    
    arms = {
        "top": measure_multi_line(filled_mask, cx, cy, "top"),
        "bottom": measure_multi_line(filled_mask, cx, cy, "bottom"),
        "left": measure_multi_line(filled_mask, cx, cy, "left"),
        "right": measure_multi_line(filled_mask, cx, cy, "right"),
    }
    
    # Calculate ratios
    vertical_total = arms["top"] + arms["bottom"]
    horizontal_total = arms["left"] + arms["right"]
    
    ratios = {
        "top_to_bottom": arms["top"] / arms["bottom"] if arms["bottom"] > 0 else float("inf"),
        "bottom_to_top": arms["bottom"] / arms["top"] if arms["top"] > 0 else float("inf"),
        "left_to_right": arms["left"] / arms["right"] if arms["right"] > 0 else float("inf"),
        "right_to_left": arms["right"] / arms["left"] if arms["left"] > 0 else float("inf"),
        "vertical_to_horizontal": vertical_total / horizontal_total if horizontal_total > 0 else float("inf"),
        "horizontal_to_vertical": horizontal_total / vertical_total if vertical_total > 0 else float("inf"),
    }
    
    return {
        "center": (cx, cy),
        "arms": arms,
        "ratios": ratios,
        "contour": contour,
        "mask": filled_mask,
        "vertical_total": vertical_total,
        "horizontal_total": horizontal_total,
    }


def visualize_results(bgr: np.ndarray, results: Dict, output_path: str = None) -> np.ndarray:
    """Create detailed visualization of measurements."""
    vis = bgr.copy()
    
    cx, cy = results["center"]
    arms = results["arms"]
    
    # Draw the detected contour
    if results.get("contour") is not None:
        cv2.drawContours(vis, [results["contour"]], -1, (0, 255, 0), 1)
    
    # Draw center point
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
    cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 0), 1)
    
    # Draw arm measurements with colors
    colors = {
        "top": (255, 100, 100),      # Light blue
        "bottom": (100, 255, 255),   # Yellow
        "left": (100, 100, 255),     # Light red
        "right": (255, 100, 255),    # Magenta
    }
    
    # Draw measurement lines
    cv2.line(vis, (int(cx), int(cy)), (int(cx), int(cy - arms["top"])), colors["top"], 2)
    cv2.line(vis, (int(cx), int(cy)), (int(cx), int(cy + arms["bottom"])), colors["bottom"], 2)
    cv2.line(vis, (int(cx), int(cy)), (int(cx - arms["left"]), int(cy)), colors["left"], 2)
    cv2.line(vis, (int(cx), int(cy)), (int(cx + arms["right"]), int(cy)), colors["right"], 2)
    
    # Draw end markers
    cv2.circle(vis, (int(cx), int(cy - arms["top"])), 3, colors["top"], -1)
    cv2.circle(vis, (int(cx), int(cy + arms["bottom"])), 3, colors["bottom"], -1)
    cv2.circle(vis, (int(cx - arms["left"]), int(cy)), 3, colors["left"], -1)
    cv2.circle(vis, (int(cx + arms["right"]), int(cy)), 3, colors["right"], -1)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    # Position labels with background for readability
    def put_text_with_bg(img, text, pos, color):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (pos[0]-2, pos[1]-th-2), (pos[0]+tw+2, pos[1]+2), (255, 255, 255), -1)
        cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    
    put_text_with_bg(vis, f"Top: {arms['top']:.2f}px", (int(cx)+10, int(cy-arms['top']/2)), colors["top"])
    put_text_with_bg(vis, f"Bottom: {arms['bottom']:.2f}px", (int(cx)+10, int(cy+arms['bottom']/2)), colors["bottom"])
    put_text_with_bg(vis, f"Left: {arms['left']:.2f}px", (int(cx-arms['left']/2)-50, int(cy)-15), colors["left"])
    put_text_with_bg(vis, f"Right: {arms['right']:.2f}px", (int(cx+arms['right']/2)-20, int(cy)-15), colors["right"])
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure cross arm lengths with subpixel accuracy")
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
        results = measure_cross_arms(gray, debug=args.debug)
    except ValueError as e:
        print(f"Error detecting cross: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("CROSS ARM MEASUREMENTS (Refined)")
    print("=" * 70)
    
    print(f"\nImage dimensions: {gray.shape[1]} x {gray.shape[0]} pixels")
    print(f"Cross center: ({results['center'][0]:.2f}, {results['center'][1]:.2f})")
    
    print("\n" + "-" * 70)
    print("HALF ARM LENGTHS (from center to edge)")
    print("-" * 70)
    print(f"  Top arm:      {results['arms']['top']:10.2f} pixels")
    print(f"  Bottom arm:   {results['arms']['bottom']:10.2f} pixels")
    print(f"  Left arm:     {results['arms']['left']:10.2f} pixels")
    print(f"  Right arm:    {results['arms']['right']:10.2f} pixels")
    
    print("\n" + "-" * 70)
    print("FULL ARM LENGTHS")
    print("-" * 70)
    print(f"  Vertical (top + bottom):    {results['vertical_total']:10.2f} pixels")
    print(f"  Horizontal (left + right):  {results['horizontal_total']:10.2f} pixels")
    
    print("\n" + "-" * 70)
    print("ARM RATIOS")
    print("-" * 70)
    print(f"  Top / Bottom:               {results['ratios']['top_to_bottom']:10.4f}")
    print(f"  Bottom / Top:               {results['ratios']['bottom_to_top']:10.4f}")
    print(f"  Left / Right:               {results['ratios']['left_to_right']:10.4f}")
    print(f"  Right / Left:               {results['ratios']['right_to_left']:10.4f}")
    print(f"  Vertical / Horizontal:      {results['ratios']['vertical_to_horizontal']:10.4f}")
    print(f"  Horizontal / Vertical:      {results['ratios']['horizontal_to_vertical']:10.4f}")
    
    print("=" * 70 + "\n")
    
    # Create visualization
    output_path = args.output or "cross_refined_measurement.png"
    visualize_results(bgr, results, output_path)
    print(f"Visualization saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
