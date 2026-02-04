#!/usr/bin/env python3
"""
Comprehensive Cross Arm Measurement Analysis

This script measures the half-arm lengths of a cross marker and calculates arm ratios.
Designed for analyzing fiducial cross markers in microscopy/calibration images.

Usage:
    python cross_measurement_analysis.py <image_path>
    python cross_measurement_analysis.py  # Will try to find cross images in workspace
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import json


class CrossMeasurement:
    """Class for measuring cross marker arms."""
    
    def __init__(self, image_path=None, gray_image=None):
        """Initialize with either image path or grayscale numpy array."""
        if image_path:
            self.image_path = Path(image_path)
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            if len(img.shape) == 3:
                self.original = img
                self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                self.original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.gray = img
        elif gray_image is not None:
            self.gray = gray_image
            self.original = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            self.image_path = None
        else:
            raise ValueError("Must provide either image_path or gray_image")
        
        self.height, self.width = self.gray.shape
        self.results = None
        self.contour = None
        self.edges = None
    
    def detect_cross(self, method='combined'):
        """
        Detect the cross shape in the image.
        
        Args:
            method: 'canny', 'gradient', or 'combined'
        
        Returns:
            contour of detected cross, or None if not found
        """
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 1.5)
        
        # Method 1: Canny edge detection
        v = np.median(self.gray)
        edges_canny = cv2.Canny(blurred, int(0.67 * v), int(1.33 * v))
        
        # Method 2: Lower threshold Canny
        edges_low = cv2.Canny(blurred, 10, 50)
        
        # Method 3: Gradient magnitude thresholding
        gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_norm = (grad_mag / (grad_mag.max() + 1e-6) * 255).astype(np.uint8)
        _, edges_grad = cv2.threshold(grad_norm, 15, 255, cv2.THRESH_BINARY)
        
        # Combine
        if method == 'canny':
            self.edges = edges_canny
        elif method == 'gradient':
            self.edges = edges_grad
        else:  # combined
            self.edges = cv2.bitwise_or(edges_canny, edges_low)
            self.edges = cv2.bitwise_or(self.edges, edges_grad)
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        self.edges = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(self.edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Find best cross candidate
        center_x, center_y = self.width / 2, self.height / 2
        best_score = -1
        best_contour = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Too small
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w/2, y + h/2
            
            # Distance from center
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            # Aspect ratio (cross should be roughly square)
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            # Score: prefer larger, more centered, more square
            score = area * aspect / (1 + dist * 0.01)
            
            if score > best_score:
                best_score = score
                best_contour = cnt
        
        self.contour = best_contour
        return best_contour
    
    def measure_arms(self, center=None):
        """
        Measure the half-arm lengths from center to each extreme point.
        
        Args:
            center: Optional (cx, cy) tuple. If None, calculated from contour.
        
        Returns:
            Dictionary with measurement results
        """
        if self.contour is None:
            raise ValueError("No cross detected. Run detect_cross() first.")
        
        pts = self.contour.reshape(-1, 2)
        
        # Calculate center
        if center is None:
            M = cv2.moments(self.contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        else:
            cx, cy = center
        
        # Find extreme points (arm tips)
        top_idx = np.argmin(pts[:, 1])
        bottom_idx = np.argmax(pts[:, 1])
        left_idx = np.argmin(pts[:, 0])
        right_idx = np.argmax(pts[:, 0])
        
        extremes = {
            'top': pts[top_idx],
            'bottom': pts[bottom_idx],
            'left': pts[left_idx],
            'right': pts[right_idx]
        }
        
        # Calculate half-arm lengths
        half_arms = {
            'top': float(cy - extremes['top'][1]),
            'bottom': float(extremes['bottom'][1] - cy),
            'left': float(cx - extremes['left'][0]),
            'right': float(extremes['right'][0] - cx)
        }
        
        # Full arm lengths
        full_arms = {
            'horizontal': half_arms['left'] + half_arms['right'],
            'vertical': half_arms['top'] + half_arms['bottom']
        }
        
        # Calculate ratios
        ratios = {}
        
        # Half-arm ratios
        if half_arms['right'] > 0:
            ratios['left/right'] = half_arms['left'] / half_arms['right']
        if half_arms['left'] > 0:
            ratios['right/left'] = half_arms['right'] / half_arms['left']
        if half_arms['bottom'] > 0:
            ratios['top/bottom'] = half_arms['top'] / half_arms['bottom']
        if half_arms['top'] > 0:
            ratios['bottom/top'] = half_arms['bottom'] / half_arms['top']
        
        # Full arm ratios
        if full_arms['vertical'] > 0:
            ratios['horizontal/vertical'] = full_arms['horizontal'] / full_arms['vertical']
        if full_arms['horizontal'] > 0:
            ratios['vertical/horizontal'] = full_arms['vertical'] / full_arms['horizontal']
        
        self.results = {
            'image_size': (self.width, self.height),
            'center': (float(cx), float(cy)),
            'half_arms': half_arms,
            'full_arms': full_arms,
            'ratios': ratios,
            'extreme_points': {k: (int(v[0]), int(v[1])) for k, v in extremes.items()}
        }
        
        return self.results
    
    def print_results(self):
        """Print formatted measurement results."""
        if self.results is None:
            print("No results available. Run measure_arms() first.")
            return
        
        r = self.results
        
        print()
        print("=" * 72)
        print("CROSS ARM MEASUREMENT RESULTS")
        print("=" * 72)
        print()
        
        print(f"Image dimensions: {r['image_size'][0]} x {r['image_size'][1]} pixels")
        print(f"Cross center:     ({r['center'][0]:.2f}, {r['center'][1]:.2f})")
        print()
        
        print("-" * 72)
        print("HALF-ARM LENGTHS (from center to arm tip)")
        print("-" * 72)
        print()
        print(f"  {'Direction':<12} {'Length (px)':>12}")
        print(f"  {'-'*12} {'-'*12}")
        for direction in ['top', 'bottom', 'left', 'right']:
            length = r['half_arms'][direction]
            print(f"  {direction.capitalize():<12} {length:>12.2f}")
        
        print()
        print("-" * 72)
        print("FULL ARM LENGTHS")
        print("-" * 72)
        print()
        print(f"  Horizontal (left + right): {r['full_arms']['horizontal']:.2f} pixels")
        print(f"  Vertical   (top + bottom): {r['full_arms']['vertical']:.2f} pixels")
        
        print()
        print("-" * 72)
        print("ARM RATIOS")
        print("-" * 72)
        print()
        print(f"  {'Ratio':<25} {'Value':>10}")
        print(f"  {'-'*25} {'-'*10}")
        for name, value in r['ratios'].items():
            print(f"  {name:<25} {value:>10.4f}")
        
        print()
        print("-" * 72)
        print("EXTREME POINTS (arm tip coordinates)")
        print("-" * 72)
        print()
        for direction, pt in r['extreme_points'].items():
            print(f"  {direction.capitalize():<12}: ({pt[0]:>4d}, {pt[1]:>4d})")
        
        print()
        print("=" * 72)
    
    def save_visualization(self, output_path):
        """Save visualization of measurements."""
        if self.results is None or self.contour is None:
            print("No results to visualize")
            return
        
        vis = self.original.copy()
        
        # Draw contour
        cv2.drawContours(vis, [self.contour], -1, (0, 255, 0), 2)
        
        r = self.results
        cx, cy = r['center']
        
        # Draw center point
        cv2.circle(vis, (int(cx), int(cy)), 10, (0, 0, 255), -1)
        cv2.circle(vis, (int(cx), int(cy)), 12, (255, 255, 255), 2)
        
        # Draw arm lines with colors
        colors = {
            'top': (255, 100, 100),      # Light blue
            'bottom': (255, 100, 255),   # Pink
            'left': (100, 255, 255),     # Yellow
            'right': (100, 255, 100)     # Light green
        }
        
        for direction, pt in r['extreme_points'].items():
            color = colors[direction]
            cv2.line(vis, (int(cx), int(cy)), pt, color, 3)
            cv2.circle(vis, pt, 7, color, -1)
            cv2.circle(vis, pt, 9, (255, 255, 255), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Position labels near the tips
        ha = r['half_arms']
        ext = r['extreme_points']
        
        labels = [
            (f"Top: {ha['top']:.1f}px", (ext['top'][0] + 15, ext['top'][1] + 25)),
            (f"Bottom: {ha['bottom']:.1f}px", (ext['bottom'][0] + 15, ext['bottom'][1] - 10)),
            (f"Left: {ha['left']:.1f}px", (ext['left'][0] + 10, ext['left'][1] - 10)),
            (f"Right: {ha['right']:.1f}px", (ext['right'][0] - 130, ext['right'][1] - 10)),
        ]
        
        for text, pos in labels:
            # Draw text with background
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(vis, (pos[0]-2, pos[1]-th-2), (pos[0]+tw+2, pos[1]+2), (0, 0, 0), -1)
            cv2.putText(vis, text, pos, font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(str(output_path), vis)
        print(f"Visualization saved to: {output_path}")
    
    def save_results_json(self, output_path):
        """Save results to JSON file."""
        if self.results is None:
            return
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")


def analyze_cross_image(image_path, output_dir=None):
    """
    Analyze a cross image and return measurements.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory for output files (default: same as input)
    
    Returns:
        Dictionary with measurement results
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
    
    stem = image_path.stem
    
    print(f"\nAnalyzing: {image_path}")
    print("-" * 50)
    
    try:
        cm = CrossMeasurement(str(image_path))
        
        # Detect and measure
        contour = cm.detect_cross()
        
        if contour is None:
            print("ERROR: Could not detect cross in image")
            # Save debug edges
            if cm.edges is not None:
                cv2.imwrite(str(output_dir / f"debug_edges_{stem}.png"), cm.edges)
            return None
        
        results = cm.measure_arms()
        
        # Output results
        cm.print_results()
        cm.save_visualization(output_dir / f"cross_measured_{stem}.png")
        cm.save_results_json(output_dir / f"cross_results_{stem}.json")
        
        # Save edges
        if cm.edges is not None:
            cv2.imwrite(str(output_dir / f"edges_{stem}.png"), cm.edges)
        
        return results
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Find images in workspace
        workspace = Path('/workspace')
        image_paths = []
        
        for pattern in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            for f in workspace.glob(pattern):
                # Skip output files
                if any(skip in f.name for skip in ['measured', 'edge', 'debug', 'result']):
                    continue
                image_paths.append(str(f))
        
        if not image_paths:
            print("Usage: python cross_measurement_analysis.py <image_path> [more_images...]")
            print("\nNo suitable images found in workspace.")
            return 1
    
    print("=" * 72)
    print("CROSS ARM MEASUREMENT ANALYSIS")
    print("=" * 72)
    
    all_results = {}
    
    for img_path in image_paths:
        results = analyze_cross_image(img_path, output_dir=Path('/workspace'))
        if results:
            all_results[img_path] = results
    
    # Summary if multiple images
    if len(all_results) > 1:
        print("\n" + "=" * 72)
        print("SUMMARY OF ALL MEASUREMENTS")
        print("=" * 72)
        for path, r in all_results.items():
            print(f"\n{Path(path).name}:")
            print(f"  Half-arms: T={r['half_arms']['top']:.1f}, B={r['half_arms']['bottom']:.1f}, "
                  f"L={r['half_arms']['left']:.1f}, R={r['half_arms']['right']:.1f}")
            print(f"  Ratios: L/R={r['ratios'].get('left/right', 0):.4f}, "
                  f"T/B={r['ratios'].get('top/bottom', 0):.4f}")
    
    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
