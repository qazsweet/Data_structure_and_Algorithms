#!/usr/bin/env python3
"""
Subpixel Saddle Point Detection

Detects saddle points (e.g., checkerboard corners) using the Hessian determinant
and refines their locations to subpixel accuracy using quadratic surface fitting.
"""

import argparse
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from pathlib import Path


def compute_hessian(image: np.ndarray, sigma: float = 0.15) -> tuple:
    """
    Compute the Hessian matrix components of an image.
    
    Args:
        image: Grayscale image normalized to [0, 1]
        sigma: Gaussian smoothing sigma
        
    Returns:
        Tuple of (dxx, dyy, dxy, smoothed_image)
    """
    # Smooth to handle real-world noise
    smoothed = gaussian_filter(image, sigma=sigma)
    
    # Compute first derivatives using central differences
    dy, dx = np.gradient(smoothed)
    
    # Compute second derivatives (Hessian components)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    
    return dxx, dyy, dxy, smoothed


def compute_det_hessian(dxx: np.ndarray, dyy: np.ndarray, dxy: np.ndarray) -> np.ndarray:
    """
    Compute the determinant of the Hessian matrix.
    
    A negative determinant indicates a saddle point (eigenvalues have opposite signs).
    
    Args:
        dxx, dyy, dxy: Hessian matrix components
        
    Returns:
        Determinant of Hessian array
    """
    return (dxx * dyy) - (dxy ** 2)


def find_saddle_candidates(det_hessian: np.ndarray, 
                           min_distance: int = 5,
                           threshold_ratio: float = 0.1,
                           border_margin: int = 5) -> np.ndarray:
    """
    Find saddle point candidates from the Hessian determinant.
    
    Saddle points have negative determinant, so we look for local minima
    of the determinant (or equivalently, local maxima of -det_hessian).
    
    Args:
        det_hessian: Determinant of Hessian array
        min_distance: Minimum distance between detected points
        threshold_ratio: Threshold as ratio of the maximum saddle strength
        border_margin: Margin from image borders to exclude
        
    Returns:
        Array of (row, col) coordinates of saddle candidates
    """
    # Saddle strength: negative determinant values (inverted for peak finding)
    saddle_strength = -det_hessian.copy()
    saddle_strength[saddle_strength < 0] = 0  # Only keep negative determinant areas
    
    # Threshold based on maximum value
    max_val = np.max(saddle_strength)
    if max_val <= 0:
        return np.array([])
    
    threshold = threshold_ratio * max_val
    
    # Find local maxima using maximum filter
    size = 2 * min_distance + 1
    local_max = maximum_filter(saddle_strength, size=size)
    
    # Points that are local maxima and above threshold
    is_peak = (saddle_strength == local_max) & (saddle_strength > threshold)
    
    # Exclude border regions
    is_peak[:border_margin, :] = False
    is_peak[-border_margin:, :] = False
    is_peak[:, :border_margin] = False
    is_peak[:, -border_margin:] = False
    
    # Get coordinates
    coords = np.argwhere(is_peak)
    
    return coords


def subpixel_refine_quadratic(det_hessian: np.ndarray, 
                              candidates: np.ndarray,
                              window_size: int = 3) -> np.ndarray:
    """
    Refine saddle point locations to subpixel accuracy using quadratic fitting.
    
    Fits a 2D quadratic surface around each candidate point and finds
    the extremum of that surface.
    
    The quadratic model is:
        f(x,y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    
    The extremum is at:
        x = (b*d - c*e) / (c^2 - 2*a*b)
        y = (a*e - c*d) / (c^2 - 2*a*b)
    
    Args:
        det_hessian: Determinant of Hessian array
        candidates: Array of (row, col) coordinates
        window_size: Size of the fitting window (must be odd, >= 3)
        
    Returns:
        Array of refined (row, col) subpixel coordinates
    """
    if len(candidates) == 0:
        return np.array([])
    
    half_win = window_size // 2
    h, w = det_hessian.shape
    
    refined = []
    
    for row, col in candidates:
        # Ensure window fits in image
        if (row < half_win or row >= h - half_win or
            col < half_win or col >= w - half_win):
            refined.append([float(row), float(col)])
            continue
        
        # Extract local window
        window = det_hessian[row - half_win:row + half_win + 1,
                            col - half_win:col + half_win + 1]
        
        # Build coordinate grids centered at (0, 0)
        y_coords, x_coords = np.mgrid[-half_win:half_win + 1, 
                                       -half_win:half_win + 1]
        
        # Flatten for least squares
        x = x_coords.flatten()
        y = y_coords.flatten()
        z = window.flatten()
        
        # Design matrix for quadratic fit: [x^2, y^2, xy, x, y, 1]
        A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
        
        # Solve least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c, d, e, f = coeffs
            
            # Find extremum of the quadratic
            # Gradient: [2ax + cy + d, 2by + cx + e] = [0, 0]
            # Matrix form: [[2a, c], [c, 2b]] @ [x, y] = [-d, -e]
            denom = 4 * a * b - c ** 2
            
            if abs(denom) > 1e-10:
                dx = (c * e - 2 * b * d) / denom
                dy = (c * d - 2 * a * e) / denom
                
                # Limit refinement to within the window
                if abs(dx) <= half_win and abs(dy) <= half_win:
                    refined.append([row + dy, col + dx])
                else:
                    refined.append([float(row), float(col)])
            else:
                refined.append([float(row), float(col)])
                
        except np.linalg.LinAlgError:
            refined.append([float(row), float(col)])
    
    return np.array(refined)


def subpixel_refine_taylor(dxx: np.ndarray, dyy: np.ndarray, dxy: np.ndarray,
                           det_hessian: np.ndarray,
                           candidates: np.ndarray) -> np.ndarray:
    """
    Refine saddle point locations using Taylor expansion.
    
    Uses the second-order Taylor expansion of the determinant of Hessian
    to find the subpixel location of the extremum.
    
    Args:
        dxx, dyy, dxy: Hessian components
        det_hessian: Determinant of Hessian array
        candidates: Array of (row, col) coordinates
        
    Returns:
        Array of refined (row, col) subpixel coordinates
    """
    if len(candidates) == 0:
        return np.array([])
    
    h, w = det_hessian.shape
    refined = []
    
    # Compute gradients of det_hessian for Taylor expansion
    det_dy, det_dx = np.gradient(det_hessian)
    det_dyy, _ = np.gradient(det_dy)
    _, det_dxx = np.gradient(det_dx)
    det_dxy, _ = np.gradient(det_dx)
    
    for row, col in candidates:
        if row < 1 or row >= h - 1 or col < 1 or col >= w - 1:
            refined.append([float(row), float(col)])
            continue
        
        # Get local derivatives at the candidate point
        gx = det_dx[row, col]
        gy = det_dy[row, col]
        hxx = det_dxx[row, col]
        hyy = det_dyy[row, col]
        hxy = det_dxy[row, col]
        
        # Solve for offset: H @ delta = -g
        # [[hxx, hxy], [hxy, hyy]] @ [dx, dy] = [-gx, -gy]
        denom = hxx * hyy - hxy ** 2
        
        if abs(denom) > 1e-10:
            dx = (hxy * gy - hyy * gx) / denom
            dy = (hxy * gx - hxx * gy) / denom
            
            # Limit refinement to reasonable range
            if abs(dx) <= 1.0 and abs(dy) <= 1.0:
                refined.append([row + dy, col + dx])
            else:
                refined.append([float(row), float(col)])
        else:
            refined.append([float(row), float(col)])
    
    return np.array(refined)


def detect_saddle_points_subpixel(image: np.ndarray,
                                   sigma: float = 0.15,
                                   min_distance: int = 5,
                                   threshold_ratio: float = 0.1,
                                   refinement_method: str = 'quadratic',
                                   border_margin: int = 5) -> tuple:
    """
    Detect saddle points with subpixel accuracy.
    
    Args:
        image: Input image (grayscale or BGR)
        sigma: Gaussian smoothing sigma
        min_distance: Minimum distance between detected points
        threshold_ratio: Threshold as ratio of max saddle strength
        refinement_method: 'quadratic' or 'taylor'
        border_margin: Margin from image borders to exclude
        
    Returns:
        Tuple of (subpixel_coords, integer_coords, det_hessian)
        - subpixel_coords: Nx2 array of (row, col) with subpixel precision
        - integer_coords: Nx2 array of initial integer (row, col) candidates
        - det_hessian: The determinant of Hessian array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to float [0, 1]
    gray = gray.astype(float) / 255.0
    
    # Compute Hessian components
    dxx, dyy, dxy, smoothed = compute_hessian(gray, sigma=sigma)
    
    # Compute determinant of Hessian
    det_hessian = compute_det_hessian(dxx, dyy, dxy)
    
    # Find saddle candidates (integer precision)
    candidates = find_saddle_candidates(det_hessian, 
                                         min_distance=min_distance,
                                         threshold_ratio=threshold_ratio,
                                         border_margin=border_margin)
    
    # Refine to subpixel precision
    if refinement_method == 'quadratic':
        refined = subpixel_refine_quadratic(det_hessian, candidates)
    elif refinement_method == 'taylor':
        refined = subpixel_refine_taylor(dxx, dyy, dxy, det_hessian, candidates)
    else:
        raise ValueError(f"Unknown refinement method: {refinement_method}")
    
    return refined, candidates, det_hessian


def draw_saddle_points(image: np.ndarray, 
                       points: np.ndarray,
                       color: tuple = (0, 0, 255),
                       radius: int = 3,
                       thickness: int = 1) -> np.ndarray:
    """
    Draw saddle points on an image.
    
    Args:
        image: Input image (will be converted to BGR if grayscale)
        points: Nx2 array of (row, col) coordinates
        color: BGR color tuple
        radius: Circle radius
        thickness: Circle thickness (-1 for filled)
        
    Returns:
        Image with drawn points
    """
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    
    for point in points:
        row, col = point
        # Draw at subpixel location (cv2.circle uses integer coords, but we can shift)
        center = (int(round(col)), int(round(row)))
        cv2.circle(output, center, radius, color, thickness)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Detect saddle points with subpixel precision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='Gaussian smoothing sigma')
    parser.add_argument('--min-distance', type=int, default=5,
                        help='Minimum distance between detected points')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold ratio (0-1) of max saddle strength')
    parser.add_argument('--method', type=str, default='quadratic',
                        choices=['quadratic', 'taylor'],
                        help='Subpixel refinement method')
    parser.add_argument('--border-margin', type=int, default=5,
                        help='Border margin to exclude')
    parser.add_argument('--out', type=str, default=None,
                        help='Output image path (default: saddle_<input>.png)')
    parser.add_argument('--save-coords', type=str, default=None,
                        help='Save coordinates to CSV file')
    parser.add_argument('--show-hessian', action='store_true',
                        help='Save Hessian determinant visualization')
    
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1
    
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return 1
    
    print(f"Loaded image: {image_path} ({image.shape})")
    
    # Detect saddle points
    subpixel_points, integer_points, det_hessian = detect_saddle_points_subpixel(
        image,
        sigma=args.sigma,
        min_distance=args.min_distance,
        threshold_ratio=args.threshold,
        refinement_method=args.method,
        border_margin=args.border_margin
    )
    
    print(f"Found {len(subpixel_points)} saddle points")
    
    # Draw results
    result = draw_saddle_points(image, subpixel_points, color=(0, 0, 255), radius=3)
    
    # Save output image
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = image_path.parent / f"saddle_{image_path.stem}.png"
    
    cv2.imwrite(str(out_path), result)
    print(f"Saved result to: {out_path}")
    
    # Save coordinates if requested
    if args.save_coords:
        coords_path = Path(args.save_coords)
        np.savetxt(coords_path, subpixel_points, delimiter=',', 
                   header='row,col', comments='', fmt='%.6f')
        print(f"Saved coordinates to: {coords_path}")
    
    # Save Hessian visualization if requested
    if args.show_hessian:
        hessian_path = image_path.parent / f"hessian_{image_path.stem}.png"
        # Normalize for visualization
        det_norm = det_hessian - det_hessian.min()
        det_norm = (det_norm / det_norm.max() * 255).astype(np.uint8)
        cv2.imwrite(str(hessian_path), det_norm)
        print(f"Saved Hessian visualization to: {hessian_path}")
    
    # Print first few points
    if len(subpixel_points) > 0:
        print("\nFirst 10 saddle points (row, col):")
        for i, (row, col) in enumerate(subpixel_points[:10]):
            print(f"  {i+1}: ({row:.4f}, {col:.4f})")
    
    return 0


if __name__ == '__main__':
    exit(main())
