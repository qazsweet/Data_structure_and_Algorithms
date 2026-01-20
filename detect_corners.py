"""
Chessboard Corner Detection using Saddle Points with Subpixel Precision

This module implements a saddle point-based approach to detect chessboard corners.
At each corner of a chessboard (where 4 squares meet), the intensity function forms
a saddle point. We detect these saddle points and refine their locations to subpixel
accuracy using quadratic surface fitting.

Theory:
-------
A saddle point occurs where:
1. The gradient is zero (or near zero)
2. The Hessian matrix has eigenvalues of opposite signs (determinant < 0)

For a 2D intensity function I(x, y), the Hessian matrix is:
    H = | Ixx  Ixy |
        | Ixy  Iyy |

At a saddle point: det(H) = Ixx * Iyy - Ixy^2 < 0

Subpixel refinement:
-------------------
We fit a quadratic surface around each detected saddle point:
    I(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f

The saddle point location is found by solving:
    dI/dx = 2*a*x + c*y + d = 0
    dI/dy = 2*b*y + c*x + e = 0
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import sys
import os


def compute_image_derivatives(gray, sigma=1.0):
    """
    Compute first and second order image derivatives using Gaussian smoothing.
    
    Args:
        gray: Grayscale image (float64)
        sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Dictionary containing Ix, Iy, Ixx, Iyy, Ixy derivatives
    """
    # Smooth the image first to reduce noise
    smoothed = gaussian_filter(gray, sigma=sigma)
    
    # First derivatives using Sobel operators
    Ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    # Second derivatives
    Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)
    
    return {
        'Ix': Ix,
        'Iy': Iy,
        'Ixx': Ixx,
        'Iyy': Iyy,
        'Ixy': Ixy,
        'smoothed': smoothed
    }


def compute_chessboard_saddle_response(gray, sigma=1.5):
    """
    Compute a response map specific to chessboard corners (saddle points).
    
    At a chessboard corner, the intensity forms a saddle pattern:
    - In one diagonal direction, intensity increases
    - In the perpendicular diagonal direction, intensity decreases
    
    We detect this using:
    1. Hessian determinant < 0 (opposite curvatures)
    2. Large magnitude of the mixed derivative Ixy
    3. Small magnitude of |Ixx - Iyy| (balanced curvature magnitudes)
    
    Args:
        gray: Grayscale image (float)
        sigma: Smoothing parameter
        
    Returns:
        saddle_response: Response map (higher = more likely saddle point)
        det_hessian: Determinant of Hessian
        derivs: Dictionary of derivatives
    """
    gray_float = gray.astype(np.float64)
    derivs = compute_image_derivatives(gray_float, sigma)
    
    Ixx = derivs['Ixx']
    Iyy = derivs['Iyy']
    Ixy = derivs['Ixy']
    
    # Determinant of Hessian
    det_hessian = Ixx * Iyy - Ixy ** 2
    
    # For chessboard saddles:
    # 1. det(H) should be strongly negative
    # 2. |Ixy| should be large (cross term dominates)
    # 3. The principal curvatures should be roughly equal in magnitude
    
    # Saddle response: emphasizes strong negative determinant with large |Ixy|
    # We use -det(H) * |Ixy| / (|Ixx| + |Iyy| + eps) to normalize
    eps = 1e-6
    saddle_response = np.zeros_like(det_hessian)
    
    # Only consider points with negative determinant (actual saddle points)
    neg_mask = det_hessian < 0
    
    # Compute response only where determinant is negative
    saddle_response[neg_mask] = (
        -det_hessian[neg_mask] * np.abs(Ixy[neg_mask]) / 
        (np.abs(Ixx[neg_mask]) + np.abs(Iyy[neg_mask]) + eps)
    )
    
    return saddle_response, det_hessian, derivs


def detect_saddle_points(gray, sigma=1.0, threshold_ratio=0.1, min_distance=10,
                         corner_quality_threshold=0.01):
    """
    Detect saddle points in an image using the Hessian matrix.
    
    A saddle point has det(H) < 0, meaning eigenvalues have opposite signs.
    We use the chessboard-specific saddle response for better detection.
    
    Args:
        gray: Grayscale image
        sigma: Gaussian smoothing sigma for derivative computation
        threshold_ratio: Threshold for saddle point detection (relative to max response)
        min_distance: Minimum distance between detected saddle points
        corner_quality_threshold: Not used in current implementation
        
    Returns:
        List of (x, y) coordinates of detected saddle points
    """
    # Compute chessboard-specific saddle response
    saddle_response, det_hessian, derivs = compute_chessboard_saddle_response(gray, sigma)
    
    # Threshold on saddle response
    max_response = np.max(saddle_response)
    threshold = max_response * threshold_ratio
    response_mask = saddle_response > threshold
    
    # Apply non-maximum suppression
    kernel_size = 2 * min_distance + 1
    local_max = ndimage.maximum_filter(saddle_response, size=kernel_size)
    local_max_mask = (saddle_response == local_max) & response_mask
    
    # Exclude border regions
    border = max(10, min_distance)
    local_max_mask[:border, :] = False
    local_max_mask[-border:, :] = False
    local_max_mask[:, :border] = False
    local_max_mask[:, -border:] = False
    
    # Extract saddle point coordinates
    saddle_points = np.argwhere(local_max_mask)
    
    # Sort by response strength (strongest first)
    responses = [saddle_response[pt[0], pt[1]] for pt in saddle_points]
    sorted_indices = np.argsort(responses)[::-1]
    saddle_points = saddle_points[sorted_indices]
    
    # Convert from (row, col) to (x, y) format
    saddle_points = [(pt[1], pt[0]) for pt in saddle_points]
    
    return saddle_points, det_hessian, derivs


def refine_saddle_point_subpixel(gray, x, y, window_size=5):
    """
    Refine saddle point location to subpixel accuracy using quadratic surface fitting.
    
    We fit a quadratic surface:
        I(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    
    And solve for the saddle point where gradients are zero.
    
    Args:
        gray: Grayscale image (float)
        x, y: Initial integer saddle point coordinates
        window_size: Size of the window for fitting (must be odd)
        
    Returns:
        (x_refined, y_refined): Subpixel saddle point coordinates
        success: Boolean indicating if refinement was successful
    """
    x, y = int(round(x)), int(round(y))
    half_win = window_size // 2
    
    # Check bounds
    h, w = gray.shape
    if (x - half_win < 0 or x + half_win >= w or 
        y - half_win < 0 or y + half_win >= h):
        return (float(x), float(y)), False
    
    # Extract window
    window = gray[y - half_win:y + half_win + 1, 
                  x - half_win:x + half_win + 1].astype(np.float64)
    
    # Create coordinate grids (relative to center)
    coords = np.arange(-half_win, half_win + 1)
    X, Y = np.meshgrid(coords, coords)
    
    # Flatten for least squares fitting
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = window.flatten()
    
    # Build design matrix for quadratic fit:
    # I = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    A = np.column_stack([
        x_flat**2,      # a
        y_flat**2,      # b
        x_flat * y_flat, # c
        x_flat,         # d
        y_flat,         # e
        np.ones_like(x_flat)  # f
    ])
    
    # Solve least squares: A @ coeffs = z
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z_flat, rcond=None)
    except np.linalg.LinAlgError:
        return (float(x), float(y)), False
    
    a, b, c, d, e, f = coeffs
    
    # Solve for saddle point (where gradient = 0):
    # dI/dx = 2*a*x + c*y + d = 0
    # dI/dy = c*x + 2*b*y + e = 0
    #
    # Matrix form: [[2a, c], [c, 2b]] @ [x, y] = [-d, -e]
    
    H_grad = np.array([[2*a, c], [c, 2*b]])
    grad_rhs = np.array([-d, -e])
    
    try:
        det = np.linalg.det(H_grad)
        if abs(det) < 1e-10:
            return (float(x), float(y)), False
            
        offset = np.linalg.solve(H_grad, grad_rhs)
    except np.linalg.LinAlgError:
        return (float(x), float(y)), False
    
    # Check if the offset is reasonable (within window)
    if abs(offset[0]) > half_win or abs(offset[1]) > half_win:
        return (float(x), float(y)), False
    
    # Verify it's actually a saddle point (det(Hessian) < 0)
    # Hessian of the quadratic: [[2a, c], [c, 2b]]
    hessian_det = 4*a*b - c**2
    if hessian_det >= 0:
        # Not a saddle point
        return (float(x), float(y)), False
    
    x_refined = x + offset[0]
    y_refined = y + offset[1]
    
    return (x_refined, y_refined), True


def detect_chessboard_saddle_corners(gray, sigma=1.5, threshold_ratio=0.05, 
                                     min_distance=15, subpixel_window=7,
                                     corner_quality_threshold=0.01):
    """
    Detect chessboard corners using saddle point detection with subpixel refinement.
    
    Args:
        gray: Grayscale image
        sigma: Gaussian smoothing sigma
        threshold_ratio: Threshold for saddle detection
        min_distance: Minimum distance between corners
        subpixel_window: Window size for subpixel refinement
        corner_quality_threshold: Minimum Harris corner response
        
    Returns:
        corners: List of (x, y) subpixel corner coordinates
        corners_int: List of initial integer coordinates
        det_hessian: Determinant of Hessian image (for visualization)
    """
    # Detect saddle points
    saddle_points, det_hessian, derivs = detect_saddle_points(
        gray, sigma=sigma, threshold_ratio=threshold_ratio, 
        min_distance=min_distance, corner_quality_threshold=corner_quality_threshold
    )
    
    # Refine each saddle point to subpixel accuracy
    refined_corners = []
    initial_corners = []
    
    gray_float = gray.astype(np.float64)
    
    for (x, y) in saddle_points:
        (x_ref, y_ref), success = refine_saddle_point_subpixel(
            gray_float, x, y, window_size=subpixel_window
        )
        
        if success:
            refined_corners.append((x_ref, y_ref))
            initial_corners.append((x, y))
    
    return refined_corners, initial_corners, det_hessian


def filter_corners_by_grid(corners, pattern_size=None, tolerance=0.3):
    """
    Filter and sort corners to match a chessboard grid pattern.
    
    Args:
        corners: List of (x, y) corner coordinates
        pattern_size: Expected (rows, cols) of inner corners, or None for auto-detection
        tolerance: Tolerance for grid alignment
        
    Returns:
        Filtered and sorted corners matching the grid pattern
    """
    if len(corners) < 4:
        return corners
    
    corners_array = np.array(corners)
    
    # Sort by y first, then by x to get row-major order
    # This is a simple approach - more robust methods exist
    sorted_indices = np.lexsort((corners_array[:, 0], corners_array[:, 1]))
    sorted_corners = corners_array[sorted_indices]
    
    return sorted_corners.tolist()


def detect_chessboard_corners_saddle(image_path, pattern_size=(7, 6), 
                                     sigma=1.5, threshold_ratio=0.03,
                                     min_distance=15, subpixel_window=7,
                                     use_opencv_refinement=True):
    """
    Main function to detect chessboard corners using saddle point method.
    
    This combines:
    1. Saddle point detection via Hessian matrix analysis
    2. Subpixel refinement via quadratic surface fitting
    3. Optional additional refinement using OpenCV's cornerSubPix
    
    Args:
        image_path: Path to the image file
        pattern_size: Expected (rows, cols) of inner corners
        sigma: Gaussian smoothing sigma for derivative computation
        threshold_ratio: Threshold for saddle point detection
        min_distance: Minimum distance between detected corners
        subpixel_window: Window size for quadratic fitting
        use_opencv_refinement: Whether to apply OpenCV cornerSubPix as final step
        
    Returns:
        Tuple of (success, corners, image_with_corners)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False, None, None

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Detecting saddle points in {image_path}...")
    
    # Detect saddle point corners
    corners, corners_int, det_hessian = detect_chessboard_saddle_corners(
        gray, 
        sigma=sigma, 
        threshold_ratio=threshold_ratio,
        min_distance=min_distance,
        subpixel_window=subpixel_window
    )
    
    print(f"Found {len(corners)} potential saddle point corners")
    
    if len(corners) == 0:
        print("No corners detected!")
        return False, None, img
    
    # Convert to numpy array for OpenCV
    corners_array = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    
    # Optional: Apply OpenCV's cornerSubPix for additional refinement
    if use_opencv_refinement and len(corners) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray, corners_array, (5, 5), (-1, -1), criteria
        )
        corners_array = corners_refined
    
    # Draw corners on the image
    img_with_corners = img.copy()
    
    # Draw initial integer corners in blue
    for (x, y) in corners_int:
        cv2.circle(img_with_corners, (int(x), int(y)), 5, (255, 0, 0), 1)
    
    # Draw refined subpixel corners in green
    for corner in corners_array:
        x, y = corner.ravel()
        cv2.circle(img_with_corners, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        # Draw crosshair for precise location
        cv2.line(img_with_corners, 
                 (int(round(x))-5, int(round(y))), 
                 (int(round(x))+5, int(round(y))), (0, 255, 0), 1)
        cv2.line(img_with_corners, 
                 (int(round(x)), int(round(y))-5), 
                 (int(round(x)), int(round(y))+5), (0, 255, 0), 1)
    
    # Create visualization of Hessian determinant
    det_normalized = np.clip(det_hessian, np.percentile(det_hessian, 1), 
                             np.percentile(det_hessian, 99))
    det_normalized = ((det_normalized - det_normalized.min()) / 
                      (det_normalized.max() - det_normalized.min()) * 255).astype(np.uint8)
    det_colored = cv2.applyColorMap(det_normalized, cv2.COLORMAP_JET)
    
    return True, corners_array, img_with_corners, det_colored


def detect_chessboard_corners_opencv(image_path, pattern_size=(7, 6)):
    """
    Detects chessboard corners using OpenCV's built-in method (for comparison).

    Args:
        image_path (str): Path to the image file.
        pattern_size (tuple): Number of inner corners (rows, columns).

    Returns:
        Tuple of (success, corners, image_with_corners)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False, None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        print(f"OpenCV found {len(corners)} corners")
        
        # Refine corner locations for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw corners
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret)
        
        return True, corners, img_with_corners
    else:
        print(f"OpenCV did not find chessboard pattern")
        return False, None, img


def compare_methods(image_path, pattern_size=(7, 6)):
    """
    Compare saddle point method with OpenCV's built-in method.
    """
    print("=" * 60)
    print("Comparing Saddle Point vs OpenCV Methods")
    print("=" * 60)
    
    # Saddle point method
    print("\n[1] Saddle Point Method with Subpixel Refinement:")
    result_saddle = detect_chessboard_corners_saddle(image_path, pattern_size)
    
    if len(result_saddle) == 4:
        success_saddle, corners_saddle, img_saddle, det_hessian = result_saddle
    else:
        success_saddle, corners_saddle, img_saddle = result_saddle[:3]
        det_hessian = None
    
    # OpenCV method
    print("\n[2] OpenCV Built-in Method:")
    success_opencv, corners_opencv, img_opencv = detect_chessboard_corners_opencv(
        image_path, pattern_size
    )
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if img_saddle is not None:
        cv2.imwrite(f"saddle_corners_{base_name}.png", img_saddle)
        print(f"Saved: saddle_corners_{base_name}.png")
        
    if det_hessian is not None:
        cv2.imwrite(f"hessian_det_{base_name}.png", det_hessian)
        print(f"Saved: hessian_det_{base_name}.png")
        
    if img_opencv is not None:
        cv2.imwrite(f"opencv_corners_{base_name}.png", img_opencv)
        print(f"Saved: opencv_corners_{base_name}.png")
    
    # Print corner coordinates if both methods succeeded
    if success_saddle and corners_saddle is not None:
        print(f"\nSaddle method found {len(corners_saddle)} corners")
        print("Sample corners (first 5):")
        for i, corner in enumerate(corners_saddle[:5]):
            x, y = corner.ravel()
            print(f"  Corner {i+1}: ({x:.3f}, {y:.3f})")
    
    if success_opencv and corners_opencv is not None:
        print(f"\nOpenCV method found {len(corners_opencv)} corners")
        print("Sample corners (first 5):")
        for i, corner in enumerate(corners_opencv[:5]):
            x, y = corner.ravel()
            print(f"  Corner {i+1}: ({x:.3f}, {y:.3f})")
    
    return (success_saddle, corners_saddle), (success_opencv, corners_opencv)


def create_synthetic_chessboard(size=(640, 480), square_size=50, offset=(50, 50)):
    """
    Create a synthetic chessboard image for testing.
    
    Args:
        size: Image size (width, height)
        square_size: Size of each square in pixels
        offset: Offset from top-left corner
        
    Returns:
        Grayscale chessboard image
    """
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 200  # Gray background
    
    ox, oy = offset
    num_cols = (size[0] - 2*ox) // square_size
    num_rows = (size[1] - 2*oy) // square_size
    
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = ox + col * square_size
            y1 = oy + row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            if (row + col) % 2 == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)  # White
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), 0, -1)    # Black
    
    # Add slight blur to make it more realistic
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img, (num_cols - 1, num_rows - 1)


def test_with_synthetic():
    """
    Test the saddle point detection with a synthetic chessboard.
    """
    print("Creating synthetic chessboard...")
    gray, inner_corners = create_synthetic_chessboard(
        size=(800, 600), square_size=60, offset=(80, 60)
    )
    
    # Save synthetic image
    cv2.imwrite("synthetic_chessboard.png", gray)
    print(f"Synthetic chessboard saved (expected inner corners: {inner_corners})")
    
    # Detect corners
    # For a 60-pixel square size, corners should be ~60 pixels apart
    # Use min_distance slightly less than half the square size
    print("\nDetecting corners with saddle point method...")
    corners, corners_int, det_hessian = detect_chessboard_saddle_corners(
        gray, sigma=2.0, threshold_ratio=0.1, min_distance=50, 
        subpixel_window=11, corner_quality_threshold=0.05
    )
    
    print(f"Detected {len(corners)} corners (expected: {inner_corners[0] * inner_corners[1]})")
    
    # Visualize
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x, y) in enumerate(corners):
        cv2.circle(img_color, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        # Draw index number
        cv2.putText(img_color, str(i), (int(x)+5, int(y)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    cv2.imwrite("synthetic_detected.png", img_color)
    print("Detection result saved as synthetic_detected.png")
    
    # Print accuracy for corners near expected grid positions
    print("\nCorner coordinates (subpixel precision):")
    for i, (x, y) in enumerate(corners[:min(10, len(corners))]):
        print(f"  Corner {i}: ({x:.4f}, {y:.4f})")
    
    return corners


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Chessboard Corner Detection using Saddle Points")
        print("=" * 50)
        print("\nUsage: python detect_corners.py <command> [options]")
        print("\nCommands:")
        print("  detect <image_path> [rows] [cols]  - Detect corners in image")
        print("  compare <image_path> [rows] [cols] - Compare saddle vs OpenCV methods")
        print("  test                               - Test with synthetic chessboard")
        print("\nExamples:")
        print("  python detect_corners.py detect chessboard.jpg 7 6")
        print("  python detect_corners.py compare calibration.png 9 6")
        print("  python detect_corners.py test")
        sys.exit(0)

    command = sys.argv[1].lower()
    
    if command == "test":
        test_with_synthetic()
        
    elif command in ["detect", "compare"]:
        if len(sys.argv) < 3:
            print(f"Error: {command} requires an image path")
            sys.exit(1)
            
        image_path = sys.argv[2]
        
        # Default pattern size
        rows, cols = 7, 6
        
        if len(sys.argv) >= 5:
            try:
                rows = int(sys.argv[3])
                cols = int(sys.argv[4])
            except ValueError:
                print("Rows and columns must be integers.")
                sys.exit(1)
        
        pattern_size = (rows, cols)
        print(f"Pattern size: {pattern_size}")
        
        if command == "detect":
            result = detect_chessboard_corners_saddle(image_path, pattern_size)
            if len(result) >= 3 and result[0]:
                output_name = f"corners_{os.path.basename(image_path)}"
                cv2.imwrite(output_name, result[2])
                print(f"Result saved as {output_name}")
        else:
            compare_methods(image_path, pattern_size)
    else:
        # Legacy: treat first argument as image path
        image_path = sys.argv[1]
        rows, cols = 7, 6
        
        if len(sys.argv) >= 4:
            try:
                rows = int(sys.argv[2])
                cols = int(sys.argv[3])
            except ValueError:
                print("Rows and columns must be integers.")
                sys.exit(1)
        
        pattern_size = (rows, cols)
        print(f"Looking for chessboard with pattern size: {pattern_size}")
        compare_methods(image_path, pattern_size)
