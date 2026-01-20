import cv2
import numpy as np
import glob
import sys
import os

def detect_chessboard_corners(image_path, pattern_size=(7, 6)):
    """
    Detects chessboard corners in an image using Saddle Points for subpixel accuracy.

    Args:
        image_path (str): Path to the image file.
        pattern_size (tuple): Number of inner corners per chessboard row and column (rows, columns).

    Returns:
        bool: True if corners were found, False otherwise.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check availability of findChessboardCornersSB
    if hasattr(cv2, 'findChessboardCornersSB'):
        # Use Sector-Based (Saddle Point) method with subpixel accuracy
        # CALIB_CB_ACCURACY enables subpixel refinement using saddle points
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        print("Using findChessboardCornersSB (Saddle Point method)...")
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    else:
        print("Warning: findChessboardCornersSB not available. Using standard findChessboardCorners.")
        # Fallback to standard method
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
             # Refine corner locations for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if ret:
        print(f"Corners found in {image_path}")
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        
        # Save the result
        output_filename = f"corners_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, img)
        print(f"Result saved as {output_filename}")
        
        # Print coordinates of detected corners
        print("Detected Corner Coordinates:")
        print(corners)
        return True
    else:
        print(f"Corners NOT found in {image_path}. Check pattern_size or image quality.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_corners.py <path_to_image> [rows] [columns]")
        print("Example: python detect_corners.py chessboard.jpg 7 6")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # Default pattern size
    rows = 7
    cols = 6
    
    if len(sys.argv) >= 4:
        try:
            rows = int(sys.argv[2])
            cols = int(sys.argv[3])
        except ValueError:
            print("Rows and columns must be integers.")
            sys.exit(1)
            
    pattern_size = (rows, cols)
    print(f"Looking for chessboard with pattern size: {pattern_size}")
    
    detect_chessboard_corners(image_path, pattern_size)
