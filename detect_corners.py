import cv2
import numpy as np
import glob
import sys
import os

def detect_chessboard_corners(image_path, pattern_size=(7, 6)):
    """
    Detects chessboard corners in an image.

    Args:
        image_path (str): Path to the image file.
        pattern_size (tuple): Number of inner corners per chessboard row and column (rows, columns).
                              Note: This is (squares_in_row - 1, squares_in_col - 1).

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

    # Find the chess board corners
    # flags can be added for better detection, e.g., cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        print(f"Corners found in {image_path}")
        
        # Refine corner locations for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        
        # Save the result
        output_filename = f"corners_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, img)
        print(f"Result saved as {output_filename}")
        
        # Print coordinates of detected corners
        # print(corners2)
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
    
    # Default pattern size is 7x6 (standard for many calibration boards)
    # The user can override this via command line arguments
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
