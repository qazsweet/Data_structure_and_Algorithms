import cv2
import numpy as np
import os
import argparse
import glob

def calculate_sharpness(image_gray):
    """Calculate the sharpness of an image using the variance of the Laplacian."""
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    return laplacian.var()

def main():
    parser = argparse.ArgumentParser(description="Calculate sharpness plane from a sequence of images.")
    parser.add_argument("folder", help="Path to the folder containing images.")
    parser.add_argument("--rows", type=int, default=5, help="Number of block rows.")
    parser.add_argument("--cols", type=int, default=5, help="Number of block columns.")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    # Get image files and sort them
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    
    # Sort strictly by name
    image_files.sort(key=lambda x: os.path.basename(x))
    
    if not image_files:
        print(f"No images found in '{folder}'.")
        return

    print(f"Found {len(image_files)} images.")
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Error reading {image_files[0]}")
        return
    
    H, W = first_img.shape[:2]
    print(f"Image dimensions: {W}x{H}")

    n_rows = args.rows
    n_cols = args.cols
    block_h = H // n_rows
    block_w = W // n_cols

    # Store sharpness values: [row][col][image_index]
    sharpness_grid = np.zeros((n_rows, n_cols, len(image_files)))

    print("Processing images...")
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        if img.shape[:2] != (H, W):
            print(f"Warning: Image {img_path} has different dimensions {img.shape[:2]}, resizing to match first image.")
            img = cv2.resize(img, (W, H))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for r in range(n_rows):
            for c in range(n_cols):
                y0, y1 = r * block_h, (r + 1) * block_h
                x0, x1 = c * block_w, (c + 1) * block_w
                
                # Handle last block to include remaining pixels
                if r == n_rows - 1: y1 = H
                if c == n_cols - 1: x1 = W
                
                block = gray[y0:y1, x0:x1]
                score = calculate_sharpness(block)
                sharpness_grid[r, c, idx] = score
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images.")

    # Find max sharpness index for each block
    max_indices = np.argmax(sharpness_grid, axis=2)
    
    print("\nSharpest Image Indices (Grid):")
    print(max_indices)

    # Prepare data for plane fitting
    points_x = []
    points_y = []
    points_z = []

    for r in range(n_rows):
        for c in range(n_cols):
            # Calculate center coordinates
            y0, y1 = r * block_h, (r + 1) * block_h
            x0, x1 = c * block_w, (c + 1) * block_w
            if r == n_rows - 1: y1 = H
            if c == n_cols - 1: x1 = W
            
            center_x = (x0 + x1) / 2.0
            center_y = (y0 + y1) / 2.0
            
            best_idx = max_indices[r, c]
            
            points_x.append(center_x)
            points_y.append(center_y)
            points_z.append(best_idx)

    points_x = np.array(points_x)
    points_y = np.array(points_y)
    points_z = np.array(points_z)

    # Fit plane: z = a*x + b*y + d
    # A = [x, y, 1]
    A = np.column_stack((points_x, points_y, np.ones_like(points_x)))
    
    # Use least squares
    # Solution v = [a, b, d]
    v, residuals, rank, s = np.linalg.lstsq(A, points_z, rcond=None)
    a, b, d = v

    print("\nPlane Fitting Results:")
    print(f"Formula: Z = {a:.6f} * X + {b:.6f} * Y + {d:.6f}")
    
    # Calculate angles
    # Normal vector n = (a, b, -1)
    # Angle with Z-axis (0, 0, 1)
    
    normal = np.array([a, b, -1])
    norm_length = np.linalg.norm(normal)
    
    # Cosine of angle between normal and (0,0,1)
    # n . k = -1
    cos_theta = abs(-1) / (norm_length * 1)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    print(f"Angle between plane normal and Z-axis: {theta_deg:.4f} degrees")
    
    # Azimuth (direction of steepest ascent/descent)
    azimuth_deg = np.degrees(np.arctan2(b, a))
    print(f"Azimuth of the gradient: {azimuth_deg:.4f} degrees")

    # Tilt angles
    # Tilt X (rotation around Y axis sort of, slope in X direction)
    tilt_x_deg = np.degrees(np.arctan(a))
    # Tilt Y (slope in Y direction)
    tilt_y_deg = np.degrees(np.arctan(b))
    
    print(f"Slope in X direction (Tilt X): {tilt_x_deg:.4f} degrees")
    print(f"Slope in Y direction (Tilt Y): {tilt_y_deg:.4f} degrees")

if __name__ == "__main__":
    main()
