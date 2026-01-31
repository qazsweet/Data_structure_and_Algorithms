"""
Image Sharpness Analysis and Plane Fitting

This script:
1. Loads images from a folder, sorted by name
2. Divides each image into 5x5 blocks
3. Calculates sharpness for each block
4. Finds the image index with maximum sharpness for each block
5. Fits a plane to the (x_center, y_center, best_index) data
6. Outputs plane formula and angles
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from numpy.linalg import lstsq


# Physical unit constants
PIXEL_SIZE_UM = 0.32  # Distance per pixel in micrometers (µm)
Z_STEP_UM = 1.0       # Distance between each image index in micrometers (µm)


@dataclass
class BlockInfo:
    """Information about a single block in the image grid."""
    row: int
    col: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    center_x: float
    center_y: float


def get_image_files(folder_path: str, extensions: Tuple[str, ...] = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')) -> List[str]:
    """
    Get all image files from a folder, sorted by name.
    
    Args:
        folder_path: Path to the folder containing images
        extensions: Tuple of valid image extensions
        
    Returns:
        List of full paths to image files, sorted by filename
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            image_files.append(os.path.join(folder_path, filename))
    
    # Sort by filename (not full path) to handle numeric naming correctly
    image_files.sort(key=lambda x: os.path.basename(x))
    
    return image_files


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate the sharpness of an image using the variance of Laplacian.
    
    Higher values indicate sharper images.
    
    Args:
        image: Grayscale image as numpy array
        
    Returns:
        Sharpness value (variance of Laplacian)
    """
    if image is None or image.size == 0:
        return 0.0
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Variance of Laplacian is a good measure of sharpness
    sharpness = laplacian.var()
    
    return float(sharpness)


def divide_into_blocks(image_height: int, image_width: int, grid_rows: int = 5, grid_cols: int = 5) -> List[BlockInfo]:
    """
    Divide an image into a grid of blocks.
    
    Args:
        image_height: Height of the image
        image_width: Width of the image
        grid_rows: Number of rows in the grid (default 5)
        grid_cols: Number of columns in the grid (default 5)
        
    Returns:
        List of BlockInfo objects describing each block
    """
    block_height = image_height // grid_rows
    block_width = image_width // grid_cols
    
    blocks = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x_start = col * block_width
            y_start = row * block_height
            
            # Handle last row/col to include remaining pixels
            if col == grid_cols - 1:
                x_end = image_width
            else:
                x_end = (col + 1) * block_width
                
            if row == grid_rows - 1:
                y_end = image_height
            else:
                y_end = (row + 1) * block_height
            
            center_x = (x_start + x_end) / 2.0
            center_y = (y_start + y_end) / 2.0
            
            blocks.append(BlockInfo(
                row=row,
                col=col,
                x_start=x_start,
                y_start=y_start,
                x_end=x_end,
                y_end=y_end,
                center_x=center_x,
                center_y=center_y
            ))
    
    return blocks


def extract_block(image: np.ndarray, block: BlockInfo) -> np.ndarray:
    """
    Extract a block from an image.
    
    Args:
        image: The source image
        block: BlockInfo describing the region to extract
        
    Returns:
        The extracted block as a numpy array
    """
    return image[block.y_start:block.y_end, block.x_start:block.x_end]


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a plane z = ax + by + c to the given points using least squares.
    
    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        z: Array of z values (image indices with max sharpness)
        
    Returns:
        Tuple of (a, b, c) coefficients for the plane equation z = ax + by + c
    """
    # Build the design matrix: [x, y, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    
    # Solve the least squares problem
    coeffs, residuals, rank, s = lstsq(A, z, rcond=None)
    
    a, b, c = coeffs
    return float(a), float(b), float(c)


def calculate_plane_angles(a: float, b: float) -> Tuple[float, float, float]:
    """
    Calculate the angles of the plane with respect to the coordinate axes.
    
    For plane z = ax + by + c, the normal vector is (-a, -b, 1).
    
    Args:
        a: Coefficient for x in plane equation
        b: Coefficient for y in plane equation
        
    Returns:
        Tuple of (angle_x, angle_y, angle_with_xy_plane) in milliradians (mrad)
        - angle_x: Angle between plane's projection on XZ plane and XY plane
        - angle_y: Angle between plane's projection on YZ plane and XY plane
        - angle_with_xy_plane: Angle between plane normal and Z-axis
    """
    # Normal vector to the plane is (a, b, -1) for ax + by - z + c = 0
    # Or we can use (-a, -b, 1) which points "upward"
    normal = np.array([-a, -b, 1.0])
    normal_mag = np.linalg.norm(normal)
    
    # Angle with Z-axis (angle between normal and Z-axis)
    # This tells us how much the plane is tilted from horizontal
    z_axis = np.array([0, 0, 1])
    cos_angle_z = np.dot(normal, z_axis) / normal_mag
    # Convert radians to milliradians (1 rad = 1000 mrad)
    angle_with_z = np.arccos(np.clip(cos_angle_z, -1.0, 1.0)) * 1000.0
    
    # Tilt angle around X-axis (looking along X, rotation in YZ plane)
    # This is arctan(b) - the slope in Y direction
    angle_x = np.arctan(b) * 1000.0  # Convert to mrad
    
    # Tilt angle around Y-axis (looking along Y, rotation in XZ plane)
    # This is arctan(a) - the slope in X direction
    angle_y = np.arctan(a) * 1000.0  # Convert to mrad
    
    return angle_x, angle_y, angle_with_z


def process_images(
    folder_path: str,
    grid_rows: int = 5,
    grid_cols: int = 5,
    verbose: bool = True
) -> Tuple[np.ndarray, List[BlockInfo], List[str], np.ndarray]:
    """
    Process all images in a folder and calculate sharpness for each block.
    
    Args:
        folder_path: Path to folder containing images
        grid_rows: Number of rows in the block grid
        grid_cols: Number of columns in the block grid
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - sharpness_matrix: (num_images, num_blocks) array of sharpness values
        - blocks: List of BlockInfo objects
        - image_files: List of image file paths
        - best_indices: Array of image indices with max sharpness for each block
    """
    image_files = get_image_files(folder_path)
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {folder_path}")
    
    if verbose:
        print(f"Found {len(image_files)} images in {folder_path}")
        print(f"Processing with {grid_rows}x{grid_cols} block grid...")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {image_files[0]}")
    
    height, width = first_image.shape[:2]
    blocks = divide_into_blocks(height, width, grid_rows, grid_cols)
    num_blocks = len(blocks)
    
    if verbose:
        print(f"Image size: {width}x{height}")
        print(f"Block count: {num_blocks}")
    
    # Initialize sharpness matrix
    sharpness_matrix = np.zeros((len(image_files), num_blocks), dtype=np.float64)
    
    # Process each image
    for img_idx, img_path in enumerate(image_files):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}, skipping...")
            continue
        
        # Convert to grayscale once
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate sharpness for each block
        for block_idx, block in enumerate(blocks):
            block_image = extract_block(gray, block)
            sharpness = calculate_sharpness(block_image)
            sharpness_matrix[img_idx, block_idx] = sharpness
        
        if verbose and (img_idx + 1) % 10 == 0:
            print(f"  Processed {img_idx + 1}/{len(image_files)} images...")
    
    if verbose:
        print(f"  Processed {len(image_files)}/{len(image_files)} images... Done!")
    
    # Find the index of the image with maximum sharpness for each block
    best_indices = np.argmax(sharpness_matrix, axis=0)
    
    return sharpness_matrix, blocks, image_files, best_indices


def analyze_and_fit_plane(
    blocks: List[BlockInfo],
    best_indices: np.ndarray,
    pixel_size_um: float = PIXEL_SIZE_UM,
    z_step_um: float = Z_STEP_UM,
    verbose: bool = True
) -> Tuple[float, float, float, float, float, float]:
    """
    Fit a plane to the block centers and best image indices.
    
    Coordinates are converted to physical units (micrometers) before fitting:
    - x, y: pixel coordinates * pixel_size_um
    - z: image index * z_step_um
    
    Args:
        blocks: List of BlockInfo objects
        best_indices: Array of image indices with max sharpness for each block
        pixel_size_um: Size of each pixel in micrometers (default: 0.32 µm)
        z_step_um: Distance between each image index in micrometers (default: 1 µm)
        verbose: Whether to print results
        
    Returns:
        Tuple of (a, b, c, angle_x, angle_y, angle_with_z)
    """
    # Extract coordinates in pixels
    x_pixels = np.array([block.center_x for block in blocks])
    y_pixels = np.array([block.center_y for block in blocks])
    z_indices = best_indices.astype(np.float64)
    
    # Convert to physical units (micrometers)
    x_um = x_pixels * pixel_size_um
    y_um = y_pixels * pixel_size_um
    z_um = z_indices * z_step_um
    
    # Fit plane in physical units: z(µm) = a*x(µm) + b*y(µm) + c(µm)
    a, b, c = fit_plane(x_um, y_um, z_um)
    
    # Calculate angles (a and b are now dimensionless: µm/µm)
    angle_x, angle_y, angle_with_z = calculate_plane_angles(a, b)
    
    if verbose:
        print("\n" + "="*60)
        print("PLANE FITTING RESULTS")
        print("="*60)
        print(f"\nPhysical units: pixel size = {pixel_size_um} µm, z step = {z_step_um} µm")
        print(f"\nPlane equation: z(µm) = {a:.10f}*x(µm) + {b:.10f}*y(µm) + {c:.6f}")
        print(f"\nAlternatively: {a:.10f}*x + {b:.10f}*y - z + {c:.6f} = 0  (all in µm)")
        print(f"\nCoefficients (dimensionless slopes):")
        print(f"  a (dz/dx): {a:.10f}")
        print(f"  b (dz/dy): {b:.10f}")
        print(f"  c (z-intercept in µm): {c:.6f}")
        print(f"\nPlane angles:")
        print(f"  Tilt around X-axis (slope in Y direction): {angle_x:.4f} mrad")
        print(f"  Tilt around Y-axis (slope in X direction): {angle_y:.4f} mrad")
        print(f"  Angle between plane normal and Z-axis:     {angle_with_z:.4f} mrad")
        print(f"\nNormal vector: ({-a:.10f}, {-b:.10f}, 1.0)")
        print("="*60)
    
    return a, b, c, angle_x, angle_y, angle_with_z


def print_detailed_results(
    sharpness_matrix: np.ndarray,
    blocks: List[BlockInfo],
    image_files: List[str],
    best_indices: np.ndarray
):
    """Print detailed results showing sharpness values and best images for each block."""
    print("\n" + "="*60)
    print("DETAILED BLOCK ANALYSIS")
    print("="*60)
    
    num_blocks = len(blocks)
    grid_size = int(np.sqrt(num_blocks))
    
    print(f"\nBest image index for each block (0-indexed):")
    print("-" * 40)
    
    for row in range(grid_size):
        row_str = ""
        for col in range(grid_size):
            block_idx = row * grid_size + col
            row_str += f"{best_indices[block_idx]:4d} "
        print(f"Row {row}: {row_str}")
    
    print(f"\nMaximum sharpness value for each block:")
    print("-" * 40)
    
    max_sharpness = np.max(sharpness_matrix, axis=0)
    for row in range(grid_size):
        row_str = ""
        for col in range(grid_size):
            block_idx = row * grid_size + col
            row_str += f"{max_sharpness[block_idx]:10.2f} "
        print(f"Row {row}: {row_str}")
    
    print(f"\nBlock center coordinates and best index:")
    print("-" * 60)
    print(f"{'Block':>8} {'Center X':>12} {'Center Y':>12} {'Best Idx':>10} {'Max Sharp':>12}")
    print("-" * 60)
    
    for block_idx, block in enumerate(blocks):
        print(f"{block_idx:>8} {block.center_x:>12.2f} {block.center_y:>12.2f} "
              f"{best_indices[block_idx]:>10} {max_sharpness[block_idx]:>12.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze image sharpness in blocks and fit a focus plane."
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=5,
        help="Number of rows in block grid (default: 5)"
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=5,
        help="Number of columns in block grid (default: 5)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed results for each block"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (CSV format)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        # Process all images
        sharpness_matrix, blocks, image_files, best_indices = process_images(
            args.folder,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            verbose=verbose
        )
        
        # Print image list
        if verbose:
            print(f"\nImages (sorted by name):")
            for i, img_path in enumerate(image_files):
                print(f"  [{i:3d}] {os.path.basename(img_path)}")
        
        # Fit plane and get results
        a, b, c, angle_x, angle_y, angle_with_z = analyze_and_fit_plane(
            blocks, best_indices, verbose=verbose
        )
        
        # Print detailed results if requested
        if args.detailed:
            print_detailed_results(sharpness_matrix, blocks, image_files, best_indices)
        
        # Save results to CSV if requested
        if args.output:
            save_results_to_csv(args.output, sharpness_matrix, blocks, image_files, 
                              best_indices, a, b, c, angle_x, angle_y, angle_with_z)
            if verbose:
                print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def save_results_to_csv(
    output_path: str,
    sharpness_matrix: np.ndarray,
    blocks: List[BlockInfo],
    image_files: List[str],
    best_indices: np.ndarray,
    a: float, b: float, c: float,
    angle_x: float, angle_y: float, angle_with_z: float,
    pixel_size_um: float = PIXEL_SIZE_UM,
    z_step_um: float = Z_STEP_UM
):
    """Save results to a CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write plane info
        writer.writerow(['# Plane Fitting Results'])
        writer.writerow(['Pixel Size (um)', pixel_size_um])
        writer.writerow(['Z Step (um)', z_step_um])
        writer.writerow(['Plane Equation', f'z(um) = {a}*x(um) + {b}*y(um) + {c}'])
        writer.writerow(['Coefficient a (dz/dx)', a])
        writer.writerow(['Coefficient b (dz/dy)', b])
        writer.writerow(['Coefficient c (z-intercept um)', c])
        writer.writerow(['Tilt around X-axis (mrad)', angle_x])
        writer.writerow(['Tilt around Y-axis (mrad)', angle_y])
        writer.writerow(['Angle with Z-axis (mrad)', angle_with_z])
        writer.writerow([])
        
        # Write block info
        writer.writerow(['# Block Information'])
        header = ['Block Index', 'Row', 'Col', 'Center X (pixels)', 'Center Y (pixels)', 
                  'Center X (um)', 'Center Y (um)', 'Best Image Index', 'Best Z (um)',
                  'Best Image Name', 'Max Sharpness']
        writer.writerow(header)
        
        max_sharpness = np.max(sharpness_matrix, axis=0)
        for block_idx, block in enumerate(blocks):
            best_idx = best_indices[block_idx]
            writer.writerow([
                block_idx,
                block.row,
                block.col,
                block.center_x,
                block.center_y,
                block.center_x * pixel_size_um,
                block.center_y * pixel_size_um,
                best_idx,
                best_idx * z_step_um,
                os.path.basename(image_files[best_idx]),
                max_sharpness[block_idx]
            ])
        
        writer.writerow([])
        
        # Write full sharpness matrix
        writer.writerow(['# Full Sharpness Matrix (rows=images, cols=blocks)'])
        header = ['Image Index', 'Image Name'] + [f'Block {i}' for i in range(len(blocks))]
        writer.writerow(header)
        
        for img_idx, img_path in enumerate(image_files):
            row = [img_idx, os.path.basename(img_path)] + list(sharpness_matrix[img_idx])
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
