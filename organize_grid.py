import numpy as np
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix

def find_subpixel_saddle_points(image_path, sigma=0.15):
    # (Previous implementation logic will be reused or imported)
    # For now, I'll copy the logic briefly to keep it self-contained in one file for the final delivery,
    # or I can import it if I split files. I'll paste the core logic here to ensure it works.
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(float) / 255.0

    smoothed = ndimage.gaussian_filter(gray, sigma=sigma)
    dy, dx = np.gradient(smoothed)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    det_hessian = (dxx * dyy) - (dxy**2)
    saddle_strength = -det_hessian 
    saddle_strength[saddle_strength < 0] = 0 

    coordinates = peak_local_max(saddle_strength, min_distance=5, threshold_abs=0.001)

    subpixel_points = []
    
    for y, x in coordinates:
        if y < 1 or y >= saddle_strength.shape[0] - 1 or x < 1 or x >= saddle_strength.shape[1] - 1:
            continue
        patch = saddle_strength[y-1:y+2, x-1:x+2]
        gx = 0.5 * (patch[1, 2] - patch[1, 0])
        gy = 0.5 * (patch[2, 1] - patch[0, 1])
        gxx = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]
        gyy = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]
        gxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])
        H = np.array([[gxx, gxy], [gxy, gyy]])
        g = np.array([gx, gy])
        
        try:
            if np.linalg.det(H) == 0:
                continue
            delta = -np.linalg.solve(H, g)
            if abs(delta[0]) <= 1.0 and abs(delta[1]) <= 1.0:
                sub_x = x + delta[0]
                sub_y = y + delta[1]
                subpixel_points.append((sub_x, sub_y))
            else:
                subpixel_points.append((float(x), float(y)))
        except np.linalg.LinAlgError:
            subpixel_points.append((float(x), float(y)))

    return np.array(subpixel_points), img

def organize_grid(points, image_shape):
    """
    Organizes unstructured points into a structured grid.
    
    1. Find the top-left corner.
    2. Determine local grid orientation/spacing.
    3. Propagate through the grid.
    """
    if len(points) == 0:
        return []

    # 1. Find the top-left point (smallest x+y is a good heuristic for top-left in image coords)
    # Alternatively, start from the most central point and grow out, but top-left is easier to index.
    # Let's use a robust method: spatial sorting.
    
    # Sort points roughly by Y, then by X. But purely sorting is brittle to rotation.
    # Better approach: KD-Tree or Nearest Neighbors graph.
    
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Estimate median spacing to filter outlier connections
    median_spacing = np.median(distances[:, 1]) # 0 is self, 1 is nearest neighbor
    
    print(f"Estimated grid spacing: {median_spacing:.2f} pixels")
    
    # Heuristic to find a corner: points with fewer neighbors within range
    # Or just find the top-left-most point: min(x + y)
    sum_xy = points[:, 0] + points[:, 1]
    start_idx = np.argmin(sum_xy)
    start_point = points[start_idx]
    
    # This is a simplified "growth" algorithm.
    # Ideally, we want to classify 'right' and 'down' vectors.
    
    # Let's verify valid neighbors for the start point
    valid_neighbors = []
    for idx in indices[start_idx][1:]: # Skip self
        dist = np.linalg.norm(points[idx] - start_point)
        if dist < 1.5 * median_spacing: # Tolerance
            valid_neighbors.append(points[idx])
            
    if not valid_neighbors:
        print("Start point has no valid neighbors. Grid organization failed.")
        return []
        
    # Determine primary axes from start point neighbors
    # We expect 2 orthogonal neighbors for a corner
    # Vector analysis:
    vectors = [p - start_point for p in valid_neighbors]
    
    # Filter vectors that are roughly axis-aligned or consistent
    # For a general case, we just pick the two closest neighbors that are roughly orthogonal?
    # Or just assume the grid is somewhat aligned with image axes (common in calibration).
    # Let's simply sort by X and Y.
    
    # Better yet: Let's assume the grid rows are somewhat horizontal.
    # We can cluster points into "rows" based on Y-coordinate if rotation is small.
    # If rotation is large, we need RANSAC or similar.
    
    # Let's try a row-clustering approach, assuming rotation < 45 degrees.
    # 1. Sort by Y
    sorted_indices = np.argsort(points[:, 1])
    sorted_points = points[sorted_indices]
    
    rows = []
    current_row = [sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        p = sorted_points[i]
        # If y difference is small, it's the same row
        if abs(p[1] - current_row[-1][1]) < 0.5 * median_spacing:
            current_row.append(p)
        else:
            # Sort the completed row by X
            current_row.sort(key=lambda k: k[0])
            rows.append(current_row)
            current_row = [p]
    
    # Append last row
    if current_row:
        current_row.sort(key=lambda k: k[0])
        rows.append(current_row)
        
    # Check if rows form a grid structure (similar lengths)
    # Some rows might be shorter due to noise or detection failures.
    
    # Let's refine rows: align them into a rectangular grid structure (list of lists)
    max_cols = max(len(r) for r in rows)
    print(f"Detected {len(rows)} rows, max columns: {max_cols}")
    
    # Pad or align rows? 
    # For a checkerboard, we usually expect a consistent grid size (e.g. 7x7).
    # If points are missing, we might leave None.
    
    grid = []
    for r in rows:
        grid.append(np.array(r))
        
    return grid

if __name__ == "__main__":
    # Create a synthetic checkerboard again for testing
    size = 256
    squares = 8
    image = np.zeros((size, size), dtype=np.uint8)
    square_size = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 1:
                cv2.rectangle(image, (j*square_size, i*square_size), 
                            ((j+1)*square_size, (i+1)*square_size), 255, -1)
    
    # Rotate it slightly to test robustness (optional, kept simple for now)
    # M = cv2.getRotationMatrix2D((size/2, size/2), 10, 1)
    # image = cv2.warpAffine(image, M, (size, size))
    
    cv2.imwrite("test_grid.png", image)
    
    points, _ = find_subpixel_saddle_points("test_grid.png")
    print(f"Found {len(points)} raw points.")
    
    grid_rows = organize_grid(points, image.shape)
    
    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(grid_rows)))
    
    for i, row in enumerate(grid_rows):
        if len(row) > 0:
            plt.plot(row[:, 0], row[:, 1], 'o-', color=colors[i], label=f'Row {i}')
            
    plt.title("Organized Checkerboard Grid")
    plt.legend()
    plt.savefig("organized_grid_result.png")
    print("Saved organized_grid_result.png")
