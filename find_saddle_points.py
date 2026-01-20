import numpy as np
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def find_subpixel_saddle_points(image_path, sigma=0.15):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale and normalize to float [0, 1]
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(float) / 255.0

    # 2. Smooth to handle real-world noise
    smoothed = ndimage.gaussian_filter(gray, sigma=sigma)

    # 3. Compute Hessian components
    # Using np.gradient provides central differences
    dy, dx = np.gradient(smoothed)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)

    # 4. Calculate Determinant of Hessian (DoH)
    # A negative determinant indicates a saddle point
    det_hessian = (dxx * dyy) - (dxy**2)
    
    # 5. Locate specific points
    # We look for the "most intense" negative values
    saddle_strength = -det_hessian 
    saddle_strength[saddle_strength < 0] = 0 # Only keep negative determinant areas (inverted)

    # Find local peaks of 'saddle-ness' at integer precision
    # min_distance=1 ensures we don't get adjacent pixels
    coordinates = peak_local_max(saddle_strength, min_distance=5, threshold_abs=0.001)

    subpixel_points = []
    
    for y, x in coordinates:
        # Check bounds for 3x3 patch
        if y < 1 or y >= saddle_strength.shape[0] - 1 or x < 1 or x >= saddle_strength.shape[1] - 1:
            continue
            
        # Extract 3x3 patch
        patch = saddle_strength[y-1:y+2, x-1:x+2]
        
        # Compute gradient and Hessian of the saddle_strength surface at (0,0) (center of patch)
        # using central differences
        
        # First derivatives
        # df/dx at (0,0)
        gx = 0.5 * (patch[1, 2] - patch[1, 0])
        # df/dy at (0,0)
        gy = 0.5 * (patch[2, 1] - patch[0, 1])
        
        # Second derivatives
        # d2f/dx2
        gxx = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]
        # d2f/dy2
        gyy = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]
        # d2f/dxdy
        gxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])
        
        # Hessian matrix
        H = np.array([[gxx, gxy], [gxy, gyy]])
        g = np.array([gx, gy])
        
        # Solve H * delta = -g  =>  delta = -H_inv * g
        try:
            # We are looking for a maximum, so H should be negative definite
            # But we just solve for stationary point regardless
            if np.linalg.det(H) == 0:
                continue
                
            delta = -np.linalg.solve(H, g)
            
            # Check if shift is within reasonable bounds (e.g. +/- 0.5 or 1.0 pixels)
            # If the quadratic fit peak is too far, it's likely unstable or not a real peak
            if abs(delta[0]) <= 1.0 and abs(delta[1]) <= 1.0:
                # delta is (dx, dy)
                sub_x = x + delta[0]
                sub_y = y + delta[1]
                subpixel_points.append((sub_x, sub_y))
            else:
                # Fallback to integer coordinate
                subpixel_points.append((float(x), float(y)))
                
        except np.linalg.LinAlgError:
            subpixel_points.append((float(x), float(y)))

    return np.array(subpixel_points), saddle_strength

if __name__ == "__main__":
    image_path = "sample_checkerboard.png"
    
    try:
        points, response_map = find_subpixel_saddle_points(image_path)
        
        print(f"Found {len(points)} saddle points.")
        for p in points[:5]:
            print(f"Subpixel coordinate: ({p[0]:.4f}, {p[1]:.4f})")
            
        # Visualize
        img = cv2.imread(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1], c='r', s=10, marker='+')
        plt.title('Detected Saddle Points (Subpixel)')
        plt.axis('off')
        plt.savefig('result_saddle_points.png')
        print("Result saved to result_saddle_points.png")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
