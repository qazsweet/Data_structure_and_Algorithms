import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage import color, feature, measure
import checkerboard
from scipy.ndimage import gaussian_filter, laplace
from skimage.feature import peak_local_max
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndimage
import scipy.interpolate as Rbf


def robust_chessboard_detection(image_path, pattern_size=(9, 6)):

    gray = image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 使用CLAHE平衡模糊导致的低对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 2. 使用更强力的 SB (Sector Based) 算法作为第一备选
    # 这是针对模糊优化的内置算法
    ret, corners = cv2.findChessboardCornersSB(
        gray, pattern_size, 
        flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    )
    
    if ret:
        print("Success using SB algorithm!")
        return corners
    
    else:
        # 使用LSD检测直线（OpenCV 3.x+ 需单独安装或使用LineSegmentDetector_create）
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(gray)[0]
        print("lines:", lines.shape)

    
    return Lines

def get_intersections(image_gray, pattern_size=(9, 6)):
    # 1. 提取直线 (使用 FastLineDetector)
    # fld 在模糊图像上比 Hough 更稳健
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(image_gray)
    
    horizontal_lines = []
    vertical_lines = []
    
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 过滤掉过短的线段（例如小于图像宽度的 1/20）
        if length < image_gray.shape[1] / 20:
            continue
            
        # 计算斜率和截距 (y = kx + b)
        # 注意：处理垂直线时 k 为无穷大，所以建议记录 (cos theta, sin theta) 或 (k, b)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # 2. 根据角度分类 (允许一定的倾斜范围)
        if (angle < 20) or (angle > 160):
            horizontal_lines.append(line[0])
        elif (70 < angle < 110):
            vertical_lines.append(line[0])
            
    return horizontal_lines, vertical_lines

def compute_intersect(l1, l2):
    """计算两条线段所在直线的交点"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0: # 平行
        return None
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)



def sharpen_for_corners(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def solve_chessboard_by_lines(img_path):
    gray = cv2.imread(img_path, 0)
    h_lines, v_lines = get_intersections(gray)
    
    points = []
    for hl in h_lines:
        for vl in v_lines:
            p = compute_intersect(hl, vl)
            if p and (0 <= p[0] < gray.shape[1]) and (0 <= p[1] < gray.shape[0]):
                points.append(p)
                
    # 转换为numpy数组
    points = np.array(points, dtype=np.float32)
    
    # 后续：你可以使用 cv2.kmeans 或简单的距离过滤来清理重复点
    # 然后利用 cv2.findHomography(src, points, cv2.RANSAC) 来精确定位
    return points


def sort_and_filter_corners(candidate_points, pattern_size=(9, 6)):
    """
    candidate_points: 之前通过直线交点得到的列表或 numpy 数组 [(x,y), ...]
    pattern_size: (cols, rows) 这里的 cols 是长边, rows 是短边
    """
    pts = np.array(candidate_points, dtype=np.float32)
    
    # 1. 粗略筛选：通过聚类剔除背景中的孤立干扰点
    # 如果点很多，可以使用 K-Means 选出最密集的点群，或者直接进入下一步
    
    # 2. 寻找棋盘格的四个极端顶点（外壳）
    # 常用技巧：x+y 最小是左上，x+y 最大是右下，x-y 最小是左下，x-y 最大是右上
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]       # Top-left
    br = pts[np.argmax(s)]       # Bottom-right
    tr = pts[np.argmin(diff)]    # Top-right
    bl = pts[np.argmax(diff)]    # Bottom-left
    
    src_corners = np.array([tl, tr, br, bl], dtype="float32")
    
    # 3. 建立理想的网格坐标 (用于匹配)
    cols, rows = pattern_size
    # 生成理想坐标：(0,0), (1,0), (2,0)...(cols-1, rows-1)
    ideal_grid = np.zeros((cols * rows, 2), np.float32)
    ideal_grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    
    # 对应理想网格的四个角
    ideal_corners = np.array([
        [0, 0], 
        [cols-1, 0], 
        [cols-1, rows-1], 
        [0, rows-1]
    ], dtype="float32")
    
    # 4. 计算单应性矩阵 H
    # H 将理想的网格坐标映射到当前模糊图像的坐标空间
    H, _ = cv2.findHomography(ideal_corners, src_corners)
    
    # 5. 生成预测点并寻找最近邻
    # 将所有理想点映射回图像
    predicted_pts = cv2.perspectiveTransform(ideal_grid.reshape(-1, 1, 2), H).reshape(-1, 2)
    
    final_sorted_corners = []
    for p_pt in predicted_pts:
        # 在 candidate_points 中找离预测位置最近的点
        distances = np.linalg.norm(pts - p_pt, axis=1)
        idx = np.argmin(distances)
        
        # 如果距离太远（说明该位置没找到交点），可以考虑直接用预测点代替或报错
        if distances[idx] < 20: # 阈值视图像大小调整
            final_sorted_corners.append(pts[idx])
        else:
            final_sorted_corners.append(p_pt) # 或者存入预测值作为补全
            
    return np.array(final_sorted_corners).reshape(-1, 1, 2)

def sharpen_image(image):
    # 1. Blur the image slightly (Gaussian Blur)
    # ksize (5,5) and sigma 1.0 are standard for chessboard scales
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1.0)
    
    # 2. Calculate the sharpened image: 
    # Formula: Result = Original + (Original - Blurred) * Amount
    sharpened = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)
    
    return sharpened

def find_saddle_points(image, sigma=1.0, threshold=0.1):
    # 1. Smooth the image to reduce noise (crucial for second derivatives)
    smoothed = gaussian_filter(image, sigma=sigma)
    
    # 2. Compute first derivatives (Gradients)
    dy, dx = np.gradient(smoothed)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    # 3. Compute second derivatives (Hessian components)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    
    # 4. Calculate the Determinant of the Hessian
    # det(H) = Ixx*Iyy - Ixy^2
    det_hessian = (dxx * dyy) - (dxy**2)
    
    # 5. Define a Saddle Point: 
    # - Gradient is near zero (local critical point)
    # - Determinant is negative (opposite curvatures)
    saddle_mask = (det_hessian < -threshold) & (gradient_magnitude < np.percentile(gradient_magnitude, 10))
    
    return saddle_mask, det_hessian

def find_grid_intersections(horizontal, vertical, min_distance=10):
    """
    Find intersection points where horizontal and vertical grid lines meet.
    Returns list of (x, y) coordinates.
    """
    # Use HoughLinesP to detect line segments
    h_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    v_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    intersections = []
    
    if h_lines is not None and v_lines is not None:
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line[0]
            for v_line in v_lines:
                x3, y3, x4, y4 = v_line[0]
                # Compute intersection
                p = compute_intersect((x1, y1, x2, y2), (x3, y3, x4, y4))
                if p is not None:
                    intersections.append(p)
    
    # Remove duplicate points (points too close to each other)
    if len(intersections) > 0:
        intersections = np.array(intersections)
        # Use distance-based filtering
        filtered_points = []
        for point in intersections:
            if len(filtered_points) == 0:
                filtered_points.append(point)
            else:
                distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) 
                            for p in filtered_points]
                if min(distances) > min_distance:
                    filtered_points.append(point)
        return np.array(filtered_points)
    
    return np.array([])

def find_harris_corners_in_grid(image, grid_mask, block_size=3, ksize=3, k=0.04, threshold=0.01):
    """
    Find Harris corners within the grid area.
    """
    # Detect Harris corners
    harris_response = cv2.cornerHarris(image, block_size, ksize, k)
    
    # Normalize and threshold
    harris_response = cv2.dilate(harris_response, None)
    harris_thresh = threshold * harris_response.max()
    
    # Find corners within grid area
    corner_mask = (harris_response > harris_thresh) & (grid_mask > 0)
    corner_coords = np.column_stack(np.where(corner_mask))
    
    # Convert from (row, col) to (x, y)
    if len(corner_coords) > 0:
        points = corner_coords[:, [1, 0]].astype(np.float32)
        return points
    return np.array([])

def find_shi_tomasi_corners_in_grid(image, grid_mask, max_corners=200, quality_level=0.01, min_distance=10):
    """
    Find Shi-Tomasi corners within the grid area.
    """
    # Create a mask for corner detection (only in grid area)
    mask = grid_mask.copy()
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(image, maxCorners=max_corners, 
                                      qualityLevel=quality_level, 
                                      minDistance=min_distance,
                                      mask=mask)
    
    if corners is not None:
        return corners.reshape(-1, 2)
    return np.array([])

def find_saddle_points_in_grid(image, grid_mask, sigma=2.0, threshold=0.1):
    """
    Find saddle points (grid corners) within the grid area.
    """
    saddles, det_hessian = find_saddle_points(image, sigma, threshold)
    
    # Filter to only grid area
    saddle_mask = saddles & (grid_mask > 0)
    saddle_coords = np.column_stack(np.where(saddle_mask))
    
    if len(saddle_coords) > 0:
        # Convert from (row, col) to (x, y)
        points = saddle_coords[:, [1, 0]].astype(np.float32)
        return points
    return np.array([])

def combine_and_refine_feature_points(all_points, min_distance=15):
    """
    Combine feature points from multiple methods and remove duplicates.
    """
    if len(all_points) == 0:
        return np.array([])
    
    # Combine all points
    combined = np.vstack(all_points) if len(all_points) > 1 else all_points[0]
    
    # Remove duplicates using distance-based clustering
    if len(combined) == 0:
        return np.array([])
    
    refined_points = []
    for point in combined:
        if len(refined_points) == 0:
            refined_points.append(point)
        else:
            distances = [np.linalg.norm(point - p) for p in refined_points]
            if min(distances) > min_distance:
                refined_points.append(point)
    
    return np.array(refined_points)



def detect_and_save_saddles(image_path, sigma=1.5, output_csv="saddle_points.csv", keywords='test'):
    # 1. Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # [:300, :300]
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Convert to grayscale and normalize to float [0, 1]
    gray = img.astype(float) / 255.0

    # 2. Smooth to handle real-world noise
    # sigma=0.15# 2.9
    smoothed = gaussian_filter(gray, sigma=sigma)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # smoothed = clahe.apply((smoothed1*255).astype(np.uint8))


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
    # We invert det_hessian because peak_local_max finds maxima
    saddle_strength = -det_hessian 
    saddle_strength[saddle_strength < 0] = 0 # Only keep negative determinant areas
    
    # Find local peaks of 'saddle-ness'
    max_val = np.max(saddle_strength)
    print(f"Max saddle strength found: {max_val}")

    # Auto-set threshold to 10% of the maximum detected strength
    dynamic_threshold = max_val * 0.15

    coordinates = peak_local_max(
        saddle_strength, 
        min_distance=300,           # Look closer together
        threshold_abs=dynamic_threshold, # Use a relative threshold
        exclude_border=True
    )

    # 6. Save coordinates to CSV
    # coordinates are in (row, col) format -> convert to (x, y)
    df = pd.DataFrame(coordinates, columns=['y', 'x'])
    df = df[['x', 'y']] # Reorder
    df.to_csv(output_csv, index=False)
    print(f"df:{df.shape}")
    
    print(f"Detected {len(coordinates)} saddle points.")
    print(f"Coordinates saved to {output_csv}")

    img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)/255
    for coord in coordinates:
        cv2.circle(img_copy, (coord[1], coord[0]), 30, (1, 0, 1), -1) # Red dots

    plt.figure()
    plt.imshow(img_copy)
    plt.title("Detected Saddle Points")
    plt.show()
    plt.savefig(f'Detected_rgb_points_{keywords}.png')
    plt.close()
    return coordinates

def find_index_from_coordinates(coordinates):
    coordinates = np.array(coordinates) 
    sorted_coords = sorted(coordinates, key=lambda x: (x[0]+ x[1]))
    
    y1, x1 = sorted_coords[0]
    # y2, x2  = sorted_coords[1]
    # y3, x3  = sorted_coords[2]
    # y4, x4  = sorted_coords[3]
    # print(f"coord4:{x4}, {y4}")
    # # Compute distances from all points in xy_coordinates to (x1, y1)
    distances = np.linalg.norm(coordinates - np.array([y1, x1]), axis=1)
    # Find the indices of the first and second nearest points
    flat_distances = distances.flatten()
    nearest_indices = np.argpartition(flat_distances, 1)[:3]
    # Reshape flat indices to 2D indices
    nearest_points = [np.unravel_index(idx, distances.shape) for idx in nearest_indices]
    print(f"First nearest index: {nearest_points[0]}, distance: {flat_distances[nearest_indices[0]]}, coords:{coordinates[nearest_points[0]]}")
    print(f"Second nearest index: {nearest_points[1]}, distance: {flat_distances[nearest_indices[1]]}, coords:{coordinates[nearest_points[1]]}")
    print(f"Third nearest index: {nearest_points[2]}, distance: {flat_distances[nearest_indices[2]]}, coords:{coordinates[nearest_points[2]]}")
    
    y2, x2  = coordinates[nearest_points[1]]
    y3, x3  = coordinates[nearest_points[2]]
    print(f"coord1:{x1}, {y1}")
    print(f"coord2:{x2}, {y2}")
    print(f"coord3:{x3}, {y3}")

    if (x2-x1)>(y2-y1):
        delta_x = x2 - x1
        delta_y = y3 - y1
        delta_xy = y2-y1 
        delta_yx = x3-x1
    else:
        delta_x = x3 - x1
        delta_y = y2 - y1
        delta_xy = y3-y2
        delta_yx = x2-x1
    x_offset = x1
    y_offset = y1
    xi_scale = (delta_x/delta_yx - delta_xy/delta_y)
    yi_scale = (delta_y/delta_xy - delta_xy/delta_x)
    print(f"delta_x:{delta_x}, delta_y:{delta_y}, delta_xy:{delta_xy}, delta_yx:{delta_yx}")
    print(f"x_offset:{x_offset}, y_offset:{y_offset}")
    print(f"xi_scale:{xi_scale}, yi_scale:{yi_scale}")
    
    index = [(int(((x-x_offset+delta_x//2)/delta_yx-(y-y_offset)/delta_y) //xi_scale), int(((y-y_offset+delta_y//2)/delta_xy-(x-x_offset)/delta_x) //yi_scale)) for (y, x) in coordinates]

    index = np.array(index) 
    print(f"index:{index.shape}")
    print(f"coordinates:{coordinates.shape}")

    # print(f"index0:{index[:, 0]}")
    # print(f"index1:{index[:, 1]}")

    # Check for duplicates and report them
    duplicate_found = False
    count = 0
    for i in range(max(index[:, 0])+1):
        for j in range(max(index[:, 1])+1):
            indices = np.where((index[:, 0]==i) & (index[:, 1]==j))
            if len(indices[0]) > 1:
                print(f"Warning: Duplicate index found at ({i}, {j}) with {len(indices[0])} occurrences:{indices}")
                duplicate_found = True
            elif len(indices[0]) == 0:
                count += 1
                if count <10:
                    print(f"Warning: No index found for ({i}, {j})")

    if count > 10:
        print(f"Warning: Too many missing indices, count:{count}")


    # Create DataFrame for y coordinates
    df = pd.DataFrame({"x": index[:, 0], "y": index[:, 1], "value": coordinates[:, 0]})
    
    # Check for duplicates in the DataFrame
    duplicates = df.duplicated(subset=["x", "y"], keep=False)
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate (x, y) pairs in y coordinates")
        print("Duplicate rows:")
        print(df[duplicates])
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=["x", "y"], keep="first")
        print(f"Removed duplicates, remaining rows: {len(df)}")
    
    # Use pivot_table instead of pivot to handle any remaining edge cases
    y_coords = (
        df.pivot_table(index="x", columns="y", values="value", aggfunc="first") 
        .reindex(range(max(index[:, 0])+1))
        .reindex(columns=range(max(index[:, 1])+1))
        .fillna(0)
        .astype(float)
        .values
    ).T
    
    # Create DataFrame for x coordinates
    df = pd.DataFrame({"x": index[:, 0], "y": index[:, 1], "value": coordinates[:, 1]})
    
    # Check for duplicates in the DataFrame
    duplicates = df.duplicated(subset=["x", "y"], keep=False)
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate (x, y) pairs in x coordinates")
        print("Duplicate rows:")
        print(df[duplicates])
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=["x", "y"], keep="first")
        print(f"Removed duplicates, remaining rows: {len(df)}")
    
    # Use pivot_table instead of pivot to handle any remaining edge cases
    x_coords = (
        df.pivot_table(index="x", columns="y", values="value", aggfunc="first") 
        .reindex(range(max(index[:, 0])+1))
        .reindex(columns=range(max(index[:, 1])+1))
        .fillna(0)
        .astype(float)
        .values
    ).T
    print(f"x_coords:{x_coords.shape}")
    print(f"y_coords:{y_coords.shape}")
    xy_coords = np.dstack([x_coords, y_coords])
    print(f"xy_coords:{xy_coords.shape}")
    return xy_coords


def average_chessboard_block_length(xy_coordinates, eps=1e-6):
    """
    Mean distance between adjacent corners (one edge of a chessboard cell).

    xy_coordinates: (H, W, 2) from find_index_from_coordinates, [..., 0]=x, [..., 1]=y.
    Cells filled with (0, 0) for missing corners are skipped.
    """
    xy = np.asarray(xy_coordinates, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError("xy_coordinates must have shape (H, W, 2)")
    x, y = xy[..., 0], xy[..., 1]
    valid = (np.abs(x) > eps) & (np.abs(y) > eps)

    # Neighbors along axis 0 and axis 1 of the corner grid
    mask0 = valid[:-1, :] & valid[1:, :]
    d0 = np.linalg.norm(xy[1:, :, :] - xy[:-1, :, :], axis=2)
    mask1 = valid[:, :-1] & valid[:, 1:]
    d1 = np.linalg.norm(xy[:, 1:, :] - xy[:, :-1, :], axis=2)

    e0 = d0[mask0]
    e1 = d1[mask1]
    if e0.size == 0 and e1.size == 0:
        return np.nan, {}

    all_e = np.concatenate([e0.ravel(), e1.ravel()])
    stats = {
        "mean": float(np.mean(all_e)),
        "std": float(np.std(all_e)),
        "mean_edges_axis0": float(np.mean(e0)) if e0.size else np.nan,
        "mean_edges_axis1": float(np.mean(e1)) if e1.size else np.nan,
        "n_edges_axis0": int(e0.size),
        "n_edges_axis1": int(e1.size),
        "n_edges": int(all_e.size),
    }
    return float(np.mean(all_e)), stats


def create_coordinate_expect(xy_coordinates, w, h, grid, cellsize, mag):
    grid_size = float(grid) / cellsize *mag
    cx, cy = w/2, h/2
    distances = np.linalg.norm(xy_coordinates - np.array([cx, cy]), axis=2)
    cidx = np.unravel_index(np.argmin(distances), distances.shape)
    print(f"Nearest index to center ({cx:.2f},{cy:.2f}): {cidx}, point: {xy_coordinates[cidx]}")
    grid_x, grid_y = xy_coordinates.shape[0], xy_coordinates.shape[1]
    print(f"grid_x:{grid_x}, grid_y:{grid_y}")
    coord_expect1 = (np.arange(grid_y)-cidx[1]).reshape((grid_y, 1))@np.ones((1, grid_x))*grid_size + cx # xy_coordinates[cidx][1] #
    coord_expect2 = np.ones((grid_y, 1))@(np.arange(grid_x)-cidx[0]).reshape((1, grid_x))*grid_size + cy # xy_coordinates[cidx][0] #
    print(f"coord_expect1:{coord_expect1.shape}, from:{coord_expect1[0, 0]}, to:{coord_expect1[-1, 0]}")
    print(f"coord_expect2:{coord_expect2.shape}, from:{coord_expect2[0, 0]}, to:{coord_expect2[0, -1]}")
    coord_expect = np.dstack([coord_expect1.T, coord_expect2.T])
    print(f'coord_expect:{coord_expect.shape}')

    return coord_expect


def points_2_distortion_map(xy_coordinates, coord_expect, keywords='test'):
    # Flatten xy_coordinates and remove zero values (where either x or y is zero)
    xy_flat = xy_coordinates.reshape(-1, 2)
    nonzero_mask = np.all(xy_flat != 0, axis=1)
    xy_flat_nonzero = xy_flat[nonzero_mask]
    x_true = xy_flat_nonzero[:, 0]
    y_true = xy_flat_nonzero[:, 1]

    coord_flat = coord_expect.reshape(-1, 2)
    coord_flat_nonzero = coord_flat[nonzero_mask]
    x_expected = coord_flat_nonzero[:, 0]
    y_expected = coord_flat_nonzero[:, 1]

    # Reshape for sklearn
    X_expected = np.column_stack([x_expected, y_expected])
    true_locations = np.column_stack([x_true, y_true])

    # Fit polynomial function using least squares
    degree = 2  # You can adjust polynomial degree
    model_x = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_y = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Fit models for x and y separately
    model_x.fit(true_locations, x_expected)
    model_y.fit(true_locations, y_expected)

    poly_step = model_x.named_steps['polynomialfeatures']
    lr_step = model_x.named_steps['linearregression']

    # Get the names of the features (e.g., "x0", "x0^2")
    feature_names = poly_step.get_feature_names_out()

    # Zip them together and print
    print("Model Parameters:")
    print(f"Intercept: {lr_step.intercept_}")
    for name, coef in zip(feature_names, lr_step.coef_):
        print(f"{name}: {coef}")

    # Get fitted locations
    x_fitted = model_x.predict(true_locations)
    y_fitted = model_y.predict(true_locations)

    # Calculate residuals
    residuals_x = x_expected - x_fitted
    residuals_y = y_expected - y_fitted

    # Calculate total displacement residuals
    residuals_magnitude = np.sqrt(residuals_x**2 + residuals_y**2)

    # Print statistics
    print(f"Polynomial Degree: {degree}")
    print(f"X Residuals - Mean: {np.mean(residuals_x):.4f}, Std: {np.std(residuals_x):.4f}")
    print(f"Y Residuals - Mean: {np.mean(residuals_y):.4f}, Std: {np.std(residuals_y):.4f}")
    print(f"Total Residual Magnitude - Mean: {np.mean(residuals_magnitude):.4f}")

    # Create residual plots
    plt.figure()
    
    plt.quiver(x_fitted, y_fitted, residuals_x, residuals_y, 
                    angles='xy', scale_units='xy', scale=1, alpha=0.6, width=0.003)
    plt.scatter(x_expected, y_expected, s=5, alpha=0.5, label='Expected')
    plt.scatter(x_fitted, y_fitted, s=5, alpha=0.5, label='Fitted', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Residual Vectors (Expected → Fitted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.suptitle(f'Polynomial Fit Residual Analysis (Degree {degree})', y=1.02, fontsize=14)
    plt.show()
    plt.savefig(f'residual_analysis_{keywords}.png')
    plt.close()

    # Additional: Print model coefficients
    print("\nModel coefficients for X:")
    print(f"Intercept: {model_x.named_steps['linearregression'].intercept_:.4f}")
    print("Coefficients:", model_x.named_steps['linearregression'].coef_)

    print("\nModel coefficients for Y:")
    print(f"Intercept: {model_y.named_steps['linearregression'].intercept_:.4f}")
    print("Coefficients:", model_y.named_steps['linearregression'].coef_)
    max_error_x = np.max(np.abs(x_expected-x_fitted))
    max_error_y = np.max(np.abs(y_expected-y_fitted))
    print(f"Max error_x: {max_error_x}")
    print(f"Max error_y: {max_error_y}")

    return x_expected, y_expected, x_fitted, y_fitted, true_locations, residuals_x, residuals_y

def residual_2_remap_rbf(x_expected, y_expected, residuals_x, w, h):
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    rbf = Rbf(x_expected, y_expected, residuals_x, kind='thin_plate')
    interp_residuals = rbf(grid_x.ravel(), grid_y.ravel()).reshape(h, w)
    if np.isnan(interp_residuals).any():
        mask = np.isnan(interp_residuals)
        interp_residuals[mask] = Rbf(x_expected[~mask], y_expected[~mask], residuals_x[~mask], kind='nearest')(grid_x[mask], grid_y[mask])
    return interp_residuals

def residual_2_remap(x_expected, y_expected, residuals_x, w, h):

    points = np.vstack((x_expected, y_expected)).T
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    interpolated_grid = griddata(points, residuals_x, (grid_x, grid_y), method='linear')
    
    if np.isnan(interpolated_grid).any():
        mask = np.isnan(interpolated_grid)
        interpolated_grid[mask] = griddata(points, residuals_x, (grid_x[mask], grid_y[mask]), method='nearest')
        
    return interpolated_grid

def create_distortion_map(image_path, grid, cellsize, mag, keywords='test'):
    coords = detect_and_save_saddles(image_path, sigma=2.9, output_csv=f"saddle_points_{keywords}.csv")
    xy_coordinates = find_index_from_coordinates(coords)
    print(f"xy_coordinates:{xy_coordinates.shape}")
    avg_len, blk_stats = average_chessboard_block_length(xy_coordinates)
    print(
        f"Average chessboard block edge length (px): {avg_len:.4f} "
        f"(std={blk_stats.get('std', float('nan')):.4f}, "
        f"n_edges={blk_stats.get('n_edges', 0)}, "
        f"axis0_mean={blk_stats.get('mean_edges_axis0', float('nan')):.4f}, "
        f"axis1_mean={blk_stats.get('mean_edges_axis1', float('nan')):.4f})"
    )

    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    h, w = original_gray.shape
    original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
    for i in range(xy_coordinates.shape[0]):
        for j in range(xy_coordinates.shape[1]):
            x = xy_coordinates[i, j, 0]
            y = xy_coordinates[i, j, 1]
            cv2.circle(original_rgb, (int(x), int(y)), 20, (255, 0, 255), -1)
            cv2.putText(original_rgb, f"{i},{j}", (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    plt.figure()
    plt.imshow(original_rgb)
    plt.title(f'original_rgb_points_{keywords}.png')
    plt.show()
    plt.savefig(f'original_rgb_points_{keywords}.png')
    plt.close()
    print(f'w:{w}, h:{h}, grid:{grid}, cellsize:{cellsize}, mag:{mag}')
    print(f'xy_coordinates:{xy_coordinates.shape}')
    
    coord_expect = create_coordinate_expect(xy_coordinates, w, h, grid, cellsize, mag)
    print(f"coord_expect:{coord_expect.shape}")
    for i in range(coord_expect.shape[1]):
        for j in range(coord_expect.shape[0]):
            x = coord_expect[j, i, 0]
            y = coord_expect[j, i, 1]
            cv2.circle(original_rgb, (int(x), int(y)), 13, (255, 0, 0), -1)
            cv2.putText(original_rgb, f"{i},{j}", (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    plt.figure()
    plt.imshow(original_rgb)
    plt.title(f'original_rgb_coord_expect_{keywords}.png')
    plt.show()
    plt.savefig(f'original_rgb_coord_expect_{keywords}.png')
    plt.close()
    x_expected, y_expected, x_fitted, y_fitted, true_locations, residuals_x, residuals_y = points_2_distortion_map(xy_coordinates, coord_expect, keywords)

    interpolated_grid_x = residual_2_remap(x_expected, y_expected, residuals_x, w, h)
    interpolated_grid_y = residual_2_remap(x_expected, y_expected, residuals_y, w, h)
    # interpolated_grid_x_rbf = residual_2_remap_rbf(true_locations[:, 0], true_locations[:, 1], residuals_x, w, h)
    # interpolated_grid_y_rbf = residual_2_remap_rbf(true_locations[:, 0], true_locations[:, 1], residuals_y, w, h)
    
    np.save(f'distortion_map_x_{keywords}.npy', interpolated_grid_x)
    np.save(f'distortion_map_y_{keywords}.npy', interpolated_grid_y)

    print(f'interpolated_grid shape: {interpolated_grid_x.shape}')
    print(f'interpolated_grid_x: {np.max(interpolated_grid_x), np.min(interpolated_grid_x)}')
    print(f'interpolated_grid_y: {np.max(interpolated_grid_y), np.min(interpolated_grid_y)}')

    X, Y = np.meshgrid(np.arange(h), np.arange(w))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, interpolated_grid_x, cmap='viridis')
    ax.set_title('remaps (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Remap Value')
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.imshow(interpolated_grid_x)
    plt.title('interpolated_grid_x')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(interpolated_grid_y)
    plt.title('interpolated_grid_y')
    plt.axis('off')
    plt.show()

    img2distor = './chessboard/Image__2025-11-17__18-17-33.bmp'
    # img2distor = './20230321_distor/1200_1200.bmp'
    img2distor = cv2.imread(img2distor, cv2.IMREAD_GRAYSCALE)

    Y, X = np.indices((w, h), dtype=np.float32)
    # Calculate absolute map coordinates
    map_x = X + interpolated_grid_x.astype(np.float32)
    map_y = Y + interpolated_grid_y.astype(np.float32)

    remapped_image = cv2.remap(img2distor, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'remapped_image_{keywords}.png', remapped_image)
    return 0

def __main__():
    folder_path = './'
    
    grid = 100.0
    cellsize = 3.2
    mag = 10 # 5

    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            # image_path = os.path.join(folder_path, file)
            # image_path = './20230321_distor/1200_1200.png'
            
            image_path = './Landing_point_detection_offline_testdata/20260320-zj-noPZT/100um.bmp'
            flag = create_distortion_map(image_path, grid, cellsize, mag, str(file)[:-4])
            if flag == 0:
                print(f"Processing image: {image_path} success")
            else:
                print(f"Processing image: {image_path} failed")
            
            break
    
    plt.show()

    

if __name__ == "__main__":
    __main__()