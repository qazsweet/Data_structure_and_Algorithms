import numpy as np
import cv2
from skimage import morphology, measure
from scipy.interpolate import interp1d

# 假设 H, W 是图像的高度和宽度
H, W = skeleton.shape

## 5.1. 剔除照明边缘效应
margin = 15
# 创建边缘掩码：边缘区域为 True，内部为 False
border_mask = np.ones((H, W), dtype=bool)
border_mask[margin:H-margin, margin:W-margin] = False

# 连通域分析 (相当于 bwconncomp)
labels = measure.label(skeleton)
regions = measure.regionprops(labels)

skeleton_cleaned = skeleton.copy()
for reg in regions:
    # 获取当前连通域的所有像素坐标
    coords = reg.coords # 格式为 [[r1, c1], [r2, c2], ...]
    # 如果任何一个像素落在边缘掩码内，整条抹除
    if np.any(border_mask[coords[:, 0], coords[:, 1]]):
        skeleton_cleaned[coords[:, 0], coords[:, 1]] = 0

# 去除微小毛刺 (相当于 bwmorph spur)
# skimage 没有直接的 spur 10 次迭代，通常使用 thin 或简单的 remove_small_objects
skeleton_cleaned = morphology.thin(skeleton_cleaned)

## 5.2 骨架拓扑分析与二次曲线插补
# 重新获取清理后的连通域
labels = measure.label(skeleton_cleaned)
regions = measure.regionprops(labels)
skeleton_closed = np.zeros((H, W), dtype=bool)

for reg in regions:
    branch_mask = (labels == reg.label)
    
    # 寻找端点 (使用特殊的卷积核寻找只有一个邻居的像素)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    neighbor_count = cv2.filter2D(branch_mask.astype(np.uint8), -1, kernel)
    ey, ex = np.where((neighbor_count == 11) & branch_mask) # 10(中心) + 1(邻居)
    
    branch_length = reg.area
    
    if len(ex) == 0:
        # 情况 A：闭合环
        skeleton_closed |= branch_mask
        
    elif len(ex) == 2:
        # 情况 B：开口线段
        ep_dist = np.linalg.norm([ex[0]-ex[1], ey[0]-ey[1]])
        
        # 滤除规则
        if ep_dist < branch_length * 0.7 and branch_length > 10:
            by, bx = np.where(branch_mask)
            
            # 坐标系旋转以防止拟合斜率无穷大
            theta = np.arctan2(ey[1]-ey[0], ex[1]-ex[0])
            rx = bx * np.cos(theta) + by * np.sin(theta)
            ry = -bx * np.sin(theta) + by * np.cos(theta)
            
            # 拟合二次曲线 (y = ax^2 + bx + c)
            p = np.polyfit(rx, ry, 2)
            
            # 在端点间插值
            rex1 = ex[0]*np.cos(theta) + ey[0]*np.sin(theta)
            rex2 = ex[1]*np.cos(theta) + ey[1]*np.sin(theta)
            step = 0.5 if rex2 > rex1 else -0.5
            rx_interp = np.arange(rex1, rex2, step)
            ry_interp = np.polyval(p, rx_interp)
            
            # 逆旋转回原系
            x_i = np.round(rx_interp * np.cos(-theta) + ry_interp * np.sin(-theta)).astype(int)
            y_i = np.round(-rx_interp * np.sin(-theta) + ry_interp * np.cos(-theta)).astype(int)
            
            # 越界检查并写入
            valid = (x_i >= 0) & (x_i < W) & (y_i >= 0) & (y_i < H)
            branch_mask[y_i[valid], x_i[valid]] = True
            
            skeleton_closed |= branch_mask

# 再次细化确保单像素宽度
skeleton_closed = morphology.thin(skeleton_closed)

## 5.3 独立提取面积与面心
from scipy.ndimage import binary_fill_holes
labels_final = measure.label(skeleton_closed)
regions_final = measure.regionprops(labels_final)
contour_list = []

for reg in regions_final:
    temp_skel = (labels_final == reg.label)
    # 填充孔洞 (相当于 imfill holes)
    filled_contour = binary_fill_holes(temp_skel)
    
    area_skel = reg.area
    area_filled = np.sum(filled_contour)
    
    if area_filled > area_skel * 1.5:
        # 记录中心和面积
        contour_list.append({
            'area': area_filled,
            'centroid': reg.centroid, # 注意：skimage 返回 (row, col) 即 (y, x)
            'coords': reg.coords
        })

if not contour_list:
    raise ValueError("未检测到任何有效封闭的等高线！")

## 5.4 排序与极值输出
# 按面积从小到大排序
contour_sorted = sorted(contour_list, key=lambda x: x['area'])

# 提取极值点 (最小环的中心)
extremum_y, extremum_x = contour_sorted[0]['centroid']
print(f"检测到干涉极值点坐标: ({extremum_x:.2f}, {extremum_y:.2f})")

# 初始化全场高度图 (与原图等大)
height_map = np.full((H, W), np.nan)

# 分配级次：面积越小的环，级次 m 越小 (假设中心是极值点)
for m, contour in enumerate(contour_sorted):
    # m = 0, 1, 2... 代表第几个条纹级次
    # 物理高度 h = m * (lambda / 2)
    h_val = m * (lambda_val / 2.0)
    
    # 将该等高线上的所有像素点赋予相同的高度
    coords = contour['coords']
    height_map[coords[:, 0], coords[:, 1]] = h_val

## 6. 最终曲面插值 (基于修复后的骨架)
# 提取所有非 NaN 的点作为插值样本
y_idx, x_idx = np.where(~np.isnan(height_map))
z_values = height_map[y_idx, x_idx]

# 转换为物理坐标 (mm)
points_phys = np.column_stack((x_idx * pixel_pitch, y_idx * pixel_pitch))

# 生成目标插值网格 (根据之前定义的 grid_res)
grid_x, grid_y = np.meshgrid(
    np.arange(0, W * pixel_pitch, grid_res),
    np.arange(0, H * pixel_pitch, grid_res)
)

# 使用 RBF (径向基函数) 插值，比 cubic 在处理离散环带时更平滑
from scipy.interpolate import Rbf
rbf = Rbf(points_phys[:, 0], points_phys[:, 1], z_values, function='multiquadric')
z_final = rbf(grid_x, grid_y)

## 7. 可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# 左图：修复后的骨架与级次中心
ax1 = fig.add_subplot(121)
ax1.imshow(skeleton_closed, cmap='gray')
ax1.scatter(extremum_x, extremum_y, color='red', label='Extremum')
ax1.set_title("Optimized Skeleton & Center")
ax1.legend()

# 右图：3D 重建面型
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(grid_x, grid_y, z_final, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
ax2.set_title("3D Surface Reconstruction (mm)")
plt.show()

