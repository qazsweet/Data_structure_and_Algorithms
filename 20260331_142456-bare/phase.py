import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage import morphology, filters, feature

# =========================================================================
# 基于条纹骨架法的干涉图样相位解析与面型重建
# =========================================================================

## 0. 参数设置 (Parameter Initialization)
lambda_val = 460e-6  # 光源波长 (mm)
grid_res = 0.5       # 最终面型空间分辨率 (mm)
pixel_pitch = 0.01   # 每个像素代表的实际物理尺寸 (mm)

## 1. 图形读取与预处理 (Image Reading) 
file_path = './20260330_215619/20260330_215648_065.png'#  r'./list4/Basler_acA4112-20um__40713375__20260312_211722080_0211.bmp' # 请确保路径正确
file_path = './20260330_215619/20260330_215719_341.png'#  r'./list4/Basler_acA4112-20um__40713375__20260312_211722080_0211.bmp' # 请确保路径正确
img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100] # [1000:2200, 1900:3140]

if img_gray is None:
    raise FileNotFoundError(f"无法读取文件: {file_path}")

img_double = img_gray.astype(float) / 255.0

## 3. 各向异性扩散滤波 (Anisotropic Diffusion Filtering)
# 3.1 高斯预滤波
sigma_val = 2.0
img_gaussian = filters.gaussian(img_double, sigma=sigma_val)

# # 3.2 各向异性扩散 (使用 skimage 的 denoise_nl_means 或 simple diffusion)
# # Python 中常用的各向异性扩散可以使用 medfilt 或自定义，这里推荐 skimage 的 rolling_ball 或简单的 nlm
# from skimage.restoration import denoise_tv_chambolle
# img_filtered = denoise_tv_chambolle(img_gaussian, weight=0.1) # TV去噪与扩散效果类似


# from skimage.restoration import denoise_nl_means, estimate_sigma

# # 估计噪声标准差
# sigma_est = np.mean(estimate_sigma(img_gaussian))
# img_filtered = denoise_nl_means(img_gaussian, h=0.8 * sigma_est, fast_mode=True)


# plt.figure()
# plt.imshow(img_filtered, cmap='gray')
# plt.show()
img_filtered = img_gaussian

## 3. 背景归一化 (Background Normalization)
se_size = 50
se = morphology.disk(se_size)
I_max = morphology.dilation(img_filtered, se)
I_min = morphology.erosion(img_filtered, se)

denominator = I_max - I_min
denominator[denominator == 0] = 1e-12 # 避免除以0

# 归一化到 [-1, 1]
img_norm = 2 * (img_filtered - I_min) / denominator - 1

## 4. 条纹极值提取（骨架提取）
threshold = 0
BW = img_norm < threshold

# 去除孤立噪点 (Remove small objects)
BW_cleaned = morphology.remove_small_objects(BW, min_size=50)

# 形态学骨架细化 (Thinning)
skeleton = morphology.skeletonize(BW_cleaned)

plt.figure("Step 4: Skeleton")
plt.imshow(skeleton, cmap='gray')
plt.title("Fringe Skeleton")
plt.show(block=False)

## 5. 级次分配 (Fringe Order Assignment)
print("请在图像中点击同心圆环的最中心点...")
plt.figure("Click Center")
plt.imshow(skeleton, cmap='gray')
pts = plt.ginput(1, timeout=0) # 0 表示不超时
plt.close()

if not pts:
    raise ValueError("未点击中心点")
x_center, y_center = pts[0]

# 获取骨架像素坐标
row_skel, col_skel = np.where(skeleton)

# 计算物理距离
dist_to_center = np.sqrt((col_skel - x_center)**2 + (row_skel - y_center)**2) * pixel_pitch

# 简单的距离分层法分配级次
max_dist = np.max(dist_to_center)
num_bins = 50
bins = np.linspace(0, max_dist, num_bins)
bin_idx = np.digitize(dist_to_center, bins) - 1

# 计算高度 z = m * lambda / 2
z_skel = bin_idx * (lambda_val / 2.0)

## 6. 全场相位解包与曲面插值 (Surface Interpolation)
# 准备散点数据
points = np.column_stack((col_skel * pixel_pitch, row_skel * pixel_pitch))

# 生成目标网格
h, w = img_double.shape
x_phys = np.arange(0, w) * pixel_pitch
y_phys = np.arange(0, h) * pixel_pitch
grid_x, grid_y = np.meshgrid(
    np.arange(x_phys.min(), x_phys.max(), grid_res),
    np.arange(y_phys.min(), y_phys.max(), grid_res)
)

# 使用 griddata 进行插值 (cubic 对应 natural)
z_surface = griddata(points, z_skel, (grid_x, grid_y), method='cubic')

## 7. 最终结果输出 (Output)
fig = plt.figure("Step 7: 3D Surface Reconstruction")
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(grid_x, grid_y, z_surface, cmap='jet', edgecolor='none')
fig.colorbar(surf)
ax.set_title(f"Reconstructed Surface (Res: {grid_res} mm)")
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Height (mm)')
plt.show()