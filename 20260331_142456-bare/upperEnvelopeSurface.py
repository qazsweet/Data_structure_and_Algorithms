import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

# 1. 读取 2D 干涉图像 (假设已经是灰度图)
img_name = './20260330_215619/20260330_215810_652.png'
img = (cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]).astype(np.float32)
rows, cols = img.shape
# 这里构造一个模拟的牛顿环图像作为演示
# rows, cols = 512, 512
# x = np.linspace(-10, 10, cols)
# y = np.linspace(-10, 10, rows)
# X, Y = np.meshgrid(x, y)
# R2 = X**2 + Y**2
# # 构造背景 (高斯分布) + 振幅衰减 (随半径) + 干涉条纹
# background_true = 150 * np.exp(-R2/50) + 50
# amplitude_true = 80 * np.exp(-R2/30)
# phase = R2 * 2  # 变频条纹
# img = background_true + amplitude_true * np.cos(phase) + np.random.normal(0, 5, R2.shape) # 添加噪声

# ==========================================
# 2D 包络提取核心步骤
# ==========================================

# 步骤 A: 提取 2D 背景 (直流分量)
# 使用超大核的高斯滤波，窗口要远大于最粗的条纹间距
bg_estimated = gaussian_filter(img, sigma=100) 

# 步骤 B: 去除背景，得到零均值的干涉信号
img_ac = img - bg_estimated

# 步骤 C: 计算 2D 振幅包络 (调制幅度)
# 简单有效的方法：对 AC 信号取绝对值，然后再次进行低通滤波
# 滤波器的 sigma 应该接近条纹的平均周期
modulation_depth = gaussian_filter(np.abs(img_ac), sigma=20)
# 修正能量损失 (因为取绝对值后均值变为 $2A/\pi$)
modulation_depth *= np.pi / 2

# 步骤 D: 生成上下包络面
upper_envelope_2d = bg_estimated + modulation_depth
lower_envelope_2d = bg_estimated - modulation_depth

# ==========================================
# 结果可视化
# ==========================================
plt.figure(figsize=(12, 8))

plt.subplot(231); plt.imshow(img, cmap='gray'); plt.title('Original Interferogram')
plt.subplot(232); plt.imshow(bg_estimated, cmap='gray'); plt.title('Estimated Background $B(x,y)$')
plt.subplot(233); plt.imshow(modulation_depth, cmap='jet'); plt.title('Amplitude Envelope $M(x,y)$')

plt.subplot(234); plt.imshow(upper_envelope_2d, cmap='gray'); plt.title('Upper Envelope Surface')
plt.subplot(235); plt.imshow(lower_envelope_2d, cmap='gray'); plt.title('Lower Envelope Surface')

# 绘制一条过中心的剖面线进行验证
center_row = rows // 2
plt.subplot(236)
plt.plot(img[center_row, :], 'k', alpha=0.5, label='Original data')
plt.plot(upper_envelope_2d[center_row, :], 'r--', label='Upper Env')
plt.plot(lower_envelope_2d[center_row, :], 'g--', label='Lower Env')
plt.plot(bg_estimated[center_row, :], 'b', label='Background')
plt.title('Center Row Profile Verification')
plt.legend(fontsize='small')

plt.tight_layout()
plt.show()