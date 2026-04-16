import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter


img_name = './20260330_215619-bien/20260330_215810_652.png'
img = (cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]).astype(np.float32)
rows, cols = img.shape

center_x, center_y = 678, 533

# Extract ALL values on the line through (center_x, center_y) at angle_deg.
# Angle convention: angle_deg is measured from +x (to the right), counter-clockwise.
# Since image y increases downward, use dy = -sin(theta).
angle_deg = -35
theta = np.deg2rad(angle_deg)
dx = float(np.cos(theta))
dy = float(-np.sin(theta))

cx, cy = float(center_x), float(center_y)

def _t_range_for_bounds(c, d, lo, hi):
    if abs(d) < 1e-12:
        return (-np.inf, np.inf) if (lo <= c <= hi) else (np.inf, -np.inf)
    t0 = (lo - c) / d
    t1 = (hi - c) / d
    return (min(t0, t1), max(t0, t1))

tx0, tx1 = _t_range_for_bounds(cx, dx, 0.0, cols - 1.0)
ty0, ty1 = _t_range_for_bounds(cy, dy, 0.0, rows - 1.0)
t0 = max(tx0, ty0)
t1 = min(tx1, ty1)

x0, y0 = cx + t0 * dx, cy + t0 * dy
x1, y1 = cx + t1 * dx, cy + t1 * dy

n = int(np.hypot(x1 - x0, y1 - y0)) + 1
xs = np.round(np.linspace(x0, x1, n)).astype(int)
ys = np.round(np.linspace(y0, y1, n)).astype(int)

mask = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
xs, ys = xs[mask], ys[mask]

line_value = img[ys, xs]
plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.plot(line_value)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 假设 gray_data 是你从图中提取的一维灰度数组
gray_data = line_value

# 1. 减去直流分量（平滑掉那个“大拱门”背景）
# 这里可以使用高斯滤波提取背景，或者减去均值
background = np.convolve(gray_data, np.ones(50)/50, mode='same')
signal_centered = gray_data - background

# 2. 进行 Hilbert 变换
analytic_signal = hilbert(signal_centered)
amplitude_envelope = np.abs(analytic_signal) # 提取振幅包络

# 3. 计算上下包络
upper_envelope = background + amplitude_envelope
lower_envelope = background - amplitude_envelope

plt.plot(gray_data, label='Original')
plt.plot(upper_envelope, 'r--', label='Upper Envelope')
plt.plot(lower_envelope, 'g--', label='Lower Envelope')
plt.legend()
plt.show()

import pandas as pd

# 使用 pandas 的滚动窗口（rolling）
# window_size 需要根据你图中条纹的宽度来定（例如中心处约 40-60 像素）
df = pd.Series(gray_data)
upper_env = df.rolling(window=50, center=True).max()
lower_env = df.rolling(window=50, center=True).min()

# 为了平滑折线，可以再做一次平滑
upper_env_smooth = upper_env.rolling(window=20, center=True).mean()
lower_env_smooth = lower_env.rolling(window=20, center=True).mean()

plt.figure()
plt.plot(gray_data, label='Original')
plt.plot(upper_env_smooth, 'r--', label='smooth Upper Envelope')
plt.plot(lower_env_smooth, 'g--', label='smooth Lower Envelope')
plt.legend()
plt.show()

