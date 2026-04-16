import numpy as np
import os
import cv2
from collections import deque
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class DynamicPSIProcessor:
    def __init__(self, height, width):
        # 使用 deque 维持一个长度为 5 的滑动窗口
        self.window = deque(maxlen=2)
        self.height = height
        self.width = width

    def process_frame(self, new_frame):
        """
        每输入一帧新图像，尝试计算一次相位
        new_frame: 2D numpy array (uint8 or float)
        """
        # self.window.append(new_frame.astype(np.float32))
        self.window.appendleft(new_frame.astype(np.float32))
        
        # 只有当窗口填满 5 张图时才开始计算
        if len(self.window) < 2:
            return None, None, None, None, None, None, None
        
        # 将 deque 转换为 3D array 方便向量化计算 [5, H, W]
        imgs = np.array(self.window)
        
        diff = (imgs[1].astype(np.float32) - imgs[0].astype(np.float32))
        diff_min = diff.min()
        diff_max = diff.max()
        print(f"diff: {diff_min:.2f}, {diff_max:.2f}")

        fourier_diff = np.fft.fft2(diff)
        high_freq_power = np.sum(np.abs(fourier_diff))
        print(f"high_freq_power: {high_freq_power:.2f}")

        ssim_value = ssim(imgs[0].astype(np.uint8), imgs[1].astype(np.uint8))
        psnr_value = psnr(imgs[0].astype(np.uint8), imgs[1].astype(np.uint8))
        print(f"ssim: {ssim_value:.2f}, psnr: {psnr_value:.2f}")
        
        return diff, fourier_diff, diff_min, diff_max, high_freq_power, ssim_value, psnr_value

# --- 模拟算法运行 ---

# 假设图像尺寸
H, W = 512, 512
processor = DynamicPSIProcessor(H, W)

hfq_list = []
x_list = []
diff_min_list = []
diff_max_list = []
high_freq_power_list = []
ssim_list = []
psnr_list = []
# for i in range(200, 211):
#     img_name = './../20260311ya/list4/Basler_acA4112-20um__40713375__20260312_211722080_'+ str(format(i, '04d')) + '.bmp'
# for i in range(190, 203):

# image_folder = './first_frame/2'
# image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tiff'))])
# for idx, img_file in enumerate(image_files):
#     img_name = os.path.join(image_folder, img_file)
#     img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
#     # Extract last 4 consecutive digits before the file extension for i
#     import re
#     basename, _ = os.path.splitext(img_file)
#     match = re.search(r'(\d{4})(?!.*\d)', basename)  # get last 4 digit group before extension
#     if match:
#         i = int(match.group(1))
#     else:
#         i = idx  # fallback if not found

image_folder = './20260331_142456' # '../20260311ya/20260330_215619'
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
for idx, img_file in enumerate(image_files):

    img_name = os.path.join(image_folder, img_file)
    base_filename = os.path.splitext(os.path.basename(img_name))[0]
    i = int(base_filename[-3:])
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]

    fourier_img = np.fft.fft2(img)
    high_freq_power1 = np.sum(np.abs(fourier_img))
    x_list.append(idx)
    hfq_list.append(high_freq_power1)
    print(f"{i} high_freq_power: {high_freq_power1:.2f}")
    
    # To visualize the magnitude spectrum of the Fourier transform, do a shift and take the log of the absolute value
    fshift = np.fft.fftshift(fourier_img)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 取 magnitude_spectrum > 200 的部分，其余赋值为 0，然后能否转回时域
    mag_mask = magnitude_spectrum > 200
    # 构造与 fshift 相同大小的 mask, 用于频谱操作（mask 为 True 的点保留，否则为 0）
    filtered_fshift = np.zeros_like(fshift, dtype=complex)
    filtered_fshift[mag_mask] = fshift[mag_mask]

    # 逆变换回时域图像
    f_ishift = np.fft.ifftshift(filtered_fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    wrapped_phase, fourier_diff, diff_min, diff_max, high_freq_power, ssim_value, psnr_value = processor.process_frame(img)
    
    if wrapped_phase is not None:
        diff_min_list.append(diff_min)
        diff_max_list.append(diff_max)
        high_freq_power_list.append(high_freq_power)
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        print(f"成功处理第 {i+1} 帧，生成相位图。")
        # 此处得到的 wrapped_phase 是截断相位，通常需要进一步做相位解包裹 (Unwrapping)
        # 例如使用 cv2.normalize 映射到 0-255 进行预览
        display = cv2.normalize(wrapped_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        plt.figure()
        plt.title(f"Frame {i}")
        plt.subplot(141)
        plt.imshow(img, cmap='gray')
        plt.subplot(142)
        plt.imshow(display, cmap='gray')
        plt.subplot(143)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.subplot(144)
        plt.imshow(img_back, cmap='gray')
        plt.show()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x_list, hfq_list)
plt.plot(x_list[1:], high_freq_power_list)
plt.title('High Frequency Power')
plt.subplot(2, 2, 2)
plt.plot(x_list[1:], diff_min_list)
plt.plot(x_list[1:], diff_max_list)
plt.title('Diff Min and Max')
plt.subplot(2, 2, 3)
plt.plot(x_list[1:], ssim_list)
plt.title('SSIM')
plt.subplot(2, 2, 4)
plt.plot(x_list[1:], psnr_list)
plt.title('PSNR')
plt.show()