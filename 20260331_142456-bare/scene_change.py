import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.filters import sato, threshold_otsu
from skimage.morphology import skeletonize
from scipy import fft
import cv2
import os
import time

 
def find_fringe_patterns(img, gamma=5, threshold_factor=1.0, blur_size=(9, 9), blur_sigma=2, sato_sigmas=range(1, 6, 1)):
    """
    threshold_factor: < 1.0 finds more edges (e.g. 0.7)
    blur_size/blur_sigma: smaller = less smoothing = more edges
    sato_sigmas: wider range = more ridge scales
    """
    gray = (((img / 255) ** gamma) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, blur_size, blur_sigma)
    ridges = sato(blurred, sigmas=sato_sigmas, black_ridges=True)
    thresh = threshold_otsu(ridges)
    binary_ridges = ridges > (thresh * threshold_factor)  # Lower factor = more edges
    skeleton = skeletonize(binary_ridges)
    return skeleton

def fit_fringe_ellipses(skeleton_image, original_img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton_image.astype(np.uint8))
    
    output_img = original_img.copy()
    results = []

    # 遍历每一个检测到的条纹（跳过背景 0）
    for i in range(1, num_labels):
        # 提取当前条纹的所有像素点坐标
        points = np.column_stack(np.where(labels == i))
        
        # 过滤过小的杂质点（比如少于 100 个像素的线条）
        if len(points) < 100:
            continue
            
        # OpenCV 的 fitEllipse 需要 (x, y) 格式，points 目前是 (row, col)
        points_xy = np.fliplr(points).reshape(-1, 1, 2).astype(np.float32)

        try:
            # 2. 执行数学拟合
            # 返回值: ((x_center, y_center), (short_axis, long_axis), angle)
            ellipse = cv2.fitEllipse(points_xy)
            
            # 将结果存入列表
            results.append({
                "id": i,
                "center": ellipse[0],
                "axes": ellipse[1],
                "angle": ellipse[2]
            })

            # 3. 绘制拟合结果
            cv2.ellipse(output_img, ellipse, (0, 255, 0), 2)
        except cv2.error:
            continue

    return output_img, results


def find_smallest_center_curve_and_mean(skeleton, original_img):
    """
    找到skeleton中闭合曲线最中心最小的一个，并求这条曲线内像素的均值。
    
    Returns:
        mean_val: 曲线内像素均值
        contour: 选中的闭合轮廓
        mask: 曲线内部的二值掩码
    """
    # skeleton 可能是 boolean，转为 uint8
    skel_uint8 = (skeleton.astype(np.uint8) * 255) if skeleton.dtype == bool else skeleton.astype(np.uint8)
    
    # 骨架线很细，先轻微膨胀以便形成闭合轮廓
    kernel = np.ones((3, 3), np.uint8)
    skel_dilated = cv2.dilate(skel_uint8, kernel)
    
    # 使用 RETR_CCOMP 获取外层轮廓和孔洞，孔洞对应曲线内部区域
    contours, hierarchy = cv2.findContours(
        skel_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    
    h, w = original_img.shape[:2]
    img_center = np.array([w / 2, h / 2])
    
    candidates = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 30:  # 过滤过小的噪声
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = np.array([cx, cy])
        dist_to_center = np.linalg.norm(centroid - img_center)
        candidates.append({
            "contour": cnt,
            "area": area,
            "dist_to_center": dist_to_center,
            "centroid": centroid,
        })
    
    if not candidates:
        raise ValueError("未找到有效的闭合曲线")
    
    # 在中心区域内取面积最小的（最中心最小的闭合曲线）
    center_radius = min(w, h) * 0.4
    center_candidates = [c for c in candidates if c["dist_to_center"] < center_radius]
    if not center_candidates:
        center_candidates = candidates
    
    best = min(center_candidates, key=lambda c: (c["dist_to_center"], c["area"]))
    contour = best["contour"]
    
    # 创建曲线内部的掩码（填充轮廓内部）
    mask = np.zeros_like(original_img, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    
    # 计算曲线内像素均值（使用原图 img1）
    pixels_inside = original_img[mask > 0]
    if len(pixels_inside) == 0:
        raise ValueError("曲线内部无有效像素")
    mean_val = float(np.mean(pixels_inside))
    
    return mean_val, contour, mask




def fourier_denoise_2d(image, low_pass_cutoff=0.1):
    """
    使用二位傅里叶变换进行低通滤波去噪
    :param image: 输入的2D灰度图像 (numpy array)
    :param low_pass_cutoff: 截止频率比例 (0.0 到 1.0)
    """
    # 1. 执行二维傅里叶变换
    f_coeff = fft.fft2(image)
    
    # 2. 将零频分量移到频谱中心，方便处理
    f_shift = fft.fftshift(f_coeff)
    
    # 3. 创建掩模 (Mask) - 简单的圆形低通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # 中心位置
    
    # 生成坐标网格
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # 定义截止半径 (以图像尺寸的一定比例)
    radius = low_pass_cutoff * (min(rows, cols) / 2)
    mask = dist_from_center <= radius
    
    # 4. 应用掩模滤波
    f_shift_filtered = f_shift * mask
    
    # 5. 逆向移频并执行逆傅里叶变换
    f_ishift = fft.ifftshift(f_shift_filtered)
    image_denoised = fft.ifft2(f_ishift)
    
    # 返回实部
    return np.abs(image_denoised), np.log(1 + np.abs(f_shift))

def main():
    gap_point_dir = "./gap_point"
    subfolders = [os.path.join(gap_point_dir, name) for name in os.listdir(gap_point_dir) if os.path.isdir(os.path.join(gap_point_dir, name))]
    img1 = None
    img2 = None
    # 遍历每个子文件夹，只取前两张图片（名字排序后）
    for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        images.sort()
        if len(images) >= 2:
            imgname1 = os.path.join(subfolder, images[0])
            imgname2 = os.path.join(subfolder, images[1])
            imgname3 = os.path.join(subfolder, images[2])
            img1 = cv2.imread(imgname1, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
            img2 = cv2.imread(imgname2, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
            img3 = cv2.imread(imgname3, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
            w, h = img1.shape

            mean_val_temp1 = np.mean(img1)
            min_val_temp1 = np.min(img1)
            max_val_temp1 = np.max(img1)

            mean_val_temp2 = np.mean(img2)
            min_val_temp2 = np.min(img2)
            max_val_temp2 = np.max(img2)

            mean_val_temp3 = np.mean(img3)
            min_val_temp3 = np.min(img3)
            max_val_temp3 = np.max(img3)
            print(f"statistics of img1: , {mean_val_temp1:.2f}, {min_val_temp1:.2f}, {max_val_temp1:.2f}, statistics of img2: , {mean_val_temp2:.2f}, {min_val_temp2:.2f}, {max_val_temp2:.2f}, statistics of img3: , {mean_val_temp3:.2f}, {min_val_temp3:.2f}, {max_val_temp3:.2f}.")
            
            denoised1, spectrum1 = fourier_denoise_2d(img1, low_pass_cutoff=0.15)
            denoised2, spectrum2 = fourier_denoise_2d(img2, low_pass_cutoff=0.15)

            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(img1, cmap='gray')
            plt.subplot(2, 3, 2)
            plt.imshow(img2, cmap='gray')
            plt.subplot(2, 3, 3)
            plt.imshow(spectrum1, cmap='gray')
            plt.subplot(2, 3, 4)
            plt.imshow(img2-img1, cmap='gray')
            plt.subplot(2, 3, 5)
            plt.imshow(denoised2-denoised1, cmap='gray')
            plt.subplot(2, 3, 6)
            plt.imshow(spectrum2-spectrum1, cmap='gray')
            plt.show()
            
            # start = time.time()
            # img1 = cv2.resize(img1, (int(w*0.25), int(h*0.25)))
            # gray = (((img1 / 255) ** 3) * 255).astype(np.uint8)
            # skeleton1 = find_fringe_patterns(gray, gamma=1, threshold_factor=0.6, blur_size=(5, 5))
            # mean_val1, contour1, mask1 = find_smallest_center_curve_and_mean(skeleton1, img1)

            # end = time.time()
            # print(f"Time taken for img1: {end - start} seconds")

            # img2 = cv2.resize(img2, (int(w*0.25), int(h*0.25)))

            # # img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            # # img2 = cv2.GaussianBlur(img2, (5, 5), 0)

            # skeleton2 = find_fringe_patterns(img2, gamma=3, threshold_factor=0.6, blur_size=(5, 5))
            # mean_val2, contour2, mask2 = find_smallest_center_curve_and_mean(skeleton2, img2)
            # print(mean_val1, mean_val2)

                
            # fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            # axes[0,0].imshow(gray, cmap='gray')
            # axes[0,0].set_title('Original Image')
            
            # axes[0,1].imshow(skeleton1, cmap='inferno')
            # axes[0,1].set_title('Extracted Ridge Skeleton')

            # axes[0,2].imshow(mask1, cmap='gray')
            # axes[0,2].set_title('Extracted mask1')
            
            # axes[1,0].imshow(img2, cmap='gray')
            # axes[1,0].set_title('Original Image')
            
            # axes[1,1].imshow(skeleton2, cmap='inferno')
            # axes[1,1].set_title('Extracted Ridge Skeleton')
            
            # axes[1,2].imshow(mask2, cmap='gray')
            # axes[1,2].set_title('Extracted mask2')

            # plt.tight_layout()
            # plt.show()
            #  output_img2, results2 = fit_fringe_ellipses(skeleton2, img2)
            #  plt.subplot(1, 3, 1)
            #  plt.imshow(output_img1, cmap='gray')
            #  plt.title('Image 1')
            #  plt.subplot(1, 3, 2)
            #  plt.imshow(output_img2, cmap='gray')
            #  plt.title('Image 2')
            #  plt.subplot(1, 3, 3)
            #  plt.imshow(np.abs(output_img1-output_img2))
            #  plt.title('Image 1 - Image 2')
            #  plt.show()

            # break

    if img1 is None or img2 is None:
        raise ValueError("未能在任一gap_point子文件夹中找到两张图片")

  
    
    
    

if __name__ == "__main__":
    main()