import cv2
import numpy as np
import matplotlib.pyplot as plt

# file_path = './20260330_215619/20260330_215719_341.png'
# img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]

# if img is None:
#     raise FileNotFoundError(f"Could not read file: {file_path}")

# # Optionally crop to region of interest (uncomment if you want to match example context)
# # img = img[1100:2100, 2000:3100]

# # Create meshgrid for image coordinates
# h, w = img.shape
# x = np.arange(w)
# y = np.arange(h)
# X, Y = np.meshgrid(x, y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, img, cmap='gray')
# ax.set_title('3D Surface Plot of Image Intensity')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Intensity')
# plt.show()

def auto_gamma_correction(image, target_mean=128):
    # 1. 计算当前图像的平均亮度
    current_mean = np.mean(image)
    
    # 避免除以 0 或极值
    if current_mean <= 0: return image
    
    # 2. 计算需要的 Gamma 值
    # 公式：(current_mean / 255) ^ gamma = (target_mean / 255)
    gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
    
    # 限制 Gamma 的范围，防止过度畸变（通常工业场景建议在 0.25 - 4.0 之间）
    gamma = np.clip(gamma, 0.25, 4.0)

    print(f'gamma: {gamma}')
    gamma = 1.0102
    # 3. 应用查找表进行快速矫正
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    img_gamma = (((image / 255.0) ** invGamma) * 255).astype("uint8")
    return img_gamma

def find_fringe_boundary(img):
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)

    # img_gamma = auto_gamma_correction(img)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    
    # enhanced_img = cv2.resize(enhanced_img, (enhanced_img.shape[1] // 2, enhanced_img.shape[0] // 2), interpolation=cv2.INTER_AREA)


    blurred = cv2.medianBlur(enhanced_img, 7)
    blurred = cv2.GaussianBlur(blurred, (9, 9), 0)


    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 1)


    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours, img

def extract_candidates(contours, resized_img, file_path, idx):
    candidates = []
    
    fill = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > 1000 and cnt_area < 2e5:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * cnt_area / (perimeter * perimeter)
            # print(f'area: {area}, circularity: {circularity}')
            if circularity > 0.37: 
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(fill, (cX, cY), 5, (0, 0, 255), -1)
                    candidates.append({'cnt': cnt, 'circularity': circularity, 'area': cnt_area, 'cX': cX, 'cY': cY})
                cv2.drawContours(fill, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(fill, str(cnt_area), (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    candidates.sort(key=lambda x: x['area'], reverse=False)
    if len(candidates) < 3:
        print(f'no enough candidates: {len(candidates)}')
        return candidates
    
    # display output
    target_cnt = candidates[0]
    second_cnt = candidates[1]
    third_cnt = candidates[2]
    forth_cnt = candidates[3]

    cv2.drawContours(fill, [target_cnt['cnt']], -1, (255, 0, 0), 2)
    cv2.putText(fill, str(target_cnt['area']), (int(target_cnt['cnt'][0][0][0]), int(target_cnt['cnt'][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(fill, (target_cnt['cX'], target_cnt['cY']), 1, (0, 0, 255), -1)

    cv2.drawContours(fill, [second_cnt['cnt']], -1, (255, 128, 0), 2)
    cv2.putText(fill, str(second_cnt['area']), (int(second_cnt['cnt'][0][0][0]), int(second_cnt['cnt'][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(fill, (second_cnt['cX'], second_cnt['cY']), 1, (0, 0, 255), -1)

    cv2.drawContours(fill, [third_cnt['cnt']], -1, (255, 196, 0), 2)
    cv2.putText(fill, str(third_cnt['area']), (int(third_cnt['cnt'][0][0][0]), int(third_cnt['cnt'][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(fill, (third_cnt['cX'], third_cnt['cY']), 1, (0, 0, 255), -1)

    cv2.drawContours(fill, [forth_cnt['cnt']], -1, (255, 255, 0), 2)
    cv2.putText(fill, str(forth_cnt['area']), (int(forth_cnt['cnt'][0][0][0]), int(forth_cnt['cnt'][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(fill, (forth_cnt['cX'], forth_cnt['cY']), 1, (0, 0, 255), -1)


    # 计算目标轮廓(target_cnt)内的原始图像均值
    mask_inner = np.zeros(resized_img.shape, dtype=np.uint8)
    cv2.drawContours(mask_inner, [target_cnt['cnt']], -1, 255, thickness=-1)
    mean_inside_cnt = cv2.mean(resized_img, mask=mask_inner)[0]
    min_inside_cnt = np.min(resized_img[mask_inner == 255])
    print(f"Mean value inside target_cnt: {mean_inside_cnt}; Min value inside target_cnt: {min_inside_cnt}")


    # 计算第二圈(second_cnt)但排除最内圈(target_cnt)的环形均值
    mask_outer = np.zeros(resized_img.shape, dtype=np.uint8)
    cv2.drawContours(mask_outer, [second_cnt['cnt']], -1, 255, thickness=-1)
    ring_mask = cv2.subtract(mask_outer, mask_inner)
    mean_ring = cv2.mean(resized_img, mask=ring_mask)[0]
    min_ring = np.min(resized_img[ring_mask == 255])
    print(f"Mean value of ring (between second_cnt and target_cnt): {mean_ring}; Min value of ring (between second_cnt and target_cnt): {min_ring}")

    # 计算第三圈(third_cnt)但排除最内圈(target_cnt)和第二圈(second_cnt)的环形均值
    mask_outer2 = np.zeros(resized_img.shape, dtype=np.uint8)
    cv2.drawContours(mask_outer2, [third_cnt['cnt']], -1, 255, thickness=-1)
    ring_mask = cv2.subtract(mask_outer2, mask_inner+mask_outer)

    mean_ring_outer = cv2.mean(resized_img, mask=ring_mask)[0]
    min_ring_outer = np.min(resized_img[ring_mask == 1])
    print(f"Mean value of ring (between third_cnt and second_cnt): {mean_ring_outer }; Min value of ring (between third_cnt and second_cnt): {min_ring_outer }")

    # 计算第四圈(forth_cnt)但排除最内圈(target_cnt)和第二圈(second_cnt)的环形均值
    mask_outer3 = np.zeros(resized_img.shape, dtype=np.uint8)
    cv2.drawContours(mask_outer3, [forth_cnt['cnt']], -1, 255, thickness=-1)
    ring_mask = cv2.subtract(mask_outer3, mask_inner+mask_outer+mask_outer2)

    mean_ring_outer2 = cv2.mean(resized_img, mask=ring_mask)[0]
    min_ring_outer2 = np.min(resized_img[ring_mask == 1])
    print(f"Mean value of ring (between forth_cnt and third_cnt): {mean_ring_outer2 }; Min value of ring (between forth_cnt and third_cnt): {min_ring_outer2 }")

    if np.abs(mean_inside_cnt - mean_ring_outer2) > 15:
        if (mean_inside_cnt < mean_ring -5.8) and (np.abs(mean_ring_outer - mean_inside_cnt) >2.9):
            flag = True
        else:
            flag = False
    else:
        if (mean_inside_cnt < mean_ring -5.8) and (np.abs(mean_ring_outer - mean_inside_cnt) >2.9):
            flag = True
        else:
            flag = False
    
    plt.figure(figsize=(8,8))
    plt.imshow(fill)
    plt.title(f'flag:{flag}; inner:{mean_inside_cnt:.2f}; ring:{mean_ring:.2f}; ring_outer:{mean_ring_outer:.2f}; ring_outer2:{mean_ring_outer2:.2f}')
    plt.savefig(f'{file_path[2:-4]}_{idx:03d}_{flag}.png')
    plt.show()

    return candidates


import os
from natsort import natsorted

# 遍历文件夹中的图片（按文件名排序，即时间顺序）
folder_path = './20260401_164705' # 20260331_142456' # 20260330_215004' # 20260331_142456' #  
img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
img_files = natsorted(img_files)  # 自然排序时间顺序

for i, file_name in enumerate(img_files):
    file_path = os.path.join(folder_path, file_name)# file_path = './20260330_215619/20260330_215719_341.png'
    
    print('*'*30)
    print(f'processing: {file_name}')
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]
    img_gamma = auto_gamma_correction(img)
    contours, resized_img = find_fringe_boundary(img_gamma)

    candidates = extract_candidates(contours, resized_img, file_name, i)

