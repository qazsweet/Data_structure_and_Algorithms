import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure
from scipy.signal import find_peaks, savgol_filter
import time


global_id = 0
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 

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
    gamma = 2.4
    print(f'gamma: {gamma}')

    # 3. 应用查找表进行快速矫正
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def extract_innermost_dark_boundary_v2(img, area_threshold=0):
    if img is None:
        print("Error: Image not found.")
        return

    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    # enhanced_img = cv2.resize(enhanced_img, (enhanced_img.shape[1] // 2, enhanced_img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    img_gamma = auto_gamma_correction(img)
    # 2. 增强对比度 (CLAHE)
    # clipLimit 越大对比度越强，tileGridSize 是局部窗口大小
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    # Resize the enhanced image to 1/4 (both width and height)
    enhanced_img = cv2.resize(enhanced_img, (enhanced_img.shape[1] // 2, enhanced_img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    # enhanced_img = cv2.resize(enhanced_img, (enhanced_img.shape[1] // 2, enhanced_img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    # enhanced_img = (((enhanced_img / 255) ** gamma) * 255).astype(np.uint8)

    # 3. 强力降噪
    # 使用中值滤波去除细小黑点/噪点，这对干涉图效果很好
    blurred = cv2.medianBlur(enhanced_img, 7)
    blurred = cv2.GaussianBlur(blurred, (9, 9), 0)

    # 4. 自适应阈值分割 (重点修改)
    # ADAPTIVE_THRESH_GAUSSIAN_C 能在亮度不均的情况下提取出边缘
    # blockSize 必须是奇数，控制局部范围；C 是从均值中减去的常数，调整灵敏度
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 1)

    # 5. 形态学操作：先开运算去噪，再闭运算填补断裂
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # 对 closed 做一次腐蚀操作
    # closed = cv2.erode(closed, kernel, iterations=1)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # fill = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    # for cnt in contours:
    #     cv2.drawContours(fill, [cnt], -1, (0, 255, 0), 1)
    #     cv2.putText(fill, str(cv2.contourArea(cnt)), (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # plt.figure()
    # plt.imshow(fill)
    # plt.show()

    # 7. 过滤并寻找最接近中心的圆
    height, width = enhanced_img.shape
    img_center = (width // 2, height // 2)
    candidates = []

    fill = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(f'area: {area}, area_threshold: {area_threshold}')
        # 调整面积阈值：最内圈通常不小于 1000 像素（根据原图比例）
        if area_threshold == 0:
            current_area_thd = 50
        else:
            current_area_thd = area_threshold
        if current_area_thd < area < (width * height * 0.8):
          
            # 计算圆形度
            # cv2.drawContours(fill, [cnt], -1, (0, 255, 0), 1)
            # cv2.putText(fill, str(cv2.contourArea(cnt)), (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # print(f'area: {area}, circularity: {circularity}')
            
            if circularity > 0.37: 
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    dist_to_center = (cX - img_center[0])**2 + (cY - img_center[1])**2

                    candidates.append({'cnt': cnt, 'circularity': circularity, 'area': area, 'cX': cX, 'cY': cY, 'dist_to_center': dist_to_center})

                cv2.drawContours(fill, [cnt], -1, (0, 255, 0), 1)
                sort_score = circularity-dist_to_center/height/width
                cv2.putText(fill, str(cv2.contourArea(cnt))+'_'+str(f"{sort_score:.2f}"), (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # plt.figure()
    # plt.imshow(fill)
    # plt.show()
    # 8. 排序：距离中心最近的通常就是第一条暗纹
    candidates.sort(key=lambda x: x['circularity']-x['dist_to_center']/height/width, reverse=True)

    if candidates:
        fourier_diff = np.fft.fft2(img)
        sum_hfq = np.sum(np.abs(fourier_diff))
        mean_hfq = np.mean(np.abs(fourier_diff))
        # candidates[i]['mean_hfq'] = mean_hfq
        print(f'mean_hfq: {mean_hfq:.2f}, sum_hfq: {sum_hfq:.2f}')

        for i in range(len(candidates)):
            cnt = candidates[i]['cnt']
            # # 计算到图像中心的距离
            # dist_to_center = np.sqrt((cX - img_center[0])**2 + (cY - img_center[1])**2)

            # # 计算轮廓内部像素的均值
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(img, mask=mask)[0]
            candidates[i]['mean_val'] = mean_val

        if len(candidates) > 1 and (np.abs(candidates[0]['cX']-candidates[1]['cX']) + np.abs(candidates[0]['cY']-candidates[1]['cY'])) <10:


            # 计算candidates[0]['cnt']之外，candidates[1]['cnt']之内的平均值
            mask1 = np.zeros(img.shape, dtype=np.uint8)
            mask2 = np.zeros(img.shape, dtype=np.uint8)
            # mask1: candidates[0]['cnt']区域
            cv2.drawContours(mask1, [candidates[0]['cnt']], -1, 255, -1)
            # mask2: candidates[1]['cnt']区域
            cv2.drawContours(mask2, [candidates[1]['cnt']], -1, 255, -1)
            # 只保留mask2减去mask1的区域
            ring_mask = cv2.subtract(mask2, mask1)
            mean_val_ring = cv2.mean(img, mask=ring_mask)[0]
            print(f"Mean value in the ring between candidates[0] and candidates[1]: {mean_val_ring:.2f}")
            fill = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            # for idx, cdd in enumerate(candidates):
            cnt1 = candidates[0]['cnt']
            cnt2 = candidates[1]['cnt']
            cv2.drawContours(fill, [cnt1], -1, (0, 255, 0), 1)
            cv2.drawContours(fill, [cnt2], -1, (0, 0, 255), 1)
            cv2.putText(fill, str(cv2.contourArea(cnt1))+'_'+str(f"{candidates[0]['circularity']:.2f}"), (int(cnt1[0][0][0]), int(cnt1[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(fill, str(cv2.contourArea(cnt2))+'_'+str(f"{candidates[1]['circularity']:.2f}"), (int(cnt2[0][0][0]), int(cnt2[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
            plt.figure()
            plt.imshow(fill)
            plt.title(f'[{candidates[0]['cX']}, {candidates[0]['cY']}] {candidates[0]['mean_val']:.2f} and [{candidates[1]['cX']}, {candidates[1]['cY']}] {candidates[1]['mean_val']:.2f}')
            plt.show()

                
                # # result = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
                
                # # # 绿色画出原始边缘
                # # cv2.drawContours(result, [cnt], -1, (0, 255, 0), 1)
                
                # # 红色画出拟合的最小外接圆
                # (x, y), radius = cv2.minEnclosingCircle(cnt)
                # # cv2.circle(result, (int(x), int(y)), int(radius), (0, 0, 255), 2)

                # print(f"Success: Found target with mean_val {mean_val:.2f}.")
                

    else:
        print("Still no circular fringes detected. Try lowering the 'circularity' threshold or 'C' in adaptiveThreshold.")
        fill = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            cv2.drawContours(fill, [cnt], -1, (0, 255, 0), 1)
            cv2.putText(fill, str(cv2.contourArea(cnt)), (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        plt.figure()
        plt.imshow(fill)
        plt.show()

    # for idx, cdd in enumerate(candidates):
    #     print(f'idx: {idx}, mean_val: {cdd['mean_val']:.2f}, cX:{cdd['cX']}, cY:{cdd['cY']}, area:{cdd['area']}, circularity:{cdd['circularity']:.2f}')
    #     cv2.drawContours(enhanced_img, [cdd['cnt']], -1, (0, 255, 0), 1)
    #     cv2.putText(enhanced_img, str(idx)+'_'+str(f"{cdd['mean_val']:.2f}"), (int(cdd['cnt'][0][0][0]), int(cdd['cnt'][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # plt.figure()
    # plt.imshow(enhanced_img)
    # plt.show()
    
    return enhanced_img, candidates

def extract_cnt_info(cnt, cx, cy, img):
    """
    统计cnt相对于(cx, cy)中心, 每40度方向的极半径(拟合半径)

    Args:
        cnt: 轮廓点 (N, 1, 2) or (N, 2)
        cx, cy: 中心坐标

    Returns:
        angles: 角度列表（度）
        radii:  各角度方向的半径
    """
    # 确保cnt形状为(N,2)
    if cnt.ndim == 3:
        points = cnt.reshape(-1, 2)
    else:
        points = cnt

    # 计算每个点和(cx,cy)连线的极角和半径
    vectors = points - np.array([cx, cy])
    thetas = np.arctan2(vectors[:,1], vectors[:,0])   # 弧度
    dists = np.sqrt(vectors[:,0]**2 + vectors[:,1]**2)
    angles = []
    radii = []
    # 对0~360每40度扫描
    for deg in range(0, 360, 40):
        theta = np.radians(deg+30-20)
        # # 计算与当前theta最接近的前两个点的索引
        # abs_diffs = np.abs((thetas - theta + np.pi) % (2 * np.pi) - np.pi)
        # idxs = np.argsort(abs_diffs)[:2]
        # # 记录角度：主角度及其左右点
        angles.append(deg+30-20)
        # # 取对应的极径
        # radii.append(dists[idxs[0]])

        # # 特殊颜色标记前两个点（可选：在 img 上画，或者后续返回两点坐标）
        # pt1 = points[idxs[0]]
        # pt2 = points[idxs[1]]
        # # 画法线：先求线段中点
        # midpoint = (pt1 + pt2) / 2
        # # 求圆心到中点的向量
        # vec_center_to_mid = midpoint - np.array([cx, cy])
        # norm = np.linalg.norm(vec_center_to_mid)
        # if norm != 0:
        #     unit_vec = vec_center_to_mid / norm
        # else:
        #     unit_vec = np.array([0, 0])
        # # 法线方向(和圆心到中点向外一致)
        # normal_start = midpoint.astype(int)
        # normal_length = 40  # 可调整
        # normal_end = (midpoint + unit_vec * normal_length).astype(int)
        idx = np.argmin(np.abs((thetas - theta + np.pi) % (2 * np.pi) - np.pi))
        radii.append(dists[idx])

    # ax = plt.subplot(111, projection='polar')
    # ax.imshow(img, cmap='gray')
    # ax.plot(np.radians(angles), radii, marker='o')
    # ax.set_theta_zero_location('E')
    # ax.set_theta_direction(-1)

    return angles, radii

def find_valley_peak_positions(y_data, distance=5, prominence=10):
    """
    计算波谷位置
    :param distance: 相邻波谷之间的最小距离（根据你的图像看，周期大约在8-10个单位，可以设为5）
    :param prominence: 显著性，用于忽略微小噪声波动
    """
    # 1. 预处理：使用 Savitzky-Golay 滤波器平滑信号（可选，视噪声情况而定）
    # window_length 必须为奇数，polyorder 为拟合阶数
    y_smooth = savgol_filter(y_data, window_length=5, polyorder=2)
    
    # 2. 将信号取反，使波谷变成波峰
    inverted_signal = -y_smooth
    
    # 3. 寻找峰值（即原始信号的波谷）
    # height: 限制最小高度; distance: 限制波谷间的最小像素间隔
    valleys, valleys_properties = find_peaks(inverted_signal, 
                                     distance=distance, height=0.1)
    
    peaks, peaks_properties = find_peaks(y_smooth, 
                                     distance=distance, 
                                     prominence=prominence, height=1.5)
    # print(f'valleys: {valleys}, valleys_properties: {valleys_properties}')
    # print(f'peaks: {peaks}, peaks_properties: {peaks_properties}')
    return valleys, valleys_properties, peaks, peaks_properties

def find_valley_positions(y_data, distance=5, prominence=10):
    """
    计算波谷位置
    :param distance: 相邻波谷之间的最小距离（根据你的图像看，周期大约在8-10个单位，可以设为5）
    :param prominence: 显著性，用于忽略微小噪声波动
    """
    # 1. 预处理：使用 Savitzky-Golay 滤波器平滑信号（可选，视噪声情况而定）
    # window_length 必须为奇数，polyorder 为拟合阶数
    y_smooth = savgol_filter(y_data, window_length=5, polyorder=2)
    
    # 2. 将信号取反，使波谷变成波峰
    inverted_signal = -y_smooth
    
    # 3. 寻找峰值（即原始信号的波谷）
    # height: 限制最小高度; distance: 限制波谷间的最小像素间隔
    valleys, properties = find_peaks(inverted_signal, 
                                     distance=distance, 
                                     prominence=prominence)

    refined_valleys = []
    for i in valleys:
        # 边界检查：确保左右都有点可以拟合
        if i == 0 or i == len(y_data) - 1:
            refined_valleys.append(float(i))
            continue
            
        # 取波谷及其左右相邻的两个点
        y1, y2, y3 = y_data[i-1], y_data[i], y_data[i+1]
        
        # 二次抛物线顶点公式（基于偏移量）
        # 偏移量 d = (y1 - y3) / (2 * (y1 - 2*y2 + y3))
        denominator = 2 * (y1 - 2 * y2 + y3)
        if denominator == 0:
            d = 0
        else:
            d = (y1 - y3) / denominator
            
        refined_valleys.append(i + d)
        
    return np.array(refined_valleys)
    # return valleys

from scipy.signal import hilbert, detrend

def extract_phase_from_signal(y_data):
    # 1. 去趋势 (Detrending)
    # 这一步非常关键，要把信号中心对齐到 0 轴左右
    y_centered = detrend(y_data - np.mean(y_data))
    
    # 2. 希尔伯特变换
    analytic_signal = hilbert(y_centered)
    
    # 3. 提取瞬时相位 (弧度)
    # np.angle 返回的是 [-pi, pi] 之间的包裹相位
    wrapped_phase = np.angle(analytic_signal)
    
    # 4. 相位解包裹 (Unwrap)
    # 将跳变点补全，使其变为连续增长或减少的直线/曲线
    unwrapped_phase = np.unwrap(wrapped_phase)
    
    # 5. 提取瞬时振幅包络 (Optional)
    amplitude_envelope = np.abs(analytic_signal)
    
    return wrapped_phase, unwrapped_phase, amplitude_envelope, y_centered

def extract_period(enhanced_img, angles, radii, cx, cy, length):
    period_values = []
    valley_indices = []
    for angle, radius in zip(angles, radii):
        theta = np.radians(angle)
        x_start = int(round(cx ))#  + radius * np.cos(theta)))
        y_start = int(round(cy))#  + radius * np.sin(theta)))
        x_end = int(round(cx + (radius + length) * np.cos(theta)))
        y_end = int(round(cy + (radius + length) * np.sin(theta)))
        profile = measure.profile_line(enhanced_img, (y_start, x_start), (y_end, x_end), linewidth=1, mode='constant')
        # print(f'profile shape: {profile.shape}')
        period_values.append(profile)

        # y_data = profile[:100]

        # wrapped, unwrapped, envelope, centered = extract_phase_from_signal(y_data)

        # # --- 可视化 ---
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # ax1.plot(centered, label='Centered Signal', color='blue', alpha=0.6)
        # ax1.plot(envelope, '--', label='Envelope', color='black')
        # ax1.set_title("Processed Signal & Envelope")
        # ax1.legend()

        # ax2.plot(wrapped, label='Wrapped Phase (-π to π)', color='green')
        # ax2.plot(unwrapped / (2*np.pi), label='Normalized Unwrapped Phase (Cycles)', color='red')
        # ax2.set_title("Extracted Phase")
        # ax2.set_xlabel("Time / Pixels")
        # ax2.legend()

        # plt.tight_layout()
        # plt.show()

        valley_indx = find_valley_positions(profile[:100])
        valley_indices.append(valley_indx)

        valleys, valleys_properties, peaks, peaks_properties = find_valley_peak_positions(profile[:100])
        # print(f'valleys: {valleys}, valleys_properties: {valleys_properties}, peaks: {peaks}, peaks_properties: {peaks_properties}')

        global global_id
        # np.save(f'profile_{global_id}_{angle:.0f}.npy', profile)
        try:
            print(f'first valley: {valleys[0]}, {profile[valleys[0]]:.2f}, first peak: {peaks[0]}, {profile[peaks[0]]:.2f}')
        except:
            print(f'no valleys or peaks found')
        # plt.figure()
        # plt.plot(profile)
        # plt.scatter(valleys, profile[valleys], color='red')
        # plt.scatter(peaks, profile[peaks], color='blue')
        # plt.savefig(f'peak_valley_{global_id}_{angle:.0f}.png')
        # plt.show()
    global_id += 1
    # plt.figure()
    # for profile in period_values:
    #     plt.plot(profile)
    # plt.show()

    return period_values, valley_indices


def main():
    area_threshold_temp = 0
    # for i in range(81, 91):
    #     img_name = './data7/Basler_acA4112-20um__40713375__20260312_214444394_'+ str(format(i, '04d')) + '.bmp'
    # for i in range(100, 109):
    #     img_name = './data6/Basler_acA4112-20um__40713375__20260312_213842974_'+ str(format(i, '04d')) + '.bmp'
        # img_name = './data6/Inverse_FFT_of_Basler_acA4112-20um__40713375__20260312_213842974_0104.bmp'
    # for i in range(208, 220):
    #     img_name = './../20260311ya/list4/Basler_acA4112-20um__40713375__20260312_211722080_'+ str(format(i, '04d')) + '.bmp'
    import os
    image_folder = './20260330_215619'
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
    for img_file in image_files:
        img_name = os.path.join(image_folder, img_file)
        base_filename = os.path.splitext(os.path.basename(img_name))[0]
        i = int(base_filename[-3:])
        print(f'start processing {img_file}')
        original_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]
        mean_val_temp = np.mean(original_img)
        min_val_temp = np.min(original_img)
        max_val_temp = np.max(original_img)
        # original_img = cv2.imread('Basler_acA4112-20um__40713375__20260312_205810205_0141.bmp', cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]
        start_time = time.time()

        enhanced_img, candidates = extract_innermost_dark_boundary_v2(original_img, area_threshold=area_threshold_temp)
        if len(candidates) >2:
            print(f'Mean value of [{candidates[0]['cX']}, {candidates[0]['cY']}]: {candidates[0]['mean_val']:.2f}, mean values of[{candidates[1]['cX']}, {candidates[1]['cY']}]: {candidates[1]['mean_val']:.2f}')
            
        if candidates:
            target_cnt = candidates[0]
            if target_cnt['area'] > 0.9*1000*1100:
                area_threshold_temp = 0
                print('not corrected area')
                break
                
            else:
                area_threshold_temp = target_cnt['area']

            angles, radii = extract_cnt_info(target_cnt['cnt'], target_cnt['cX'], target_cnt['cY'], enhanced_img)

            period_values, valley_indices = extract_period(enhanced_img, angles, radii, target_cnt['cX'], target_cnt['cY'], 200)
            # profile = np.asarray(period_values)
            # print(f'profile shape: {period_values.shape}')
            end_time = time.time()
            print(f'idx: {i}, extract_cnt_info time: {end_time - start_time} s')


            output_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            result = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            cdd = candidates[0]
            cv2.drawContours(result, [cdd['cnt']], -1, (0, 255, 0), 1)
            print(f'idx: {i},avg_img:{mean_val_temp:.2f},min_img:{min_val_temp:.2f},max_img:{max_val_temp:.2f},avg_cnt: {cdd['mean_val']:.2f}, cX:{cdd['cX']}, cY:{cdd['cY']}, area:{cdd['area']}, circularity:{cdd['circularity']:.2f}')

            # for idx, cdd in enumerate(candidates):
            #     cv2.drawContours(result, [cdd['cnt']], -1, (0, 255, 0), 1)
            #     # print(f'cnt: {cdd['cnt'].shape}')
            #     cv2.putText(result, str(idx)+'_'+str(f"{cdd['mean_val']:.2f}"), (int(cdd['cnt'][0][0][0]), int(cdd['cnt'][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     print(f'idx: {idx}, mean_val: {cdd['mean_val']:.2f}, cX:{cdd['cX']}, cY:{cdd['cY']}, area:{cdd['area']}, circularity:{cdd['circularity']:.2f}')


            for angle, radius, period_value, valley_index in zip(angles, radii, period_values, valley_indices):
                theta = np.radians(angle)
                x_end = int(round(target_cnt['cX'] + radius * np.cos(theta)))
                y_end = int(round(target_cnt['cY'] + radius * np.sin(theta)))
                cv2.line(result, (int(round(target_cnt['cX'])), int(round(target_cnt['cY']))), (x_end, y_end), ( 255, 0, 0), 1)
                # print(f'valley_index: {valley_index}')
                for i in valley_index:
                    cv2.circle(result, (int(round(target_cnt['cX'] + (radius+i) * np.cos(theta))), int(round(target_cnt['cY'] + (radius+i) * np.sin(theta)))), 1,  (0, 0, 255), 2)
                # cv2.putText(result, f'{angle:.2f}', (int(round(x_end)), int(round(y_end))), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            plt.figure()
            # plt.subplot(131)
            plt.imshow(result)
            plt.title(f'Detected Inner Fringe with mean_val {target_cnt['mean_val']:.2f}')
            plt.show()
        else:
            print(f'idx: {i},avg_img:{mean_val_temp:.2f},min_img:{min_val_temp:.2f},max_img:{max_val_temp:.2f}, find no candidates')
if __name__ == '__main__':
    main()
# plt.subplot(132)
# plt.imshow(fill)


# plt.subplot(133)
# for i in range(profile.shape[0]):  # profile is (201, 9)
#     plt.plot(profile[i,:], label=f'Line {i+1}')
# plt.legend(loc='best')
# plt.title('Radii Lines at Each Angle')
# plt.show()


# print(f"检测到的波谷索引位置: {valley_indices}")
# plt.figure(figsize=(15, 4))
# plt.plot(y_data, label='Original Signal', color='#1f77b4')
# plt.scatter(valley_indices, y_data[valley_indices], color='red', marker='v', label='Detected Valleys')

# # 标注波谷坐标
# for idx in valley_indices:
#     plt.text(idx, y_data[idx]-5, f"{idx}", ha='center', va='top', color='red', fontsize=9)

# plt.title("Valley Detection in Interference Signal")
# plt.xlabel("Time / Pixels")
# plt.ylabel("Intensity")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()



# print(f"检测到的波谷索引位置: {valley_indices}")
# plt.figure(figsize=(15, 4))
# plt.plot(y_data, label='Original Signal', color='#1f77b4')
# plt.scatter(valley_indices, y_data[valley_indices], color='red', marker='v', label='Detected Valleys')

# # 标注波谷坐标
# for idx in valley_indices:
#     plt.text(idx, y_data[idx]-5, f"{idx}", ha='center', va='top', color='red', fontsize=9)

# plt.title("Valley Detection in Interference Signal")
# plt.xlabel("Time / Pixels")
# plt.ylabel("Intensity")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()