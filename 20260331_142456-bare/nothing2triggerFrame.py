import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class HyperSensitiveTrigger:
    def __init__(self, sensitivity=2.0):
        self.bg_model = None
        self.bg_std_map = None
        self.sensitivity = sensitivity # 灵敏度，越小越灵敏（建议 1.5 - 2.5）

    def build_background(self, initial_frames):
        stack = np.array(initial_frames).astype(np.float32)
        self.bg_model = np.mean(stack, axis=0)
        # 关键：计算背景中每个像素点的固有噪声水平
        self.bg_std_map = np.std(stack, axis=0) + 1e-5 
        print("高灵敏度背景模型已建立。")

    def process_frame(self, frame):
        # 1. 预处理：轻微模糊平衡散斑噪声
        curr = cv2.GaussianBlur(frame.astype(np.float32), (3, 3), 0)
        
        # 2. 计算标准分 (Z-Score)
        # 这个步骤会将每个像素的变化量归一化到其自身的噪声水平上
        # 这样即使是暗处的微弱变化也会被放大
        z_score_map = np.abs(curr - self.bg_model) / self.bg_std_map
        
        # 3. 提取显著变化区域
        # 即使只有几个像素点开始形成条纹，max 或 top-k 均值也会跳变
        # 取 99.9 百分位点的值，代表图中变化最剧烈的部分
        peak_signal = np.percentile(z_score_map, 99.9)
        
        return peak_signal

class InterferometryTrigger:
    def __init__(self, threshold_factor=3.5, roi=None):
        """
        :param threshold_factor: 灵敏度系数，越大越不容易误触发
        :param roi: 兴趣区域 [y1, y2, x1, x2]，建议关注镜片中心区域
        """
        self.bg_model = None
        self.threshold_factor = threshold_factor
        self.roi = roi
        self.diff_history = []
        self.is_triggered = False
        self.trigger_frame_idx = -1

    def build_background(self, initial_frames):
        """
        使用前 N 帧（无干涉条纹时）建立高精度的平均背景
        """
        stack = np.array(initial_frames).astype(np.float32)
        self.bg_model = np.mean(stack, axis=0)
        
        # 计算背景的基准噪声水平 (标准差)
        self.base_std = np.std(self.bg_model)
        print("背景模型构建完成。")

    def process_frame(self, frame, frame_idx):
        if self.is_triggered:
            return False

        # 1. 预处理：高斯滤波去噪
        blurred = cv2.GaussianBlur(frame.astype(np.float32), (5, 5), 0)
        
        # 2. 区域裁剪 (ROI)
        if self.roi:
            y1, y2, x1, x2 = self.roi
            curr_roi = blurred[y1:y2, x1:x2]
            bg_roi = self.bg_model[y1:y2, x1:x2]
        else:
            curr_roi, bg_roi = blurred, self.bg_model

        # 3. 背景减除：计算绝对差值图
        diff = cv2.absdiff(curr_roi, bg_roi)

        # 4. 特征提取：计算差分图的标准差 (反映了图像结构的突变)
        current_std = np.std(diff)
        self.diff_history.append(current_std)

        # 5. 动态判定
        # 如果当前波动的标准差显著高于初始背景的扰动，判定为条纹出现
        if len(self.diff_history) > 5:
            moving_avg = np.mean(self.diff_history[:-1])
            if current_std > moving_avg * self.threshold_factor:
                self.is_triggered = True
                self.trigger_frame_idx = frame_idx
                return True
        
        return False

# --- 模拟与测试脚本 ---

# 1. 模拟生成背景和干涉图像序列（实际使用时替换为 cv2.VideoCapture）
def generate_mock_data():
    frames = []
    bg = np.full((512, 512), 100, dtype=np.uint8)
    # 添加背景噪声
    for i in range(50):
        noise = np.random.normal(0, 2, bg.shape).astype(np.uint8)
        frames.append(cv2.add(bg, noise))
    
    # 从第 30 帧开始引入微弱的环形条纹
    x, y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
    r = np.sqrt(x**2 + y**2)
    for i in range(20):
        # 条纹强度随时间逐渐增强
        strength = i * 2 
        fringe = (strength * (1 + np.sin(r * 100))).astype(np.uint8)
        noise = np.random.normal(0, 2, bg.shape).astype(np.uint8)
        frames.append(cv2.add(cv2.add(bg, fringe), noise))
    return frames


def main():
    
    sequence = []
     
    image_folder = './20260331_142456' # list2' # '../20260311ya/20260330_215619'
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
    for img_file in image_files:
        img_name = os.path.join(image_folder, img_file)
        base_filename = os.path.splitext(os.path.basename(img_name))[0]
        i = int(base_filename[-3:])

        print(f'start processing {i}')
        original_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]
        sequence.append(original_img)
    
    hyper_sensitive_detector = HyperSensitiveTrigger(sensitivity=2.0)
    hyper_sensitive_detector.build_background(sequence[:10])

    peak_signal_list = []
    for idx, frame in enumerate(sequence):
        peak_signal = hyper_sensitive_detector.process_frame(frame)
        print(f'peak_signal: {peak_signal} [{idx}]{image_files[idx]}')
        peak_signal_list.append(peak_signal)

    average_peak_signal = np.mean(peak_signal_list[:10])
    sigma = np.std(peak_signal_list[:10])
    threshold = average_peak_signal + sigma*4
    indices = [i for i, v in enumerate(peak_signal_list) if v > threshold]
    first_index = indices[0] if len(indices) > 0 else None
    second_index = indices[1] if len(indices) > 1 else None

    print(f'mean of peak_signal_list: {average_peak_signal}')
    print(f'sigma of peak_signal_list: {sigma}')
    print(f'average + sigma threshold: {threshold}')
    print(f'first index > threshold: {first_index}')
    print(f'second index > threshold: {second_index}')

    plt.figure(figsize=(10, 5))
    plt.plot(peak_signal_list)
    plt.scatter(first_index, peak_signal_list[first_index], color='red', label=f'{image_files[first_index][-8:-4]}: {peak_signal_list[first_index]:.2f}')
    plt.scatter(second_index, peak_signal_list[second_index], color='blue', label=f'{image_files[second_index][-8:-4]}: {peak_signal_list[second_index]:.2f}')
    plt.title(f'peak_signal {image_folder[2:]}')
    plt.xlabel('frame')
    plt.ylabel('peak_signal')
    plt.legend()
    plt.show()
    
    # detector = InterferometryTrigger(threshold_factor=2.5, roi=(128, 384, 128, 384))

    # # 第一步：建立背景（假设前10帧是干净的）
    # detector.build_background(sequence[:10])

    # print(f'start processing {len(sequence)} frames')

    # 第二步：逐帧检测
    # for idx, frame in enumerate(sequence):
    #     print(f'start processing {idx}')
    #     if detector.process_frame(frame, idx):
    #         print(f"检测到干涉条纹！触发帧索引: {idx}")
    #         diff_img = cv2.absdiff(frame.astype(np.float32), detector.bg_model.astype(np.float32))

    #         # 可视化结果
    #         plt.figure(figsize=(10, 5))
    #         plt.subplot(121), plt.title("Trigger Frame"), plt.imshow(frame, cmap='gray')
    #         plt.subplot(122), plt.title("Diff (Signal)"), plt.imshow(diff_img, cmap='jet')
    #         plt.title(f'[{idx}]{image_files[idx]}')
    #         plt.show()
    #     else:
    #         print(f"未检测到干涉条纹！帧索引: {idx}")

if __name__ == "__main__":

    main()