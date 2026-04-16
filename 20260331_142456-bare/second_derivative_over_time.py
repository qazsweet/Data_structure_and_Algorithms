import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class ContrastGradientDetector:
    def __init__(self, window_size=5, roi_size=10, sensitivity=3.0):
        self.window_size = window_size
        self.roi_size = roi_size
        self.sensitivity = sensitivity  # 灵敏度因子，用于判定二阶导数突变
        
        # 存储中心ROI的平均亮度
        self.brightness_history = deque(maxlen=window_size + 2) 
        self.is_contacted = False

    def get_roi_brightness(self, frame):
        """ 获取图像中心 10x10 区域的平均亮度 """
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        r = self.roi_size // 2
        
        # 截取中心ROI
        roi = frame[cy-r:cy+r, cx-r:cx+r]
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
        return np.mean(roi)

    def detect_trigger(self, frame, frame_idx):
        """ 时序逻辑：寻找二阶导数峰值 """
        current_brightness = self.get_roi_brightness(frame)
        self.brightness_history.append(current_brightness)

        # 必须攒够至少5帧来计算稳定的梯度
        if len(self.brightness_history) < self.window_size:
            return False

        # 1. 计算一阶导数（亮度变化率）
        # diffs[i] 表示第 i 帧到第 i+1 帧的变化
        history_list = list(self.brightness_history)
        v1 = np.diff(history_list) 

        # 2. 计算二阶导数（变化率的变化 - 即加速度）
        v2 = np.diff(v1)

        # 3. 判定逻辑：
        # 接触瞬间，由于从“剧烈变暗/变亮”突然进入“静止暗斑”，
        # 二阶导数会产生一个脉冲响应。
        last_v2 = abs(v2[-1])
        avg_v2_history = np.mean(np.abs(v2[:-1])) if len(v2) > 1 else 0

        # 如果当前的加速度远大于历史平均波动，且当前亮度处于低位
        if not self.is_contacted:
            if last_v2 > (avg_v2_history * self.sensitivity) and current_brightness < 80:
                # 额外校验：接触后亮度应趋于稳定（方差减小）
                self.is_contacted = True
                return True

        return False

# --- 实时处理模拟 ---
def main():
    # 假设使用相机采集或读取视频
    # 建议 roi_size 根据实际干涉环中心大小调整，TTM对准中通常中心斑较大
    detector = ContrastGradientDetector(window_size=5, roi_size=1000, sensitivity=2.0)
    
    frame_idx = 0

    for i in range(190, 203):
        img_name = './../20260311ya/list5/Basler_acA4112-20um__40713375__20260312_212456552_'+ str(format(i, '04d')) + '.bmp'
        frame = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
        

        # 触发检测
        if detector.detect_trigger(frame, i):
            print(f"Trigger! Contact detected at frame {i}")
            # 在画面上标出ROI区域方便观察
            cv2.rectangle(frame, (frame.shape[1]//2-6, frame.shape[0]//2-6), 
                          (frame.shape[1]//2+6, frame.shape[0]//2+6), (0, 255, 0), 2)
            cv2.putText(frame, "CONTACT: FIRST FRAME", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        plt.figure()
        plt.imshow(frame, cmap='gray')
        plt.title('Real-time TTM Monitoring')
        plt.show()

        frame_idx += 1


if __name__ == "__main__":
    main()