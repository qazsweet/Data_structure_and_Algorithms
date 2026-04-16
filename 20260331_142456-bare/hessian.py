import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

class NewtonRingContactDetector:
    def __init__(self, threshold=0.5, buffer_size=5):
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.response_history = []
        self.is_contacted = False
        self.contact_frame_idx = -1

    def detect_blob_response(self, frame):
        """ 使用Hessian矩阵检测中心暗斑响应 """
        # 1. 预处理：灰度化与高斯模糊（滤除高频噪声）
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 2. 计算Hessian矩阵及其特征值
        # sigma越大，对越大的斑点敏感（接触后的黑斑通常比干涉条纹粗大）
        H = hessian_matrix(blurred, sigma=3.0, order='rc')
        i1, i2 = hessian_matrix_eigvals(H)
        
        # 对于暗斑（Dark Blob），特征值应为正且较大
        # 我们取较大特征值的平均值作为当前帧的“接触强度”指标
        response = np.max(i1) 
        return response

    def process_frame(self, frame, frame_idx):
        """ 时序处理逻辑 """
        current_res = self.detect_blob_response(frame)
        print(f"current_res: {current_res}")
        self.response_history.append(current_res)
        
        if len(self.response_history) > self.buffer_size:
            self.response_history.pop(0)
            
            # 计算时序方差：接触前条纹移动导致响应剧烈波动，接触后响应趋于稳定高值
            std_dev = np.std(self.response_history)
            mean_val = np.mean(self.response_history)
            print(f"std_dev: {std_dev}, mean_val: {mean_val}")
            
            # 判定条件：平均响应强度足够高，且近期波动显著降低
            if not self.is_contacted and mean_val > self.threshold and std_dev < (mean_val * 0.1):
                self.is_contacted = True
                self.contact_frame_idx = frame_idx
                return True
        
        return False

# --- 模拟处理流程 ---
def run_detection():
    detector = NewtonRingContactDetector(threshold=15.0) # 根据实际图像量级调整阈值
    frame_idx = 0

    for i in range(190, 203):
        img_name = './../20260311ya/list5/Basler_acA4112-20um__40713375__20260312_212456552_'+ str(format(i, '04d')) + '.bmp'

    # for i in range(136, 144):
    #     img_name = './../20260311ya/list2/Basler_acA4112-20um__40713375__20260312_205810205_'+ str(format(i, '04d')) + '.bmp'
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
        triggered = detector.process_frame(img, i)
        
        if triggered:
            print(f"Detected Contact at Frame: {i}")
            # 可以在此处绘制标记
            cv2.putText(frame, "CONTACT DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title('Newton Rings Stream')
        plt.show()

if __name__ == "__main__":
    run_detection()