import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
import os 


class DynamicPSIProcessor:
    def __init__(self, height, width):
        # 使用 deque 维持一个长度为 5 的滑动窗口
        self.window = deque(maxlen=5)
        self.height = height
        self.width = width

    def process_frame(self, new_frame):
        """
        每输入一帧新图像，尝试计算一次相位
        new_frame: 2D numpy array (uint8 or float)
        """
        self.window.append(new_frame.astype(np.float32))
        
        # 只有当窗口填满 5 张图时才开始计算
        if len(self.window) < 5:
            return None
        
        # 将 deque 转换为 3D array 方便向量化计算 [5, H, W]
        imgs = np.array(self.window)
        
        # Hariharan 5-step 公式:
        # phi = arctan2( 2*(I2 - I4), I1 - 2*I3 + I5 )
        # 索引对应：I1=imgs[0], I2=imgs[1], I3=imgs[2], I4=imgs[3], I5=imgs[4]
        
        numerator = 2.0 * (imgs[1] - imgs[3])
        denominator = imgs[0] - 2.0 * imgs[2] + imgs[4]
        
        # 使用 arctan2 处理全象限相位，范围 (-pi, pi]
        phase = np.arctan2(numerator, denominator)

        twoB = (imgs[4] - imgs[2])**2 + (imgs[1] - imgs[3])**2
        print(f"twoB: {twoB.mean():.2f}")
        
        return twoB

# --- 模拟算法运行 ---

# 假设图像尺寸
H, W = 512, 512
processor = DynamicPSIProcessor(H, W)

# # # 模拟 10 帧干涉图序列
# for i in range(200, 211):
#     img_name = './../20260311ya/list4/Basler_acA4112-20um__40713375__20260312_211722080_'+ str(format(i, '04d')) + '.bmp'
# # # for i in range(10):
# # for i in range(190, 203):
# #     img_name = './../20260311ya/list5/Basler_acA4112-20um__40713375__20260312_212456552_'+ str(format(i, '04d')) + '.bmp'


image_folder = './20260331_142456' # '../20260311ya/20260330_215619'
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
for img_file in image_files:

    img_name = os.path.join(image_folder, img_file)
    base_filename = os.path.splitext(os.path.basename(img_name))[0]
    i = int(base_filename[-3:])

    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1000:2200, 1900:3140]
    
    wrapped_phase = processor.process_frame(img)
    
    if wrapped_phase is not None:
        print(f"成功处理第 {i+1} 帧，生成相位图。")
        # 此处得到的 wrapped_phase 是截断相位，通常需要进一步做相位解包裹 (Unwrapping)
        # 例如使用 cv2.normalize 映射到 0-255 进行预览
        display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        plt.figure()
        plt.imshow(display, cmap='gray')
        plt.title(f"Frame {i}")
        plt.show()