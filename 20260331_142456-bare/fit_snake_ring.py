import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import os

def process_interference_image(image_path, output_path='result.png'):
    """
    使用 Snake 算法提取干涉条纹的平滑外边界。
    """
    # 1. 载入图像
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    print(f"Processing image of size: {w}x{h}")

    # ==========================================
    # 2. 预处理：生成纹理能量图 (Texture Energy Map)
    # 目标是创建一个图像，其中有干涉条纹的地方是亮的，背景是暗的。
    # 我们使用局部方差（Local Variance）来做到这一点。
    # ==========================================
    
    # 将图像转换为浮点数进行计算
    img_float = img.astype(np.float32)
    
    # 计算局部均值 E[X] (使用均值滤波)
    blur_kernel_size = 15 # 窗口大小需要能覆盖几条条纹
    mean = cv2.blur(img_float, (blur_kernel_size, blur_kernel_size))
    
    # 计算局部均方值 E[X^2]
    mean_sq = cv2.blur(img_float**2, (blur_kernel_size, blur_kernel_size))
    
    # 计算局部方差 Var(X) = E[X^2] - (E[X])^2
    variance = mean_sq - mean**2
    # 确保没有负值
    variance = np.maximum(variance, 0)
    
    # 归一化到 [0, 1] 范围，供 skimage 使用
    variance_norm = cv2.normalize(variance, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # 对方差图进行轻微高斯平滑，去除局部噪声，让 Snake 收敛更顺滑
    energy_image = gaussian(variance_norm, sigma=2)

    # ==========================================
    # 3. 创建初始轮廓 (Initial Contour)
    # Snake 需要一个起始位置。我们通过形态学方法自动生成它。
    # ==========================================
    
    # 将方差图转回 8-bit 用于 OpenCv 处理
    var_8u = (variance_norm * 255).astype(np.uint8)
    
    # 二值化
    _, thresh = cv2.threshold(var_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学闭运算：填充条纹中间的暗部，连成一个实心区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 再次膨胀，确保初始轮廓完全包裹住干涉环（Snake 是向内收缩的）
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 提取初始轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: Could not find any initial contours.")
        return
        
    # 获取最大的轮廓（假设干涉环是最大的纹理区域）
    fill = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(fill, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(fill, cv2.COLOR_BGR2RGB))
    plt.title('Fringe Boundary')
    plt.axis('off')
    plt.show()
    cnt = max(contours, key=cv2.contourArea)
    
    # 将 OpenCV 格式的轮廓 [N, 1, 2] 转换为 Snake 需要的 [N, 2] (y, x) 格式
    initial_snake = cnt.squeeze()
    # 交换 x, y 坐标，因为 skimage 接收 (row, col)
    initial_snake = np.flip(initial_snake, axis=1)

    # ==========================================
    # 4. 运行 Snake 算法 (Active Contour)
    # ==========================================
    
    # 优化参数:
    # alpha: 连续性（弹性），越大越倾向于成为直/平滑线。
    # beta:  曲率（刚性），越大越倾向于成为直线。
    # gamma: 迭代步长。
    print("Running Snake algorithm...")
    
    # 我们希望 Snake 能够“贴合”纹理能量的边界
    # 在这个能量图中，最外圈条纹是亮度下降最快的地方。
    
    snake_coords = active_contour(
        energy_image,           # 运行在纹理能量图上
        initial_snake,          # 初始位置
        alpha=0.015,            # 允许一定弹性
        beta=0.1,               # 保持一定平滑
        gamma=0.01,             # 步长
        w_line=0,               # 不寻找亮线
        w_edge=1,               # 寻找能量图的边缘
        max_num_iter=2500,      # 迭代次数
        boundary_condition='cyclic' # 闭合曲线
    )
    
    print("Snake optimization complete.")

    # ==========================================
    # 5. 可视化和保存结果
    # ==========================================
    
    # 绘制结果在原图上
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap=plt.cm.gray)
    
    # 绘制初始轮廓（绿色）
    ax.plot(initial_snake[:, 1], initial_snake[:, 0], '--g', lw=2, label='Initial Contour')
    # 绘制 Snake 收敛后的轮廓（红色）
    ax.plot(snake_coords[:, 1], snake_coords[:, 0], '-r', lw=3, label='Optimized Snake Boundary')
    
    ax.set_title("Snake Algorithm Boundary Detection")
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Result saved to {output_path}")
    
    # (可选) 将 Snake 轮廓绘制回 OpenCV 图像并保存
    # 转换坐标回 (x, y) 整数
    processed_snake = np.flip(snake_coords, axis=1).astype(np.int32)
    processed_snake = processed_snake.reshape((-1, 1, 2))
    
    # 绘制红线
    cv2.polylines(img_color, [processed_snake], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.imwrite('opencv_result.png', img_color)
    print("OpenCV result image 'opencv_result.png' also saved.")

    # plt.show() # 如果有 GUI 界面可以取消注释此行查看

if __name__ == '__main__':
    # 请确保 image_0.png 存在
    input_filename = './20260330_215619/20260330_215719_341.png'
    process_interference_image(input_filename)