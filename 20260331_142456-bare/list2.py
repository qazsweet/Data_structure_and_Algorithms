import os
import cv2
import numpy as np


def get_sorted_image_files(folder):
    # List all image files in the folder (you can add more extensions if needed)
    exts = ('.bmp', '.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    paths = [os.path.join(folder, f) for f in files]
    # Sort by file creation/modification time
    paths.sort(key=lambda fp: os.path.getmtime(fp))
    return paths

def process_images_in_folder(folder):
    image_files = get_sorted_image_files(folder)
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        # print(f"Processing image: {img_path}")
        # TODO: Add your image processing code here
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        img_min = np.min(img)
        img_max = np.max(img)
        Visibility = (img_max - img_min) / (img_max + img_min)
        print(f"img:{img_path.split('.')[-2][-4:]}, sobel:{np.sum(sobel):.2f}, Visibility:{Visibility:.2f}")

if __name__ == "__main__":
    image_folder = "./list4"
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist.")
    else:
        process_images_in_folder(image_folder)