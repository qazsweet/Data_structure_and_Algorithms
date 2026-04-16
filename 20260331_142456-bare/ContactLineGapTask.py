from Base import BaseExecuteTask
import traceback
import json
import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass, asdict
import time
from Base.pylog import DebugPrinter, LogLevel


@dataclass
class ContactLineGapInput:
    width_image: int
    height_image: int
    threshold: float

@dataclass
class ContactLineGapOutput:
    gap_flag: int
    status:bool
    error_message:str=None

class ContactLineGapTask(BaseExecuteTask.BaseExecuteTask):
    def __init__(self):
        time_point = time.strftime("%Y%m%d_%H", time.localtime(time.time()))
        self.min_logger_level = "INFO"
        self.logger = DebugPrinter(
            min_level=LogLevel[self.min_logger_level], 
            prefix="[PYTHON]",
            log_folder="./PyLogs",
            log_file= f"ContactLineGapTask_{time_point}.log",
        )

    def GetTaskName(self)->str:
        return "ContactLineGapTask"

    def auto_gamma_correction(self, image, target_mean=128):
        current_mean = np.mean(image)
        if current_mean <= 0: return image
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.25, 4.0)
        # gamma = 1.0102
        
        self.logger(f'[ContactLineGapTask] image gamma: {gamma:.4f}', level=LogLevel.INFO)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                        for i in np.arange(0, 256)]).astype("uint8")
        # img_gamma = cv2.LUT(image, table)
        img_gamma = np.array(((image / 255.0) ** invGamma) * 255 ).astype("uint8")
        return gamma, img_gamma

    def find_fringe_boundary(self, img):
        try:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.logger(f'[ContactLineGapTask] find_fringe_boundary resize: {e}', level=LogLevel.ERROR)
            return [], img
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(img)
        except Exception as e:
            self.logger(f'[ContactLineGapTask] find_fringe_boundary createCLAHE: {e}', level=LogLevel.ERROR)
            return [], img
        try:
            blurred = cv2.medianBlur(enhanced_img, 7)
            blurred = cv2.GaussianBlur(blurred, (9, 9), 0)
        except Exception as e:
            self.logger(f'[ContactLineGapTask] find_fringe_boundary GaussianBlur: {e}', level=LogLevel.ERROR)
            return [], img
        try:
            thresh = cv2.adaptiveThreshold(blurred, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 21, 1)
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            self.logger(f'[ContactLineGapTask] find_fringe_boundary morphologyEx: {e}', level=LogLevel.ERROR)
            return [], img

        try:
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            self.logger(f'[ContactLineGapTask] find_fringe_boundary findContours: {e}', level=LogLevel.ERROR)
            return [], img
        
        self.logger(f'[ContactLineGapTask] generate {len(contours)} contours', level=LogLevel.INFO)
        return contours, img

    def extract_candidates(self, contours, resized_img, gamma, thd)->int:
        candidates = []

        height, width = resized_img.shape
        self.logger(f'[ContactLineGapTask] height {height}, width {width}', level=LogLevel.INFO)
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > 1000 and cnt_area < height*width*0.8:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * cnt_area / (perimeter * perimeter)
                # print(f'area: {area}, circularity: {circularity}')
                if circularity > 0.37: 
                    
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        candidates.append({'cnt': cnt, 'circularity': circularity, 'area': cnt_area, 'cX': cX, 'cY': cY})

        candidates.sort(key=lambda x: x['area'], reverse=False)
        
        self.logger(f'[ContactLineGapTask] extract {len(candidates)} candidates', level=LogLevel.INFO)
        if len(candidates) < 3:
            self.logger(f'[ContactLineGapTask] {len(candidates)} no enough candidates.', level=LogLevel.INFO)
            return -1
        
        # display output
        target_cnt = candidates[0]
        self.logger(f'[ContactLineGapTask] target_cnt area: {target_cnt['area']:.4f}, cX: {target_cnt['cX']:.4f}, cY: {target_cnt['cY']:.4f}', level=LogLevel.INFO)
        second_cnt = candidates[1]
        self.logger(f'[ContactLineGapTask] second_cnt area: {second_cnt['area']:.4f}, cX: {second_cnt['cX']:.4f}, cY: {second_cnt['cY']:.4f}', level=LogLevel.INFO)
        third_cnt = candidates[2]
        self.logger(f'[ContactLineGapTask] third_cnt area: {third_cnt['area']:.4f}, cX: {third_cnt['cX']:.4f}, cY: {third_cnt['cY']:.4f}', level=LogLevel.INFO)

        mask_inner = np.zeros(resized_img.shape, dtype=np.uint8)
        cv2.drawContours(mask_inner, [target_cnt['cnt']], -1, 255, thickness=-1)
        mean_inside_cnt = cv2.mean(resized_img, mask=mask_inner)[0]
        min_inside_cnt = np.min(resized_img[mask_inner == 255])
        self.logger(f'[ContactLineGapTask] Mean value inside target_cnt {mean_inside_cnt:.4f}, Min value inside target_cnt {min_inside_cnt:.4f}', level=LogLevel.INFO)
        mask_outer = np.zeros(resized_img.shape, dtype=np.uint8)
        cv2.drawContours(mask_outer, [second_cnt['cnt']], -1, 255, thickness=-1)
        ring_mask = cv2.subtract(mask_outer, mask_inner)
        mean_ring = cv2.mean(resized_img, mask=ring_mask)[0]
        min_ring = np.min(resized_img[ring_mask == 255])
        self.logger(f'[ContactLineGapTask] Mean value of ring (between second_cnt and target_cnt): {mean_ring:.4f}, Min value of ring (between second_cnt and target_cnt): {min_ring:.4f}', level=LogLevel.INFO)


        if np.abs(mean_inside_cnt - mean_ring) > 15:
            mask_outer2 = np.zeros(resized_img.shape, dtype=np.uint8)
            cv2.drawContours(mask_outer2, [third_cnt['cnt']], -1, 255, thickness=-1)
            ring_mask = cv2.subtract(mask_outer2, mask_inner+mask_outer)

            mean_ring_outer = cv2.mean(resized_img, mask=ring_mask)[0]
            min_ring_outer = np.min(resized_img[ring_mask == 1])
            self.logger(f"Mean value of ring (between third_cnt and second_cnt): {mean_ring_outer:.4f}; Min value of ring (between third_cnt and second_cnt): {min_ring_outer:.4f}", level=LogLevel.INFO)

            if (mean_inside_cnt < mean_ring - thd) and (np.abs(mean_ring_outer - mean_inside_cnt) >thd/2):
                flag = 1
            else:
                flag = 0
        else:
            if len(candidates) < 4:
                self.logger(f'[ContactLineGapTask] {len(candidates)} no enough candidates.', level=LogLevel.INFO)
                return -1
            else:
                        
                target_cnt = candidates[1]
                self.logger(f'[ContactLineGapTask] target_cnt area: {target_cnt['area']:.4f}, cX: {target_cnt['cX']:.4f}, cY: {target_cnt['cY']:.4f}', level=LogLevel.INFO)
                second_cnt = candidates[2]
                self.logger(f'[ContactLineGapTask] second_cnt area: {second_cnt['area']:.4f}, cX: {second_cnt['cX']:.4f}, cY: {second_cnt['cY']:.4f}', level=LogLevel.INFO)
                third_cnt = candidates[3]
                self.logger(f'[ContactLineGapTask] third_cnt area: {third_cnt['area']:.4f}, cX: {third_cnt['cX']:.4f}, cY: {third_cnt['cY']:.4f}', level=LogLevel.INFO)

                mask_inner = np.zeros(resized_img.shape, dtype=np.uint8)
                cv2.drawContours(mask_inner, [target_cnt['cnt']], -1, 255, thickness=-1)
                mean_inside_cnt = cv2.mean(resized_img, mask=mask_inner)[0]
                min_inside_cnt = np.min(resized_img[mask_inner == 255])
                self.logger(f'[ContactLineGapTask] Mean value inside target_cnt {mean_inside_cnt:.4f}, Min value inside target_cnt {min_inside_cnt:.4f}', level=LogLevel.INFO)
                mask_outer = np.zeros(resized_img.shape, dtype=np.uint8)
                cv2.drawContours(mask_outer, [second_cnt['cnt']], -1, 255, thickness=-1)
                ring_mask = cv2.subtract(mask_outer, mask_inner)
                mean_ring = cv2.mean(resized_img, mask=ring_mask)[0]
                min_ring = np.min(resized_img[ring_mask == 255])
                self.logger(f'[ContactLineGapTask] Mean value of ring (between second_cnt and target_cnt): {mean_ring:.4f}, Min value of ring (between second_cnt and target_cnt): {min_ring:.4f}', level=LogLevel.INFO)

                mask_outer2 = np.zeros(resized_img.shape, dtype=np.uint8)
                cv2.drawContours(mask_outer2, [third_cnt['cnt']], -1, 255, thickness=-1)
                ring_mask = cv2.subtract(mask_outer2, mask_inner+mask_outer)

                mean_ring_outer = cv2.mean(resized_img, mask=ring_mask)[0]
                min_ring_outer = np.min(resized_img[ring_mask == 1])
                self.logger(f"Mean value of ring (between third_cnt and second_cnt): {mean_ring_outer:.4f}; Min value of ring (between third_cnt and second_cnt): {min_ring_outer:.4f}", level=LogLevel.INFO)

                if (mean_inside_cnt < mean_ring - thd) and ((mean_inside_cnt - mean_ring_outer) >thd/2):
                    flag = 1
                else:
                    flag = 0

        self.logger(f'[ContactLineGapTask] candidate flag: {flag}', level=LogLevel.INFO)
        return flag

    def check_contact_line_gap(self, img0, thd):
        
        img = img0[1100:2100, 2000:3100]
		# try:
        gamma, img_gamma = self.auto_gamma_correction(img)
        contours, resized_img = self.find_fringe_boundary(img_gamma)
        flag = self.extract_candidates(contours, resized_img, gamma, thd)
        # except Exception as e:
        # self.logger(f'[ContactLineGapTask] check_contact_line_gap fail {e}', level=LogLevel.INFO)
        self.logger(f'[ContactLineGapTask] check_contact_line_gap return {flag}', level=LogLevel.INFO)
        return flag

    
    def Execute(self, paramString: str, vMemDatas : list[BaseExecuteTask.MemData] ):
        self.logger(f'[ContactLineGapTask] Execute start', level=LogLevel.INFO)
        self.logger(f'[ContactLineGapTask] Execute start {sys.executable}', level=LogLevel.INFO)
        
        try:
            json_dict = json.loads(paramString)
            image_info = ContactLineGapInput(**json_dict)
            self.logger(f'[ContactLineGapTask] ContactLineGapInput load successfully: Width={image_info.width_image}, Height={image_info.height_image}', level=LogLevel.INFO)
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f'error_msg:{error_msg}')
            self.logger(f'[ContactLineGapTask] ContactLineGapInput load failed: {e}', level=LogLevel.ERROR)
            self.logger(f'[ContactLineGapTask] ContactLineGapInput load failed: {error_msg}', level=LogLevel.ERROR)
            self.records = ContactLineGapOutput(gap_flag=-1, status=False, error_message=str(f'[ContactLineGapTask] ContactLineGapInput load failed: {e};{error_msg}'))
            return 0
        
        try:
            thd = image_info.threshold
            img0 = vMemDatas[0].as_numpy(dtype=np.uint8).reshape((image_info.height_image, image_info.width_image))
            self.logger(f'[ContactLineGapTask] image load successfully : {img0.shape}', level=LogLevel.INFO)
        except Exception as e:
            error_msg = traceback.format_exc()
            self.logger(f'[ContactLineGapTask] image load failed: {error_msg}', level=LogLevel.ERROR)
            self.logger(f'[ContactLineGapTask] {len(vMemDatas)} image load failed: {e}', level=LogLevel.ERROR)
            self.records = ContactLineGapOutput(gap_flag=-1, status=False, error_message=str(f'[ContactLineGapTask] image load failed: {e}'))
            return 0


        img = img0[1100:2100, 2000:3100]
        try:
            gamma, img_gamma = self.auto_gamma_correction(img)
        except Exception as e:
            error_msg = traceback.format_exc()
            self.logger(f'[ContactLineGapTask] auto_gamma_correction failed: {error_msg}', level=LogLevel.ERROR)
            self.logger(f'[ContactLineGapTask] auto_gamma_correction fail {e}', level=LogLevel.INFO)
            return 0

        try:
            contours, resized_img = self.find_fringe_boundary(img_gamma)
        except Exception as e:
            error_msg = traceback.format_exc()
            self.logger(f'[ContactLineGapTask] find_fringe_boundary failed: {error_msg}', level=LogLevel.ERROR)
            self.logger(f'[ContactLineGapTask] find_fringe_boundary failed {e}', level=LogLevel.INFO)
            return 0

        try:
            gap_flag = self.extract_candidates(contours, resized_img, gamma, thd)
        except Exception as e:
            error_msg = traceback.format_exc()
            self.logger(f'[ContactLineGapTask] extract_candidates failed: {error_msg}', level=LogLevel.ERROR)
            self.logger(f'[ContactLineGapTask] extract_candidates failed {e}', level=LogLevel.INFO)
            
            self.records = ContactLineGapOutput(gap_flag=-1, status=False, error_message=str(f'[ContactLineGapTask] check_contact_line_gap failed: {e}'))
            return 0

        self.records = ContactLineGapOutput(gap_flag=gap_flag, status=True)
        self.logger(f'[ContactLineGapTask] Execute end', level=LogLevel.INFO)
        return 1

    def GetExecuteResult(self)->memoryview:
        ret_value = json.dumps(asdict(self.records), ensure_ascii=False)
        data_bytes = ret_value.encode('utf-8') 
        return memoryview(data_bytes)

if __name__ == "__main__":


    exetask = ContactLineGapTask()
    print('testExecute:'+exetask.GetTaskName())
    testAC = ContactLineGapInput(
        width_image = 4112,
        height_image = 3008,
        threshold = 5.8
    )
    parameter = json.dumps(asdict(testAC), ensure_ascii=False)

    
    folder = './checkimages'
    files = [f for f in os.listdir(folder) if f.lower().endswith('png')]
    paths = [os.path.join(folder, f) for f in files]
    paths.sort(key=lambda fp: os.path.getmtime(fp))
    
    
    for img_path in paths:
        print("="*20, img_path)
        vmemdata = []
        bmpimg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        bmpimg = bmpimg.tobytes()
        mv = memoryview(bmpimg)
        mmdata = BaseExecuteTask.MemData(mv)
        vmemdata.append(mmdata)

        exetask.Execute(parameter, vmemdata)

        json_str= exetask.GetExecuteResult()
        data_bytes = bytes(json_str).decode()
        json_dict = json.loads(data_bytes)
        record = ContactLineGapOutput(**json_dict)

        print(f"ContactLineGapOutput: {record}")
        print(f"gap_flag: {record.gap_flag}")
        print(f"Status: {record.status}")
        print(f"Error Message: {record.error_message}")