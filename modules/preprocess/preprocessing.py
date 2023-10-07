import cv2
import imgaug.augmenters as iaa
import numpy as np
from glob import glob
from natsort import natsorted
import os
#CPU 사용
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import torch
from modules.augmentation.Generate_VF import *

def check_IQI(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=21)
    kernel1 = np.ones((5,1), np.uint8)
    kernel2 = np.ones((1,5), np.uint8)


    # 그라디언트 크기 계산
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_x ** 2)

    # 경계를 0과 1로 이루어진 이진 마스크로 변환
    binary_mask = np.uint8(gradient_magnitude > np.percentile(gradient_magnitude, 93))

    #팽창
    binary_mask = cv2.dilate(binary_mask, kernel2, iterations=3)

    #특정 크기 이하의 덩어리를 제거
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000:
            cv2.drawContours(binary_mask, [cnt], -1, 0, -1)


    #침식
    binary_mask = cv2.erode(binary_mask, kernel2, iterations=3)
    binary_mask = cv2.erode(binary_mask, kernel1, iterations=1)

    #특정 크기 이하의 덩어리를 제거
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            cv2.drawContours(binary_mask, [cnt], -1, 0, -1)

    binary_mask = cv2.dilate(binary_mask, kernel1, iterations=2)
    binary_mask = cv2.dilate(binary_mask, kernel2, iterations=2)

    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def find_IQI(image_list):
    check_IQI_list = []
    found_iqi_break = False
    for i in range(len(image_list)):
        image = image_list[i]
        if check_IQI(image) <= 1:
            image = image_list[i+1]
            if check_IQI(image) <= 1:
                check_IQI_list.append(i-1)
                found_iqi_break = True
                break
    if found_iqi_break:
        for i in range(len(image_list)-1, 0, -1):
            image = image_list[i]

            if check_IQI(image) <= 1:
                image = image_list[i-1]
                if check_IQI(image) <= 1:
                    check_IQI_list.append(i+1)
                    break

    return check_IQI_list


class ImageContainer:
    def __init__(self, image_path):
        self.image_path_list = natsorted(glob(os.path.join(image_path, "*")))
        self.image_list = [cv2.imread(image_path) for image_path in self.image_path_list]
        self.image_list = [cv2.resize(image, (512, 512)) for image in self.image_list]
        #self.image_list = [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in self.image_list]
        
    def find_IQI(self):
        self.IQI_list = find_IQI(self.image_list)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = self.image_list[idx]
        return image
    
    def __iter__(self):
        for image in self.image_list:
            yield image
            
    def __repr__(self):
        return f"ImageContainer(image_path={self.image_list})"
    
    def __str__(self):
        return f"ImageContainer(image_path={self.image_list})"
    
    def __add__(self, other):
        return ImageContainer(self.image_list + other.image_list)
    
    def preprocess(self, preprocess):
        if preprocess == "Normalize":
            self.image_list = [cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) for image in self.image_list]
        elif preprocess == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            self.image_list = [clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) for image in self.image_list]
        elif preprocess == "EqualizeHist":
            self.image_list = [cv2.equalizeHist(image) for image in self.image_list]

  
    def inference_tensorflow(self, model_path):
        self.mask_list = []
        self.model = tf.keras.models.load_model(model_path)
        self.test_images = self.image_list.copy()
        self.test_masks = [self.model.predict(image[tf.newaxis, ...], verbose=0)[0] for image in self.test_images]
        self.test_masks = [(self.test_masks[i] > 0.3).astype(np.uint8) for i in range(len(self.test_masks))]
        self.test_masks = [cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX) for mask in self.test_masks]

    # def inference_pytorch(self, model_path):
    #     self.mask_list = []
    #     self.model = torch.load(model_path)
    #     self.model.eval()
    #     self.test_images = [transform(image) for image in self.image_list]
    #     self.test_masks = [self.model(image.unsqueeze(0)) for image in self.test_images]
    #     self.test_masks = [(self.test_masks[i] > 0.3).astype(np.uint8) for i in range(len(self.test_masks))]
    #     self.test_masks = [cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX) for mask in self.test_masks]
    
        
    def augmentation(self, padding, fade, flaw_type, sigma, points, normalize, CT_margin, try_count, strength):
        self.augmented_image_list = []
        self.augmented_mask_list = []
        for image in self.image_path_list:
            augmented_image, augmented_mask = generate_virtual_flaw(
                image,
                flaw_type=flaw_type,
                padding=padding,
                fade=fade,
                CT_margin=CT_margin,
                sigma=sigma,
                points=points,
                normalize=normalize,
                try_count=try_count,
                strength=strength
            )
            self.augmented_image_list.append(augmented_image)
            self.augmented_mask_list.append(augmented_mask)
        
        
        
        
        
        