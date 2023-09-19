import cv2
import imgaug.augmenters as iaa
import numpy as np


def preprocess_img(img, preprocessing, target_size):
    if preprocessing == 'median_blur':
        img = cv2.medianBlur(img, ksize = 3)
    elif preprocessing == 'noise_drop':
        img = iaa.Dropout(p=(0, 0.2))(images = img).astype("uint8")
    elif preprocessing == 'his_equalized':
        img = cv2.equalizeHist(img)
    elif preprocessing == 'sobel_masking_y':
        img = cv2.Sobel(img, -1, 0, 1, delta = 128)
    elif preprocessing == 'scharr':
        img = cv2.Scharr(img, -1, 0, 1, delta=128)
    elif preprocessing == 'clahe':
        img = cv2.convertScaleAbs(img)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
        img = clahe.apply(img)
    elif preprocessing == 'normalization':
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    else:
        img = img
    return img

def get_weld_image(img, padding = 10):
    """용접 이미지에서 용접 부위만 잘라내는 함수
    
    Args:
        img (np.array): 용접 이미지
        padding (int, optional): 용접 부위 패딩. Defaults to 10.

    Returns:
        dst: 용접 부위만 잘라낸 이미지, 여백 부분은 255로 채움
    """
    # 이미지를 읽어 spatial_normalization 후 resize, rotate.
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    image = np.array(image, dtype=np.float32)
    
    # 이미지의 x축을 기준으로 평균값을 구함
    mean_list = []
    for i in range(img.shape[0]):
        mean_list.append(np.mean(image[:,i]))
        
    # 평균값의 gradient를 구함
    image = np.gradient(np.squeeze(mean_list))
    y1, y2 = img.shape[0] - int(np.argmax(image)) - padding, img.shape[0] - int(np.argmin(image)) + padding
    # 용접 부위를 잘라냄
    result = img[y1:y2, :]
    
    dst = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    dst[int((img.shape[0] - result.shape[0]) / 2):int((img.shape[0] + result.shape[0]) / 2), :] = result

    return dst