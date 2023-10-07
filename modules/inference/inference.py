import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def make_gradcam_heatmap(img_array, grad_model):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)


    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()[0][0]

def save_and_display_gradcam(img, heatmap, cam_path, alpha=0.01):
    # Load the original image
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.colormaps.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)
    
    
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    superimposed_img = cv2.cvtColor(np.float32(superimposed_img), cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    concat_img = np.concatenate((img, superimposed_img), axis=1)
    cv2.imwrite(cam_path, concat_img)


def make_gradcam_heatmap(img_array, model, layer_name):
    """
    img_array: 입력 이미지 배열 (PyTorch tensor)
    model: 사용하려는 PyTorch 모델
    layer_name: 역전파할 레이어의 이름 (문자열)
    """
    grad_model = model.eval()

    # Hook the layer to get gradients and features
    features = []
    gradients = []
    
    def forward_hook(module, input, output):
        features.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    layer = dict([module for module in grad_model.named_modules()])[layer_name]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)
    
    # Forward pass
    preds = model(img_array)
    score = preds[:, 0]
    
    # Backward pass
    score.backward()
    
    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])
    heatmap = torch.matmul(features[0].squeeze(0).permute(1, 2, 0), pooled_grads)
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap.detach().numpy(), preds.detach().numpy()[0][0]

def save_and_display_gradcam(img, heatmap, cam_path, alpha=0.4):
    """
    img: 원본 이미지 (PIL 이미지 혹은 NumPy 배열)
    heatmap: 생성된 heatmap (NumPy 배열)
    cam_path: 저장 경로 (문자열)
    alpha: heatmap의 투명도 (0~1 사이의 스칼라)
    """
    # 원본 이미지를 [0, 255]의 NumPy 배열로 변환
    if not isinstance(img, np.ndarray):
        img = TF.to_tensor(img).permute(1, 2, 0).numpy() * 255
    
    # Heatmap 처리
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 원본 이미지와 heatmap 결합
    superimposed_img = heatmap * alpha + img * (1-alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # 이미지를 결합하고 저장
    concat_img = np.concatenate((img, superimposed_img), axis=1)
    cv2.imwrite(cam_path, concat_img)