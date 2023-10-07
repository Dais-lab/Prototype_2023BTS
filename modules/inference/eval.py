import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import keras
import os
import json

def inference_result_tensor(main_json, model_path, X_test):
    model = tf.keras.models.load_model(model_path)

    for num in range(len(X_test)):
        y_prediction = model.predict(X_test[num].reshape(-1, 512, 512))
        main_json[list(main_json.keys())[num]]["predict_score"] = str(y_prediction[0][0])
    
    return main_json

def inference_result_torch(main_json, model_path, X_test):
    model = torch.load(model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for num, x in enumerate(X_test):
        x = x.unsqueeze(0).to(device)
        with torch.no_grad(): 
            y_prediction = model(x)  
    
        main_json[list(main_json.keys())[num]]["predict_score"] = str(y_prediction.cpu().numpy()[0][0])

def save_inference_result(save_path, main_json):
    with open(save_path, "w") as f:
        json.dump(main_json, f, ensure_ascii = False, indent = 4)

