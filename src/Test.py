import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import shutil
from natsort import natsorted


from modules.preprocess import preprocessing as pp
from modules.test import test as tt

"""
## Test Model. (In progress)



"""

col1, col2 = st.columns(2)
col1.write("Model Setting")
model_chkpt = col1.code("Checkpoint : single_channel_single_validation_CNN_weld_only_normalization_CNN.h5", language="json")
target_size = col1.selectbox("Target Size", options=["[512, 512, 1]", "[1024, 1024, 1]"])
if target_size == "[512, 512, 1]":
    target_size = [512, 512, 1]
else:
    target_size = [1024, 1024, 1]
weld_type = col1.selectbox("Weld Type", options=["True", "False"])
if weld_type == "False":
    preprocessing = col1.selectbox("Preprocessing", options=["None", "Histogram Equalization", "CLAHE", "Normalize"])
else:
    preprocessing = None
files = col1.file_uploader("Upload test.zip. over 300 images", accept_multiple_files=True)

col2.write("Summary")
code = f"""
Model Checkpoint : {"single_channel_single_validation_CNN_weld_only_normalization_CNN"}
Target Size : {target_size}
Weld Type : {weld_type}
Preprocessing : {preprocessing}
"""
col2.code(code, language="json")


if files is not None:
    #save images
    save_img_path = "/app/data/User/image"
    save_zip_path = "/app/data/User/zip"
    
    for file in files:
        if file.type == "image/jpeg":
            with open(os.path.join(save_img_path, file.name), "wb") as f:
                f.write(file.getbuffer())
        elif file.type == "application/zip":
            with open(os.path.join(save_zip_path, file.name), "wb") as f:
                f.write(file.getbuffer())
            
            #unzip
            os.system(f"unzip {os.path.join(save_zip_path, file.name)} -d {save_img_path}")
            

st.write("")
col1_1, col1_2, col1_3, col1_4, col1_5, col1_6, col1_7, col1_8 = st.columns(8)
if col1_8.button("Test"):
    shutil.rmtree("/app/temp")
    os.mkdir("/app/temp")
    message = st.empty()
    images_count = len(os.listdir(save_img_path))
    if images_count == 0:
        message.error("Please upload images.")
    else:
        message.info("Test Start.")
        test_image_list, test_image_label, test_image = tt.load_test("/app/data/User/image/test", target_size, weld_type = True)
        message.info("Test Image Loaded.")
        model = tf.keras.models.load_model("/app/models/CNN.h5")
        message.info("Model Loaded.")
        tt.test_model(model, test_image_list, test_image_label, test_image)
        message.success("Test Done.")
        
    
    result_df = pd.read_csv("/app/temp/test_result.csv")
    hit_rate_df = pd.read_csv("/app/temp/hit_evaluation_result.csv")
    col1, col2 = st.columns(2)
    col1.dataframe(result_df)
    col2.dataframe(hit_rate_df)
    st.image("/app/temp/hit_ratio.png")
        
        
        
    
    



            
            
    
