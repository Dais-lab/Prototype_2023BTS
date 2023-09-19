import streamlit as st
import os
from natsort import natsorted
import cv2
import time

"""
## Train Model. (In progress)

"""

col1, col2 = st.columns(2)

col1.write("하이퍼 파라미터 설정")
train_init = col1.text_input("Init", value=time.strftime("%Y%m%d%H%M%S", time.localtime()))
seed = col1.number_input("Seed", value=72)
target_size = col1.selectbox("Target Size", options=["[512, 512, 1]", "[1024, 1024, 1]"])
weld_type = col1.selectbox("Weld Type", options=["True", "False"])
model_name = col1.selectbox("Model", options=["CNN", "ResNet", "VGG16"])
epoch = col1.slider("Epoch", min_value=1, max_value=500, value=10)
batch_size = col1.slider("Batch Size", min_value=4, max_value=128, value=4)
learning_rate = col1.number_input("Learning Rate", value=0.0001, step=0.00001)
weight_decay = col1.number_input("Weight Decay", value=0.9, step=0.00001)
preprocessing = col1.selectbox("Preprocessing", options=["None", "Histogram Equalization", "CLAHE", "Normalize"])

col2.write("Summary")
code = f"""
Train Init : {train_init}
Seed : {seed}
Target Size : {target_size}
Weld Type : {weld_type}
Model : {model_name}
Epoch : {epoch}
Batch Size : {batch_size}
Preprocessing : {preprocessing}
Learning Rate : {learning_rate}
Weight Decay : {weight_decay}
"""
col2.code(code, language="json")

