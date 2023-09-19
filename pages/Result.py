import os

import streamlit as st
import shutil
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from natsort import natsorted
import cv2

from glob import glob

"""
## Result.

"""
def get_pred(df, image):
    pred = df[df["path"] == image]["predict"].values[0]
    return pred

col1, col2, col3 = st.columns(3)
try:
    test_result_df = pd.read_csv("/app/temp/test_result.csv")
    test_result_df = test_result_df.reset_index(drop=True)
    col1.dataframe(test_result_df, height=500)
    
except:
    st.error("Please inference first.")

try:
    hit_ratio_df = pd.read_csv("/app/temp/hit_evaluation_result.csv")
    col2.dataframe(hit_ratio_df, height=500)
    select_threshold = col3.slider("Select Threshold", min_value=0.1, max_value=float(1), value=0.1, step=0.001, format="%.3f")
    df_row = hit_ratio_df[hit_ratio_df["threshold"] == select_threshold]
    total_image = df_row["TN"].values[0]+df_row["TP"].values[0]+df_row["FN"].values[0]+df_row["FP"].values[0]
    predict_image = df_row["TP"].values[0]+df_row["FP"].values[0]
    col3.write(f"불량으로 판정된 이미지 : {predict_image} / {total_image}, 비율 : {predict_image/total_image*100:.2f}%")
    col3.image("/app/temp/hit_ratio.png")
except:
    col2.write("hit_evaluation_result.csv 파일이 없습니다.")
    col3.write("hit_ratio.png 파일이 없습니다. 레이블이 없는 데이터셋인 경우 생성되지 않습니다.")


col_1, col_2 = st.columns(2)

images = os.listdir(f"/app/temp/CAM")
images = natsorted(images)
images_list = []
for i in range(len(images)):
    image = cv2.imread(f"/app/temp/CAM/{images[i]}", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_list.append(image)
    
for image, i, limit in zip(images, range(len(images_list)), range(300)):
    if i == limit:
        break
    
    
    if i % 2 == 0:
        col_1.write(f"이미지 : {image}")
        col_1.write(f"예측 : {get_pred(test_result_df, image)}")
        col_1.image(images_list[i])
    elif i % 2 == 1:
        col_2.write(f"이미지 : {image}")
        col_2.write(f"예측 : {get_pred(test_result_df, image)}")
        col_2.image(images_list[i])