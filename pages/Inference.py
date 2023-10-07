import streamlit as st
import os
from natsort import natsorted
from modules.preprocess.preprocessing import *

"""
## Inference Image

"""
image_list = os.listdir("/app/temp/image")
if len(image_list) == 0:
    raise FileNotFoundError("업로드된 이미지가 없습니다.")
image_category = natsorted(image_list)

model_list = os.listdir("/app/models")
if len(model_list) == 0:
    raise FileNotFoundError("학습된 모델이 없습니다.")
model_list = natsorted(model_list)




col1, col2, col3 = st.columns([1, 1, 1])
image_category = col1.selectbox("Select Image Category", options=image_category)
preprocess = col1.selectbox("Select Preprocess", options=["None", "Normalize", "CLAHE", "EqualizeHist"])
model = col1.selectbox("Select Model", options=model_list)
inference_button = col1.button("Inference")

if inference_button:
    IMAGE = ImageContainer("/app/temp/image/" + image_category)
    IMAGE.preprocess(preprocess)
    for i in range(len(IMAGE)):
        col2.image(IMAGE[i])
        
    IMAGE.inference("/app/models/" + model)
    for i in range(len(IMAGE)):
        col3.image(IMAGE.test_masks[i])
        
    
    
    
        

