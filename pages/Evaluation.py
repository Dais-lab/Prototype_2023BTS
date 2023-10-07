import streamlit as st
import os
from natsort import natsorted
from modules.preprocess.preprocessing import *
from modules.inference.inference import *
from modules.inference.eval import *
import torch



image_list = os.listdir("/app/temp/image")
image_list = image_list[:5]

col1, col2, col3 = st.columns([1, 1, 1])
image_category = col1.selectbox("Select Image Category", options=image_category)
preprocess = col1.selectbox("Select Preprocess", options=["None", "Normalize", "CLAHE", "EqualizeHist"])
model = col1.selectbox("Select Model", options=model_list)
select_inference_type = col1.selectbox("Select_inference_type", options=["inference_tensor", "inference_torch"])