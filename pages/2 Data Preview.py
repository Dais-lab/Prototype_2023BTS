import streamlit as st
import os
from natsort import natsorted
from modules.preprocess.preprocessing import *

"""
## Image Preview

#### 10개의 이미지를 미리보기 합니다.

"""
image_list = os.listdir("/app/temp/image")
if len(image_list) == 0:
    raise FileNotFoundError("업로드된 이미지가 없습니다.")
image_category = natsorted(image_list)
col1, col2 = st.columns(2)
image_category = col1.selectbox("Select Image Category", options=image_category)
images = natsorted(os.listdir(f"/app/temp/image/{image_category}"))

IMAGE = ImageContainer(f"/app/temp/image/{image_category}")
preprocess = col2.selectbox("Select Preprocess", options=["None", "Normalize", "CLAHE", "EqualizeHist"])
IMAGE.preprocess(preprocess)

IMAGE.find_IQI()
for i in range(10):
    col1,col2,col3= st.columns([1, 1, 3])
    col1.image(IMAGE[i], width=400)
    col3.write(IMAGE.image_path_list[i].split("/")[-1])
    
    if i in IMAGE.IQI_list:
        col3.write("IQI : True")
    else:
        col3.write("IQI : False")
    
    if i < IMAGE.IQI_list[0]:
        col3.write("End Tab : True")
    elif i > IMAGE.IQI_list[1]:
        col3.write("End Tab : True")
    else:
        col3.write("End Tab : False")
        
    
