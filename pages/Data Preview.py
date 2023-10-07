import streamlit as st
import os
from natsort import natsorted
from modules.preprocess.preprocessing import *

"""
## Image Preview

"""

image_category = natsorted(os.listdir("/app/temp/image"))
col1, col2 = st.columns(2)
image_category = col1.selectbox("Select Image Category", options=image_category)
images = natsorted(os.listdir(f"/app/temp/image/{image_category}"))

IMAGE = ImageContainer(f"/app/temp/image/{image_category}")
preprocess = col2.selectbox("Select Preprocess", options=["None", "Normalize", "CLAHE", "EqualizeHist"])
IMAGE.preprocess(preprocess)

IMAGE.find_IQI()
for i in range(len(IMAGE)):
    col1,col2,col3= st.columns([1, 1, 3])
    col1.image(IMAGE[i], width=400)
    col3.write(IMAGE.image_path_list[i])
    
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
        
    
