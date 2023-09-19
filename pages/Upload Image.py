import streamlit as st
import os
from natsort import natsorted
import cv2
from glob import glob
"""
## Image Preview

"""

files = st.file_uploader("Upload images. over 300 images", accept_multiple_files=True)
if files is not None:
    save_img_path = "/app/temp/image"
    for file in files:
        with open(os.path.join(save_img_path, file.name), "wb") as f:
            f.write(file.getbuffer())
            
            
