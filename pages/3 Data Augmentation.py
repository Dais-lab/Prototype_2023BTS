import streamlit as st
import os
from natsort import natsorted
from modules.preprocess.preprocessing import *
from modules.augmentation.Generate_VF import *
import extra_streamlit_components as stx
import time
import zipfile
import io
"""
## Data Augmentation - Virtual Flaw
"""
message = st.empty()

val = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="", description="Set Parameter"),
    stx.TabBarItemData(id=2, title="", description="Preview & Download")], default=1, key="augmentation")
if val == "1":
    col1, col2 = st.columns(2)
    col1.write("## Set Parameter")
    st.session_state.flaw_type = col1.selectbox("Select flaw type.", options=["PO", "Scratch", "Leftover", "CT", "IP", "Random"])
    st.session_state.padding = col1.number_input("Padding", value=-30)
    st.session_state.fade = col1.number_input("Fade", value=50)
    if st.session_state.flaw_type == "CT":
        st.session_state.CT_margin = col1.number_input("CT margin", value=50)
    st.session_state.sigma = col1.number_input("Elastic Deform - Sigma", value=5)
    st.session_state.points = col1.number_input("Elastic Deform - Points", value=4) 
    st.session_state.try_count = col1.number_input("Try count", value=2, min_value=2, max_value=10)
    st.session_state.strength = round(float(col1.number_input("Strength", value=1.0, min_value=0.1, max_value=2.0, step=0.1)), 1)
    st.session_state.normalize = col1.selectbox("Normalize", options=[True, False])
    col2.write("## Parameter summary")
    st.session_state.aug_params = f"""
    Virtual Flaw Augmentation Parameters

    flaw type : {st.session_state.flaw_type}
    
    padding : {st.session_state.padding}
    
    fade : {st.session_state.fade}
    
    sigma : {st.session_state.sigma}
    
    points : {st.session_state.points}
    
    try count : {st.session_state.try_count}
    
    strength : {st.session_state.strength}
    
    normalize : {st.session_state.normalize}
    
    """
    col2.code(st.session_state.aug_params)
    
if val == "2":
    col1, col2 = st.columns(2)
    col1.write("### Select Image Category")
    image_list = os.listdir("/app/temp/image")
    image_category = natsorted(image_list)
    if len(image_list) == 0:
        raise FileNotFoundError("업로드된 이미지가 없습니다.")
    image_category = col1.selectbox("Select Image Category", options=image_category)
    col2.write("### Parameter summary")
    col2.code(st.session_state.aug_params)
    col3_col1, col3_col2 = col1.columns(2)
    if col3_col1.button("Augmentation"):
        images = natsorted(os.listdir(f"/app/temp/image/{image_category}"))
        IMAGE = ImageContainer(f"/app/temp/image/{image_category}")
        message.info("Augmenting...")
        IMAGE.augmentation(
            padding=st.session_state.padding, 
            fade=st.session_state.fade, 
            flaw_type=st.session_state.flaw_type, 
            sigma=st.session_state.sigma, 
            points=st.session_state.points, 
            normalize=st.session_state.normalize, 
            CT_margin=st.session_state.CT_margin, 
            try_count=st.session_state.try_count, 
            strength=st.session_state.strength
            )
        st_col1, st_col2, st_col3 = st.columns(3)
        for i in range(len(IMAGE)):
            st_col1.image(IMAGE[i])
            st_col2.image(IMAGE.augmented_image_list[i])
            st_col3.image(IMAGE.augmented_mask_list[i])
        message.info("Saving...")
        os.makedirs(f"/app/temp/image/{image_category}_augmented", exist_ok=True)
        for i in range(len(IMAGE)):
            cv2.imwrite(f"/app/temp/image/{image_category}_augmented/{images[i]}", IMAGE.augmented_image_list[i])
            cv2.imwrite(f"/app/temp/image/{image_category}_augmented/{images[i][:-4]}_mask.png", IMAGE.augmented_mask_list[i])
        message.info("Preparing for download...")
        os.makedirs(f"/app/temp/download", exist_ok=True)
        zip_file = zipfile.ZipFile(f"/app/temp/download/{image_category}_augmented.zip", "w")
        for root, dirs, files in os.walk(f"/app/temp/image/{image_category}_augmented"):
            for file in files:
                zip_file.write(os.path.join(root, file))
        zip_file.close()
        message.success("Download ready!")
        with open(f"/app/temp/download/{image_category}_augmented.zip", "rb") as f:
            data = io.BytesIO(f.read())
        col3_col2.download_button(label="Download", data=data, file_name=f"{image_category}_augmented.zip", mime="application/zip")
            
            
        
            
    
        

         
    
    
        