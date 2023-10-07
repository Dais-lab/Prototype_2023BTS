import streamlit as st
import os
from natsort import natsorted
from modules.preprocess.preprocessing import *
from modules.augmentation.Generate_VF import *
import extra_streamlit_components as stx


"""
## Data Augmentation
"""
def state_init():
    if "flaw_type" not in st.session_state:
        st.session_state.flaw_type = ""
    if "padding" not in st.session_state:
        st.session_state.padding = ""
    if "fade" not in st.session_state:
        st.session_state.fade = ""
    if "seed" not in st.session_state:
        st.session_state.seed = ""
    if "CT_margin" not in st.session_state:
        st.session_state.CT_margin = 50
    if "sigma" not in st.session_state:
        st.session_state.sigma = ""
    if "points" not in st.session_state:
        st.session_state.points = ""
    if "normalize" not in st.session_state:
        st.session_state.normalize = ""
        
state_init()
val = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="", description="Set Parameter"),
    stx.TabBarItemData(id=2, title="", description="Preview & Download")], default=1, key="augmentation")
if val == "1":
    col1, col2, col3 = st.columns(3)
    col1.write("## Set Parameter")
    st.session_state.flaw_type = col1.selectbox("Select flaw type.", options=["PO", "Scratch", "Leftover", "CT", "IP"])
    st.session_state.padding = col1.number_input("Padding", value=-30)
    st.session_state.fade = col1.number_input("Fade", value=50)
    st.session_state.seed = col1.number_input("Seed", value=42)
    if st.session_state.flaw_type == "CT":
        st.session_state.CT_margin = col1.number_input("CT margin", value=50)
    st.session_state.sigma = col1.number_input("Elastic Deform - Sigma", value=5)
    st.session_state.points = col1.number_input("Elastic Deform - Points", value=4)
    st.session_state.normalize = col1.selectbox("Normalize", options=[True, False])
    
    col2.write("## Parameter summary")
    params = f"""
    Virtual Flaw Augmentation Parameters
    flaw type : {st.session_state.flaw_type}
    padding : {st.session_state.padding}
    fade : {st.session_state.fade}
    seed : {st.session_state.seed}
    sigma : {st.session_state.sigma}
    points : {st.session_state.points}
    normalize : {st.session_state.normalize}
    """
    col2.code(params)
    
if val == "2":
    col1, col2, col3 = st.columns(3)
    col1.write("## Select Image Category")
    
    image_list = os.listdir("/app/temp/image")
    image_category = natsorted(image_list)
    if len(image_list) == 0:
        raise FileNotFoundError("업로드된 이미지가 없습니다.")
    image_category = col1.selectbox("Select Image Category", options=image_category)
    col1.write("## Parameter summary")
    params = f"""
    Virtual Flaw Augmentation Parameters
    flaw type : {st.session_state.flaw_type}
    padding : {st.session_state.padding}
    fade : {st.session_state.fade}
    seed : {st.session_state.seed}
    sigma : {st.session_state.sigma}
    points : {st.session_state.points}
    normalize : {st.session_state.normalize}
    """
    col1.code(params)
        
    images = natsorted(os.listdir(f"/app/temp/image/{image_category}"))
    IMAGE = ImageContainer(f"/app/temp/image/{image_category}")
    for i in range(3,6):
        generate_virtual_flaw(
            IMAGE.image_path_list[i],
            flaw_type=st.session_state.flaw_type,
            padding=st.session_state.padding,
            fade=st.session_state.fade,
            seed=st.session_state.seed,
            CT_margin=st.session_state.CT_margin,
            sigma=st.session_state.sigma,
            points=st.session_state.points,
            normalize=st.session_state.normalize
        )
        col2.image(IMAGE[i], width=400)
        
    
    
        