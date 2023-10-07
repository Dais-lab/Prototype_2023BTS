import streamlit as st
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def state_init():
    if "flaw_type" not in st.session_state:
        st.session_state.flaw_type = ""
    if "padding" not in st.session_state:
        st.session_state.padding = ""
    if "fade" not in st.session_state:
        st.session_state.fade = ""
    if "CT_margin" not in st.session_state:
        st.session_state.CT_margin = 50
    if "sigma" not in st.session_state:
        st.session_state.sigma = ""
    if "points" not in st.session_state:
        st.session_state.points = ""
    if "normalize" not in st.session_state:
        st.session_state.normalize = ""
    if "try_count" not in st.session_state:
        st.session_state.try_count = ""
    if "strength" not in st.session_state:
        st.session_state.strength = ""
    if "aug_params" not in st.session_state:
        st.session_state.aug_params = ""
def set_config():
    """
    Streamlit 기본 설정.
    """
    st.set_page_config(
        page_title="현대RB 프로젝트",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
        'About': "제작자 : 이창현,  https://www.notion.so/dns05018/L-Hyun-s-Portfolio-f1c904bf9f2445fb96909da6eb3d450d?pvs=4"
    }
    )
    state_init()
    
        
