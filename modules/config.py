import streamlit as st


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
        