import streamlit as st
import os
import shutil
import time

"""
## Remove Data.

업로드된 모든 데이터를 삭제합니다.

삭제하려면 "데이터 삭제"를 입력하세요.

"""

delete_text = st.text_input("", value="")
if delete_text == "데이터 삭제":
    shutil.rmtree("/app/data/User")
    os.makedirs("/app/data/User/image", exist_ok=True)
    os.makedirs("/app/data/User/zip", exist_ok=True)
    shutil.rmtree("/app/temp")
    os.makedirs("/app/temp/CAM", exist_ok=True)
    st.success("데이터 삭제 완료")
    time.sleep(5)
