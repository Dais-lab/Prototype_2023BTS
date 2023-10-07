import streamlit as st
import os
from datetime import datetime
import shutil
"""
## Image Preview

"""

files = st.file_uploader("Upload images. over 300 images", accept_multiple_files=True)
if files is not None:
    if len(files) > 0:
        save_img_path = f"/app/temp/image/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(save_img_path, exist_ok=True)
        for file in files:
            with open(os.path.join(save_img_path, file.name), "wb") as f:
                f.write(file.getbuffer())
                
            

delete_text = st.text_input("", value="", placeholder="데이터를 삭제 하려면 '데이터 삭제'를 입력하세요.")
if delete_text == "데이터 삭제":
    shutil.rmtree("/app/temp")
    os.makedirs("/app/temp", exist_ok=True)
    
    st.success("데이터 삭제 완료")
