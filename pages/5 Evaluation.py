import streamlit as st
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from natsort import natsorted
from modules.preprocess.preprocessing import *
from modules.augmentation.Generate_VF import *
from modules.models.model import *
from modules.supervised_learning import supervised
import extra_streamlit_components as stx
import time
import zipfile
import io
import json
import sys
from glob import glob
import pandas as pd
from datetime import datetime
"""
## Evaluate Model
"""
message = st.empty()

val = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="", description="Select Images and Select Model"),
    stx.TabBarItemData(id=2, title="", description="Evaluate Classification Model"),
    stx.TabBarItemData(id=3, title="", description="Evaluate Segmentation Model(Not available)"),], default=1, key="train")

if val == "1":
    col1, col2 = st.columns(2)
    model_list = glob("/app/temp/log/**/*.h5", recursive=True)
    if len(model_list) == 0:
        raise FileNotFoundError("학습된 모델이 없습니다.")
    model_list = natsorted(model_list)
    col1.write("### Select Model")
    st.session_state.model = col1.selectbox("Select Model", options=model_list)
    date = st.session_state.model.split("/")[-3]
    config = json.load(open(f"/app/temp/log/{date}/config.json"))
    col1.write("### Model Train Info")
    col1.code(json.dumps(config, indent=4))
    
    
    image_list = os.listdir("/app/temp/image")
    if len(image_list) == 0:
        raise FileNotFoundError("업로드된 이미지가 없습니다.")
    image_category = natsorted(image_list, reverse=True)
    
    col2.write("### Select Images")
    st.session_state.eval_data = col2.selectbox("Select Image Category for evaluate model. That must have test folder.", options=image_category)
    test_0_dir = f"/app/temp/image/{st.session_state.eval_data}/test/0"
    test_1_dir = f"/app/temp/image/{st.session_state.eval_data}/test/1"
    if not os.path.exists(test_0_dir):
        raise FileNotFoundError("Test 폴더가 존재하지 않습니다.")
    
    col2.write("### Evaluate Data Info")
    col2.code(f"""
    Image Category : {st.session_state.eval_data}
    
    Test 0 Image Count : {len(os.listdir(test_0_dir))}
    
    Test 1 Image Count : {len(os.listdir(test_1_dir))}
    
    """)
    
    
    
    
    

    
if val == "2":
    message = st.empty()
    MAX_LOG_COUNT = 1
    log_queue = []
    class StreamlitStdout:
        def write(self, text):
            # 로그 큐에 새로운 로그 추가
            log_queue.append(text.replace("", ""))
            # 최근 로그가 MAX_LOG_COUNT를 초과하면 가장 오래된 로그 삭제
            if len(log_queue) > MAX_LOG_COUNT:
                log_queue.pop(0)
            # 최근 로그 출력
            message.info('\n'.join(log_queue))
        
        def flush(self):
            pass  # flush 메서드를 추가하여도 되지만, 필요하지 않을 수 있습니다.

    # sys.stdout을 StreamlitStdout 클래스로 대체
    sys.stdout = StreamlitStdout()
    if st.session_state.model == "":
        raise FileNotFoundError("모델을 선택해주세요.")
    if st.session_state.eval_data == "":
        raise FileNotFoundError("평가할 데이터를 선택해주세요.")
    
    message.info("Evaluating...")
    model = st.session_state.model
    date = model.split("/")[-3]
    config = json.load(open(f"/app/temp/log/{date}/config.json"))
    config["DATE"] = datetime.now().strftime("%Y%m%d%H%M")
    config["MODEL_PATH"] = model
    config["TEST_DATA_PATH"] = f"/app/temp/image/{st.session_state.eval_data}/test"
    config["MODE"] = "test"
    Experiment = supervised.Experiment_Model(**config)
    Experiment.test_model()
    Experiment.report_model()
    message.success("Evaluate Finished!")
    
    col1, col2 = st.columns(2)
    col1.write("### Predict Value")
    df = pd.read_csv(f"/app/temp/log/{config['DATE']}/result/test_result.csv")
    col1.dataframe(df)
    col2.write("### Classification Report")
    with open(f"/app/temp/log/{config['DATE']}/result/classification_report.txt", "r") as f:
        col2.code(f.read())
        
    col1_, col2_ = st.columns(2)
    image_list = glob(f"/app/temp/log/{config['DATE']}/result/*.png")
    image_list = natsorted(image_list)
    
    for image, index in zip(image_list, range(len(image_list))):
        if index % 2 == 0:
            col1_.image(image)
        else:
            col2_.image(image)
        
    
    
        
                
                
            
    
        

         
    
    
        