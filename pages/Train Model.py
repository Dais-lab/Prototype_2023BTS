import streamlit as st
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from natsort import natsorted
from modules.preprocess.preprocessing import *
from modules.augmentation.Generate_VF import *
from modules.models.model import *
from modules.supervised_learning import supervised
# TensorFlow 로그 레벨 설정 (예: INFO 레벨 이상만 표시)
import extra_streamlit_components as stx
import time
import zipfile
import io
import json
import sys
from glob import glob
import pandas as pd
"""
## Train Model
"""
message = st.empty()

val = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="", description="Select Images and Set Hypeparameter"),
    stx.TabBarItemData(id=2, title="", description="Train Model")], default=1, key="train")
if val == "1":
    image_list = os.listdir("/app/temp/image")
    if len(image_list) == 0:
        raise FileNotFoundError("업로드된 이미지가 없습니다.")
    image_category = natsorted(image_list, reverse=True)
    col1, col2 = st.columns(2)
    col1.write("### Select Images")
    image_category = col1.selectbox("Select Image Category for train model. That must have train and test folder.", options=image_category)
    train_0_dir = f"/app/temp/image/{image_category}/train/0"
    train_1_dir = f"/app/temp/image/{image_category}/train/1"
    test_0_dir = f"/app/temp/image/{image_category}/test/0"
    test_1_dir = f"/app/temp/image/{image_category}/test/1"
    if not os.path.exists(train_0_dir):
        raise FileNotFoundError("Train 폴더가 존재하지 않습니다.")
    if not os.path.exists(test_0_dir):
        raise FileNotFoundError("Test 폴더가 존재하지 않습니다.")

    col1.write("### Set Hypeparameter")
    date = col1.text_input("Date", value=time.strftime("%Y%m%d%H%M"))
    mode = col1.selectbox("Select Mode", options=["train", "transfer_learning"])
    if mode == "transfer_learning":
        model_path = col1.text_input("Model Path")
        model_type = ""
    else:
        model_type = col1.selectbox("Select Model", options=get_model_list())
        model_path = ""
    target_size = col1.selectbox("Select Target Size", options=[[512, 512, 1], [1024, 1024, 1]])
    epochs = col1.number_input("Epochs", value=9999)
    threshold = col1.number_input("Threshold", value=0.3)
    loss_func = col1.selectbox("Select Loss Function", options=["binary_crossentropy", "binary_focal_crossentropy"])
    batch_size = col1.slider("Batch Size", min_value=8, max_value=256, value=64, step=8)
    optimizer = col1.selectbox("Select Optimizer", options=["Adam", "sgd", "rmsprop"])
    lr = col1.number_input("Learning Rate", value=0.0005, step=0.0001, format="%.4f")
    patience = col1.number_input("Patience", value=10)
    weight = col1.number_input("Select Weight for class 1", value=1)
    log_dir = f"/app/temp/log/"
    tensorboard_log_dir = f"/app/temp/log/{date}/tensorboard"
    train_dir = f"/app/temp/image/{image_category}/train"
    test_dir = f"/app/temp/image/{image_category}/test"
    memo = col1.text_input("Memo")
    
    
    train_info = f"""
    Image Category : {image_category}
    Train 0 Image Count : {len(os.listdir(train_0_dir))}
    Train 1 Image Count : {len(os.listdir(train_1_dir))}
    Test 0 Image Count : {len(os.listdir(test_0_dir))}
    Test 1 Image Count : {len(os.listdir(test_1_dir))}
    
    Date : {date}
    Model Type : {model_type}
    Model Path : {model_path}
    Mode : {mode}
    Target Size : {target_size}
    Epochs : {epochs}
    Threshold : {threshold}
    Loss Function : {loss_func}
    Batch Size : {batch_size}
    Optimizer : {optimizer}
    Learning Rate : {lr}
    Patience : {patience}
    Weight : {weight}
    Log Dir : {log_dir}
    Tensorboard Log Dir : {tensorboard_log_dir}
    Train Dir : {train_dir}
    Test Dir : {test_dir}
    Memo : {memo}
    
    """
    col2.write("### Train Info")
    col2.code(train_info)
    if col2.button("Generate Train Config"):
        json.dump({
            "DATE": date,
            "MEMO": memo,
            "MODE": mode,
            "OPTIMIZER": optimizer,
            "MODEL_PATH": model_path,
            "TARGET_SIZE": target_size,
            "EPOCHS": epochs,
            "LOG_DIR": log_dir,
            "NOTION_DATABASE_ID_FS": "",
            "NOTION_KEY": "",
            "TENSORBOARD_LOG_DIR": tensorboard_log_dir,
            "REPREDICT": "Y",
            "THRESHOLD": threshold,
            "SAVE_FALSE_IMAGE": "Y",
            "MODEL_TYPE": model_type,
            "DATA_TYPE": image_category,
            "LOSS_FUNC": loss_func,
            "BATCH_SIZE": batch_size,
            "TRAIN_DATA_PATH": train_dir,
            "VAL_DATA_PATH": "",
            "TEST_DATA_PATH": test_dir,
            "LEARNING_RATE": round(lr, 4),
            "PATIENCE": patience,
            "WEIGHT": {
                "0": 1,
                "1": weight
            }
        }, open(f"/app/temp/train_config/{date}.json", "w"))
        message.success("Train Config Generated!")
    
if val == "2":
    config_list = os.listdir("/app/temp/train_config")
    if len(config_list) == 0:
        raise FileNotFoundError("생성된 train config가 없습니다.")
    config_list = natsorted(config_list)
    col1, col2 = st.columns(2)
    col1.write("### Select Train Config")
    config = col1.selectbox("Select Train Config", options=config_list)
    train_config = json.load(open(f"/app/temp/train_config/{config}"))
    col1.write("### Train Info")
    col1.code(json.dumps(train_config, indent=4))
    message = col2.empty()
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
    
    if col1.button("Start Train"):
        Experiment = supervised.Experiment_Model(**train_config)
        Experiment.save_config(f"/app/temp/train_config/{config}")
        Experiment.train_model()
        Experiment.test_model()
        Experiment.report_model()
        message.success("Train Finished!")
        col2.write("### Report")
        image_list = glob(f"/app/temp/log/{train_config['DATE']}/result/*.png")
        image_list = natsorted(image_list)
        for image in image_list:
            col2.image(image)
            
        df = pd.read_csv(f"/app/temp/log/{train_config['DATE']}/result/test_result.csv")
        col2.dataframe(df)
        
        
                
                
            
    
        

         
    
    
        