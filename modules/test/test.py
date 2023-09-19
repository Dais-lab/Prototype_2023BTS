import cv2
import numpy as np
import pandas as pd
from glob import glob
import modules.preprocess.preprocessing as preprocessing
from sklearn.metrics import *
import matplotlib.pyplot as plt


def load_test(data_path, target_size, weld_type = False, preprocess = None):
    """테스트 데이터를 불러오는 함수
    """
    if weld_type == True and preprocess is not None:
        raise ValueError("weld_type과 preprocess를 동시에 사용할 수 없습니다.")
    #image파일만 불러오기
    test_image_path = glob(data_path + "/**/*")
    test_image_path = [i for i in test_image_path if i.split(".")[-1] in ["jpg", "png", "jpeg"]]
    try:
        test_image_label = [int(i.split("/")[-2]) for i in test_image_path]
    except:
        test_image_label = None
        
    test_image_list = []
    for i in test_image_path:
        img = cv2.imread(i, (cv2.IMREAD_GRAYSCALE if target_size[2] == 1 else cv2.IMREAD_COLOR))
        img = cv2.resize(img, (target_size[0], target_size[1]))
        if weld_type:
            img = preprocessing.get_weld_image(img)
            
        if preprocess is not None:
            img = preprocessing.preprocess_img(img, preprocess, target_size)
            
        img = np.array(img, dtype=np.float32)
        img = img / 255.
            
        test_image_list.append(img)
    test_image_list = np.array(test_image_list)
    test_image_list = test_image_list.reshape(-1, target_size[0], target_size[1], target_size[2])
    
    return test_image_list, test_image_label, test_image_path

def evaluate_data(Y_test, Y_pred):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(Y_pred)):
        if Y_test[i] == 0 and Y_pred[i] == 0:
            TN = TN + 1
        elif Y_test[i] == 0 and Y_pred[i] == 1:
            FP = FP + 1
        elif Y_test[i] == 1 and Y_pred[i] == 0:
            FN = FN + 1
        elif Y_test[i] == 1 and Y_pred[i] == 1:
            TP = TP + 1

    Recall = recall_score(Y_test, Y_pred)
    Precision = precision_score(Y_test, Y_pred)
    Accuracy = accuracy_score(Y_test, Y_pred)
    F1_Score = f1_score(Y_test, Y_pred)
    return TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision

def generate_report(df):
    thresholds = []
    total_Accuracy = []
    total_F1_Score = []
    total_Recall = []
    total_Precision = []
    total_inspection_persent = []
    total_hit_ratio = []
    total_TN = []
    total_FP = []
    total_FN = []
    total_TP = []
    total_minus_inspection_present = []
    for threshold in range(100, 1001, 1):
        threshold = threshold/1000
        df_test = df.copy()
        df_test["predict"] = df_test["predict"].apply(lambda x: 1 if x >= threshold else 0)
        inspection_df = df_test.loc[threshold < df_test["predict"]]
        temp_df = df_test.drop(inspection_df.index, axis = 0)
        if len(temp_df) == 0:
            continue
        inspection_len = len(inspection_df)
        inspection_persent = (inspection_len/len(temp_df))
        minus_inspection_persent = 1 - (inspection_len/len(temp_df))
        TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision = evaluate_data(df_test["ground_truth"], df_test["predict"])
        hit_ratio = TP/(FN + TP)
        thresholds.append(threshold)
        total_Accuracy.append(Accuracy)
        total_F1_Score.append(F1_Score)
        total_Recall.append(Recall)
        total_Precision.append(Precision)
        total_inspection_persent.append(inspection_persent)
        total_hit_ratio.append(hit_ratio)
        total_TN.append(TN)
        total_FP.append(FP)
        total_FN.append(FN)
        total_TP.append(TP)
        total_minus_inspection_present.append(minus_inspection_persent)

    df_result = pd.DataFrame({"threshold" : thresholds,
                        "TN" : total_TN,
                        "FP" : total_FP,
                        "FN" : total_FN,
                        "TP" : total_TP,
                        "hit_ratio" : total_hit_ratio,
                        "재검률" : total_inspection_persent,
                        "Accuracy" : total_Accuracy,
                        "F1_Score" : total_F1_Score,
                        "Recall" : total_Recall,
                        "precision" : total_Precision})

    df_result.to_csv("/app/temp/hit_evaluation_result.csv", index = False)
    
    x = list(df_result["threshold"])
    y1 = list(df_result["hit_ratio"])
    y2 = total_minus_inspection_present

    fig, ax1 = plt.subplots()
    plt.title("hit_ratio")
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('hit_ratio')
    line1 = ax1.plot(x, y1, color = 'red', alpha = 0.5, label = "hit_ratio(%)")

    ax2 = ax1.twinx()
    ax2.set_ylabel('1 - inspection')
    line2 = ax2.plot(x, y2, color = 'blue', alpha = 0.5, label = "1 - inspection")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    plt.savefig("/app/temp/hit_ratio.png")
    plt.close()
    
def test_model(model, test_image_list, test_image_label, test_image_path):
    test_pred = model.predict(test_image_list)
    df = pd.DataFrame()
    test_image_path = [path.split("/")[-1] for path in test_image_path]
    df["ground_truth"] = test_image_label
    df["predict"] = test_pred
    df["path"] = test_image_path
    df.to_csv("/app/temp/test_result.csv", index = False)
    generate_report(df)
    
def test_model_no_label(model, test_image_list, test_image_path):
    test_pred = model.predict(test_image_list)
    df = pd.DataFrame()
    test_image_path = [path.split("/")[-1] for path in test_image_path]
    df["predict"] = test_pred
    df["path"] = test_image_path
    df.to_csv("/app/temp/test_result.csv", index = False)