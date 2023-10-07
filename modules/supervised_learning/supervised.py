from .upload_notion import *
from modules.models.model import *

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import datetime
import matplotlib.pyplot as plt
import os
from glob import glob
import cv2
import pandas as pd
from sklearn.metrics import *
import seaborn as sns
import numpy as np
import shutil
from tqdm import tqdm


class Experiment_Model:
    def __init__(self, **kwargs):
        self.DATE = (
            kwargs["DATE"]
            if kwargs["DATE"] != ""
            else datetime.datetime.now().strftime("%Y%m%d%H%M")
        )
        self.DATE_ = (
            self.DATE[:4]
            + "-"
            + self.DATE[4:6]
            + "-"
            + self.DATE[6:8]
            + "T"
            + self.DATE[8:10]
            + ":"
            + self.DATE[10:12]
            + ":00.000+09:00"
        )
        self.LOG_ROOT_DIR = kwargs["LOG_DIR"]
        self.MODEL_DIR = f"{self.LOG_ROOT_DIR}/{self.DATE}/models"
        self.TRAIN_LOG_DIR = f"{self.LOG_ROOT_DIR}/{self.DATE}/logs"
        self.RESULT_DIR = f"{self.LOG_ROOT_DIR}/{self.DATE}/result"
        self.MODE = kwargs["MODE"]
        self.MODEL_PATH = kwargs["MODEL_PATH"]
        self.MODEL_TYPE = kwargs["MODEL_TYPE"]
        self.LOSS_FUNC = kwargs["LOSS_FUNC"]
        self.LOSS_FUNC_ = (
            "focal_loss"
            if self.LOSS_FUNC == "binary_focal_crossentropy"
            else self.LOSS_FUNC
        )
        self.WEIGHT = {int(key): value for key, value in kwargs["WEIGHT"].items()}
        self.WEIGHT_ = str(self.WEIGHT).replace("{", "(").replace("}", ")")
        self.LEARNING_RATE = kwargs["LEARNING_RATE"]
        self.OPTIMIZER = kwargs["OPTIMIZER"]
        self.TRAIN_DATA_PATH = kwargs["TRAIN_DATA_PATH"]
        self.VAL_DATA_PATH = kwargs["VAL_DATA_PATH"]
        self.TEST_DATA_PATH = kwargs["TEST_DATA_PATH"]
        self.TARGET_SIZE = kwargs["TARGET_SIZE"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.EPOCHS = kwargs["EPOCHS"]
        self.DATA_TYPE = kwargs["DATA_TYPE"]
        self.NOTION_DATABASE_ID_FS = kwargs["NOTION_DATABASE_ID_FS"]
        self.NOTION_KEY = kwargs["NOTION_KEY"]
        self.MEMO = kwargs["MEMO"]
        self.PATIENCE = kwargs["PATIENCE"]
        self.TENSORBOARD_LOG_DIR = kwargs["TENSORBOARD_LOG_DIR"]
        self.REPREDICT = kwargs["REPREDICT"]
        self.THRESHOLD = kwargs["THRESHOLD"]
        self.SAVE_FALSE_IMAGE = kwargs["SAVE_FALSE_IMAGE"]

        os.makedirs(f"{self.LOG_ROOT_DIR}/{self.DATE}/models", exist_ok=True)
        os.makedirs(f"{self.LOG_ROOT_DIR}/{self.DATE}/logs", exist_ok=True)
        os.makedirs(f"{self.LOG_ROOT_DIR}/{self.DATE}/result", exist_ok=True)

    def save_config(self, config):
        shutil.copyfile(config, f"{self.LOG_ROOT_DIR}/{self.DATE}/config.json")

    def load_train_data(self):
        if self.VAL_DATA_PATH == "":
            self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.TRAIN_DATA_PATH,
                validation_split=0.3,
                subset="training",
                seed=123,
                image_size=(self.TARGET_SIZE[0], self.TARGET_SIZE[1]),
                batch_size=self.BATCH_SIZE,
                label_mode="binary",
                color_mode=("grayscale" if self.TARGET_SIZE[2] == 1 else "rgb"),
            )
            self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.TRAIN_DATA_PATH,
                validation_split=0.3,
                subset="validation",
                seed=123,
                image_size=(self.TARGET_SIZE[0], self.TARGET_SIZE[1]),
                batch_size=self.BATCH_SIZE,
                label_mode="binary",
                color_mode=("grayscale" if self.TARGET_SIZE[2] == 1 else "rgb"),
            )
        else:
            self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.TRAIN_DATA_PATH,
                seed=123,
                image_size=(self.TARGET_SIZE[0], self.TARGET_SIZE[1]),
                batch_size=self.BATCH_SIZE,
                label_mode="binary",
                color_mode=("grayscale" if self.TARGET_SIZE[2] == 1 else "rgb"),
            )
            self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.VAL_DATA_PATH,
                seed=123,
                image_size=(self.TARGET_SIZE[0], self.TARGET_SIZE[1]),
                batch_size=self.BATCH_SIZE,
                label_mode="binary",
                color_mode=("grayscale" if self.TARGET_SIZE[2] == 1 else "rgb"),
            )
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = (
                self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            )
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def load_train_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            if self.MODE == "train":
                self.model = get_model(self.MODEL_TYPE, self.TARGET_SIZE)
            elif self.MODE == "transfer_learning":
                self.model = tf.keras.models.load_model(self.MODEL_PATH)

            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.LEARNING_RATE,
                first_decay_steps=100,
                t_mul=2,
                m_mul=0.9,
                alpha=0,
                name=None,
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=self.LOSS_FUNC,
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                ],
            )

    def get_callbacks(self):
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.999,
            patience=20,
            verbose=2,
            min_delta=1e-4,
            min_lr=1e-9,
            mode="auto",
        )

        checkpoint = ModelCheckpoint(
            filepath=f"{self.MODEL_DIR}/{self.MODEL_TYPE}.h5",
            verbose=2,
            save_best_only=True,
            monitor="val_loss",
            mode="auto",
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            verbose=2,
            mode="auto",
            patience=self.PATIENCE,
        )

        csv_logger = CSVLogger(
            f"{self.TRAIN_LOG_DIR}/{self.MODEL_TYPE}.csv",
            append=True,
        )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.TENSORBOARD_LOG_DIR, histogram_freq=1
        )
        return [reduce_lr, checkpoint, early_stopping, csv_logger, tensorboard_callback]

    def draw_learning_curve(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["binary_accuracy"], label="accuracy")
        plt.plot(self.history.history["val_binary_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.RESULT_DIR, "learning_curve.png"), dpi=500)
        plt.close()
        plt.figure(figsize=(12, 4))
        plt.plot(self.history.history["lr"], label="lr")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.RESULT_DIR, "learning_rate.png"), dpi=500)
        plt.close()

    def train_model(self):
        self.load_train_data()
        self.load_train_model()
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.EPOCHS,
            class_weight=self.WEIGHT,
            callbacks=self.get_callbacks(),
            verbose=1,
            workers=40,
            use_multiprocessing=True,
        )
        self.model.save(f"{self.MODEL_DIR}/{self.MODEL_TYPE}_9999.h5")
        self.draw_learning_curve()

    def load_test_data(self):
        test_image = glob(self.TEST_DATA_PATH + "/**/*.png", recursive=True)
        test_image_label = [int(i.split("/")[-2]) for i in test_image]
        test_image_list = []
        for i in tqdm(test_image):
            img = (
                cv2.imread(i, cv2.IMREAD_GRAYSCALE)
                if self.TARGET_SIZE[2] == 1
                else cv2.imread(i, cv2.IMREAD_COLOR)
            )
            img = cv2.resize(img, (self.TARGET_SIZE[0], self.TARGET_SIZE[1]))
            test_image_list.append(img)
        test_image_list = np.array(test_image_list)
        test_image_list = (
            test_image_list.reshape(-1, self.TARGET_SIZE[0], self.TARGET_SIZE[1], 1)
            if self.TARGET_SIZE[2] == 1
            else test_image_list.reshape(
                -1, self.TARGET_SIZE[0], self.TARGET_SIZE[1], 3
            )
        )
        self.test_image_list, self.test_image_label, self.test_image = (
            test_image_list,
            test_image_label,
            test_image,
        )

    def load_best_model(self):
        model_list = glob(self.MODEL_DIR + "/*.h5")
        model_list.sort()
        print(model_list[0])
        self.model = tf.keras.models.load_model(model_list[0])

    def load_test_model(self):
        self.model = tf.keras.models.load_model(self.MODEL_PATH)

    def test_model(self):
        if self.REPREDICT == "Y":
            if os.path.exists(self.RESULT_DIR + "/test_result.csv"):
                os.remove(self.RESULT_DIR + "/test_result.csv")

        if os.path.exists(self.RESULT_DIR + "/test_result.csv"):
            print("Test result already exists.")
            self.test_pred = pd.read_csv(self.RESULT_DIR + "/test_result.csv")[
                "0"
            ].to_list()
            self.test_pred = np.array(self.test_pred).reshape(-1, 1)
            self.test_image_label = pd.read_csv(self.RESULT_DIR + "/test_result.csv")[
                "label"
            ].to_list()
            self.test_image = pd.read_csv(self.RESULT_DIR + "/test_result.csv")[
                "image_path"
            ].to_list()
        else:
            self.load_test_data()
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                self.load_best_model() if self.MODEL_PATH == "" else self.load_test_model()
            self.test_pred = self.model.predict(self.test_image_list)
            df = pd.DataFrame(self.test_pred)
            df["label"] = self.test_image_label
            df["image_path"] = self.test_image
            df.to_csv(f"{self.RESULT_DIR}/test_result.csv", index=False)

        if self.SAVE_FALSE_IMAGE == "Y":
            test_pred = np.where(self.test_pred > self.THRESHOLD, 1, 0)
            shutil.rmtree(self.RESULT_DIR + "/false_image", ignore_errors=True)
            os.makedirs(self.RESULT_DIR + "/false_image", exist_ok=True)
            for i in range(len(self.test_image)):
                if self.test_image_label[i] != test_pred[i]:
                    shutil.copy(self.test_image[i], self.RESULT_DIR + "/false_image")

    def draw_threshold_change_curves(self):
        Threshold_list = []
        Precision_list = []
        Recall_list = []
        F1_score_list = []
        Accuracy_list = []
        len_0 = self.test_image_label.count(0)
        len_1 = self.test_image_label.count(1)
        for i in range(0, 1000):
            Threshold = i / 1000
            Threshold_list.append(Threshold)
            test_pred = np.where(self.test_pred > Threshold, 1, 0)
            report = classification_report(
                self.test_image_label,
                test_pred,
                labels=[0, 1],
                target_names=["class 0", "class 1"],
                digits=4,
                zero_division=0,
                output_dict=True,
            )
            F1_score_list.append(report["class 1"]["f1-score"])
            Recall_list.append(report["class 1"]["recall"])
            Precision_list.append(report["class 1"]["precision"])
            Accuracy_list.append(
                (
                    report["class 1"]["precision"] * len_0
                    + report["class 1"]["recall"] * len_1
                )
                / (len_0 + len_1)
            )
        df = pd.DataFrame(
            {
                "Threshold": Threshold_list,
                "Precision": Precision_list,
                "Recall": Recall_list,
                "F1_score": F1_score_list,
                "Accuracy": Accuracy_list,
            }
        )
        df.to_csv(self.RESULT_DIR + f"/metric_score_threshold_.csv", index=False)
        df = df.sort_values(by=["F1_score"], ascending=False)
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(Threshold_list, Precision_list, label="Precision")
        plt.xlabel("Threshold")
        plt.ylabel("Precision")
        plt.subplot(2, 2, 2)
        plt.plot(Threshold_list, Recall_list, label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Recall")
        plt.subplot(2, 2, 3)
        plt.plot(Threshold_list, F1_score_list, label="F1_score")
        plt.xlabel("Threshold")
        plt.ylabel("F1_score")
        plt.subplot(2, 2, 4)
        plt.plot(Threshold_list, Accuracy_list, label="Accuracy")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.suptitle("Metric Score Threshold Change")
        plt.savefig(self.RESULT_DIR + f"/metric_score_threshold_change.png", dpi=500)
        df_threshold = df[df["Threshold"] == self.THRESHOLD]
        (
            self.result_threshold,
            self.result_f1,
            self.result_recall,
            self.result_precision,
            self.result_accuracy,
        ) = (
            df_threshold.iloc[0]["Threshold"],
            df_threshold.iloc[0]["F1_score"],
            df_threshold.iloc[0]["Recall"],
            df_threshold.iloc[0]["Precision"],
            df_threshold.iloc[0]["Accuracy"],
        )
        # self.result_threshold, self.result_f1, self.result_recall, self.result_precision, self.result_accuracy = df.iloc[0]["Threshold"], df.iloc[0]["F1_score"], df.iloc[0]["Recall"], df.iloc[0]["Precision"], df.iloc[0]["Accuracy"]

    def write_classification_report(self):
        self.result_test_pred = np.where(self.test_pred > self.result_threshold, 1, 0)
        self.classification_report = classification_report(
            self.test_image_label,
            self.result_test_pred,
            target_names=["Accept", "Reject"],
            digits=4,
            zero_division=0,
        )
        with open(self.RESULT_DIR + f"/classification_report.txt", "w") as f:
            f.write(self.classification_report)

    def draw_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.test_image_label, self.test_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            f"Receiver operating characteristic - threshold : {self.result_threshold}"
            + self.MODEL_TYPE
        )
        plt.legend(loc="lower right")
        plt.savefig(self.RESULT_DIR + f"/roc_curve.png", dpi=500)
        self.result_auroc = roc_auc

    def draw_confusion_matrix(self):
        plt.figure(figsize=(10, 10))
        plt.title(
            f"Confusion matrix - threshold : {self.result_threshold}" + self.MODEL_TYPE
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        sns.heatmap(
            confusion_matrix(self.test_image_label, self.result_test_pred),
            annot=True,
            fmt="d",
            cmap=plt.cm.Blues,
        )
        plt.savefig(self.RESULT_DIR + f"/confusion_matrix.png", dpi=500)

    def draw_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(
            self.test_image_label, self.test_pred
        )
        # AUCPR
        aucpr = auc(recall, precision)
        plt.figure(figsize=(10, 10))
        plt.plot(
            recall, precision, label="Precision-Recall curve (area = %0.2f)" % aucpr
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision-Recall curve - threshold : {self.result_threshold}"
            + self.MODEL_TYPE
        )
        plt.legend(loc="lower right")
        plt.savefig(self.RESULT_DIR + f"/precision_recall_curve.png", dpi=500)
        self.result_aucpr = aucpr

    def report_model(self):
        self.draw_threshold_change_curves()
        self.write_classification_report()
        self.draw_roc_curve()
        self.draw_confusion_matrix()
        self.draw_precision_recall_curve()

    def upload_notion(self):
        notion = NotionDatabase(self.NOTION_DATABASE_ID_FS, self.NOTION_KEY)
        page_values = {
            "AUCPR": round(self.result_aucpr, 4),
            "AUROC": round(self.result_auroc, 4),
            "Accuracy": round(self.result_accuracy, 4),
            "Batch Size": str(self.BATCH_SIZE),
            "Class Weight": self.WEIGHT_,
            "Dataset": self.DATA_TYPE,
            "F1 Score": round(self.result_f1, 4),
            "Input Size": str(self.TARGET_SIZE)
            .replace(",", "")
            .replace("[", "(")
            .replace("]", ")"),
            "Learning Rate": self.LEARNING_RATE,
            "Loss Func": self.LOSS_FUNC_,
            "Model": self.MODEL_TYPE,
            "Precision": round(self.result_precision, 4),
            "Recall": round(self.result_recall, 4),
            "Threshold": self.result_threshold,
            "상태": "Done",
            "실행 일시": self.DATE_,
            "테스트 일시": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+09:00"),
            "Memo": self.MEMO,
            "Patience": self.PATIENCE,
        }
        notion.upload_page_values(page_values)
