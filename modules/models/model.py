import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.keras.models import *

# Model
from tensorflow.keras.applications import *


class BaseModel:
    def __init__(self, **kwargs):
        self.target_size = kwargs["TARGET_SIZE"]

    def get_model(self):
        raise NotImplementedError


class CNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Conv2D(63, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(63, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(63, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Flatten(),
                Dense(1611, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class CNN3(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                # augmentation
                Conv2D(96, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(96, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Flatten(),
                Dense(2048, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class CNN5(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                # RandomFlip("horizontal"),
                # RandomRotation(0.1),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Flatten(),
                Dense(2048, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class CNN4(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                Flatten(),
                Dense(2048, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class CNN6(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                RandomFlip("horizontal"),
                RandomRotation(0.1),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPool2D(2, 2),
                BatchNormalization(),
                Flatten(),
                Dense(2048, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class CNN2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPool2D(2, 2),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPool2D(2, 2),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPool2D(2, 2),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPool2D(2, 2),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPool2D(2, 2),
                Conv2D(32, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                Flatten(),
                Dense(1024, activation="gelu"),
                Dropout(0.3),
                Dense(512, activation="gelu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class VGG16(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.VGG16(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class ResNet50(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.ResNet50(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class InceptionV3(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class InceptionResNetV2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.InceptionResNetV2(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class MobileNetV2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.MobileNetV2(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class DenseNet121(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.DenseNet121(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class Xception(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.Xception(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class EfficientNetB0(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.EfficientNetB0(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class NASNetMobile(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.NASNetMobile(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model


class NASNetLarge(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        model = tf.keras.models.Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.NASNetLarge(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model
    
def get_model_list():
    return [
        "CNN",
        "VGG16",
        "ResNet50",
        "InceptionV3",
        "InceptionResNetV2",
        "MobileNetV2",
        "DenseNet121",
        "Xception",
        "EfficientNetB0",
        "NASNetMobile",
        "NASNetLarge",
        "CNN2",
        "CNN3",
        "CNN4",
        "CNN5",
        "CNN6",
    ]


def get_model(model_name, target_size):
    model_mapping = {
        "CNN": CNN,
        "VGG16": VGG16,
        "ResNet50": ResNet50,
        "InceptionV3": InceptionV3,
        "InceptionResNetV2": InceptionResNetV2,
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "Xception": Xception,
        "EfficientNetB0": EfficientNetB0,
        "NASNetMobile": NASNetMobile,
        "NASNetLarge": NASNetLarge,
        "CNN2": CNN2,
        "CNN3": CNN3,
        "CNN4": CNN4,
        "CNN5": CNN5,
        "CNN6": CNN6,
    }

    if model_name in model_mapping:
        return model_mapping[model_name](TARGET_SIZE=target_size).get_model()
    else:
        raise ValueError("Invalid model name: ", model_name)
