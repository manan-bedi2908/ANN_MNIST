import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def load_data():
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    return (X_train_full, y_train_full), (X_test, y_test)


def sep_valid_data(X_train_full, y_train_full, X_test):
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    return (X_valid, X_train), (y_valid, y_train), X_test


def sequence_model():
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    return model_clf


def save_model(model_clf):
    model_clf.save("model.h5")
