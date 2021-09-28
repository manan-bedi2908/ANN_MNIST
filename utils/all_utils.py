import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os
import logging


def load_data():
    """
    It is used to load the data (Both training and testing)
    Args:
        None
    Returns:
        tuple: It returns the tuples of training and test set
    """
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    return (X_train_full, y_train_full), (X_test, y_test)


def sep_valid_data(X_train_full, y_train_full, X_test):
    """
    It is used to separate the Validation Data from the training data and also didviding each image pixel by 255
    Args:
        Train Set and Test Set
    Returns:
        tuple: It returns the updated train and test set along with Validation Data
    """
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    return (X_valid, X_train), (y_valid, y_train), X_test


def sequence_model():
    """
    It is used to add the layers to the Deep Neural Network.
    Flattening layer for Flatting the input data to be fed to the model
    Two Hidden Layers with ReLU activation function
    Output Layer having Softmax Activation Functions since we 10 classes
    """
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    return model_clf


def save_model(model_clf):
    """
    It saves the model in h5 format.
    Args:
        model classifier
    """
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    modelPath = os.path.join(model_dir, "model.h5")
    model_clf.save(modelPath)
