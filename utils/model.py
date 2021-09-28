import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class MNIST_DIGIT:
    def __init__(self, LOSS_FUNCTION, OPTIMIZER, METRICS, EPOCHS):
        """
        Initializes Loss Function, Optimizer, Evaluation Metric and Number of Epochs
        """
        self.LOSS_FUNCTION = "sparse_categorical_crossentropy"
        self.OPTIMIZER = "SGD"
        self.METRICS = ["accuracy"]
        self.EPOCHS = 30

    def fit(self, model_clf, X_valid, y_valid, X_train, y_train):
        """
        It fits the dataset to the Deep Neural Network model
        Args:
            Model Classifier, Validation Data, Training Data
        Returns:
            Trained Model
        """
        LOSS_FUNCTION = self.LOSS_FUNCTION
        OPTIMIZER = self.OPTIMIZER
        METRICS = self.METRICS
        model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
        EPOCHS = self.EPOCHS
        VALIDATION = (X_valid, y_valid)
        history = model_clf.fit(
            X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION
        )
        return history

    def predict(self, X_test, model_clf):
        """
        It predicts the output on a new data given
        Args:
            New Data of which output we want to predict.
        Returns:
            Prediction to which class the digit belongs (0-9)
        """
        X_new = X_test[:3]
        y_prob = model_clf.predict(X_new)
        y_prob.round(3)
        Y_pred = np.argmax(y_prob, axis=-1)
        return Y_pred
