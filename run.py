from utils.all_utils import load_data, sep_valid_data, sequence_model, save_model
from utils.model import MNIST_DIGIT
import os


def main(LOSS_FUNCTION, OPTIMIZER, METRICS, EPOCHS):
    (X_train_full, y_train_full), (X_test, y_test) = load_data()
    (X_valid, X_train), (y_valid, y_train), X_test = sep_valid_data(
        X_train_full, y_train_full, X_test
    )
    model_clf = sequence_model()
    model = MNIST_DIGIT(LOSS_FUNCTION, OPTIMIZER, METRICS, EPOCHS)
    history = model.fit(model_clf, X_valid, y_valid, X_train, y_train)
    pred = model.predict(X_test, model_clf)
    save_model(model_clf)


if __name__ == "__main__":
    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]
    EPOCHS = 30
    main(
        LOSS_FUNCTION=LOSS_FUNCTION, OPTIMIZER=OPTIMIZER, METRICS=METRICS, EPOCHS=EPOCHS
    )
