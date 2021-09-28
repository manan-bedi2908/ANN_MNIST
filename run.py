from utils.all_utils import load_data, sep_valid_data, sequence_model, save_model
from utils.model import MNIST_DIGIT


def main():
    (X_train_full, y_train_full), (X_test, y_test) = load_data()
    (X_valid, X_train), (y_valid, y_train), X_test = sep_valid_data(
        X_train_full, y_train_full, X_test
    )
    model_clf = sequence_model()
    history = MNIST_DIGIT.fit(model_clf, X_valid, y_valid, X_train, y_train)
    pred = MNIST_DIGIT.predict(X_test, model_clf)
    save_model(model_clf)


if __name__ == "__main__":
    main()
