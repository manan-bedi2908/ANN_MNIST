from utils.all_utils import load_data, sep_valid_data, sequence_model, save_model
from utils.model import MNIST_DIGIT
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    level=logging.INFO,
    format=logging_str,
    filemode="a",
)


def main(LOSS_FUNCTION, OPTIMIZER, METRICS, EPOCHS):
    try:
        (X_train_full, y_train_full), (X_test, y_test) = load_data()
    except Exception as e:
        logging.info(f"Exception: {e}")
    else:
        logging.info("Successfully loaded data")

    try:
        (X_valid, X_train), (y_valid, y_train), X_test = sep_valid_data(
            X_train_full, y_train_full, X_test
        )
    except Exception as e:
        logging.info(f"Exception: {e}")
    else:
        logging.info("Successfully separated the Validation Data")

    try:
        model_clf = sequence_model()
    except Exception as e:
        logging.info(f"Exception: {e}")
    else:
        logging.info("Successfully added the layers")

    try:
        model = MNIST_DIGIT(LOSS_FUNCTION, OPTIMIZER, METRICS, EPOCHS)
        history = model.fit(model_clf, X_valid, y_valid, X_train, y_train)
    except Exception as e:
        logging.info(f"Exception: {e}")
    else:
        logging.info("Successfully trained the dataset on the training dataset")

    try:
        pred = model.predict(X_test, model_clf)
    except Exception as e:
        logging.info(f"Exception: {e}")
    else:
        logging.info(f"Predictions given")

    try:
        save_model(model_clf)
    except Exception as e:
        logging.info(f"Exception: {e}")
    else:
        logging.info("Saved the model h5 file")


if __name__ == "__main__":
    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]
    EPOCHS = 30
    main(
        LOSS_FUNCTION=LOSS_FUNCTION, OPTIMIZER=OPTIMIZER, METRICS=METRICS, EPOCHS=EPOCHS
    )
