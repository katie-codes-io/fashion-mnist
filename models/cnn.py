from tensorflow import keras
from keras.utils import np_utils

from models.model import Model
from lib.object_io import dump_model, load_model


class CNN(Model):
    # declare instance variables
    selected_model = None
    model = None
    batch_size = 32

    # CNN architectures
    models = {
        "LeNet": {
            "layers": [
                keras.layers.ZeroPadding2D(padding=2, input_shape=(28, 28, 1), name="INPUT"),
                keras.layers.Conv2D(6, 5, strides=1, padding="valid", activation="tanh", input_shape=(32, 32, 1),
                                    name="C1"),
                keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding="same", input_shape=(28, 28, 1),
                                              name="S2"),
                keras.layers.Conv2D(16, 5, strides=1, padding="valid", activation="tanh", input_shape=(14, 14, 1),
                                    name="C3"),
                keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding="same", input_shape=(10, 10, 1),
                                              name="S4"),
                keras.layers.Conv2D(120, 5, strides=1, padding="valid", activation="tanh", input_shape=(5, 5, 1),
                                    name="C5"),
                keras.layers.Flatten(),
                keras.layers.Dense(84, activation="tanh", name="F6"),
                keras.layers.Dense(10, activation="softmax", name="OUTPUT")
            ],
            "ref": "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998)."
                   "Gradient-based learning applied to document recognition."
                   "Proceedings of the IEEE, 86(11), 2278-2324."
        }
    }

    def __init__(self, selected_model, pretrained_model=None):
        """
        Constructor - initialise the hyperparameters

        :param selected_model: string specifying the model to use
        :param pretrained_model: pretrained model, None if not supplied
        """
        print(f"Convolutional Neural Network [{selected_model}]")
        self.selected_model = selected_model
        self.load_data()

        if pretrained_model is not None:
            self.model = load_model(pretrained_model)

            # print the model summary
            print(self.model.summary())

    def train(self):
        """
        Method for training the model

        :return:
        """
        print("Training")

        # prepare data
        train_X = self.train_X / 255.
        train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
        train_y = np_utils.to_categorical(self.train_y, 10)

        # prepare model
        architecture = self.models.get(self.selected_model)
        self.model = keras.models.Sequential(architecture.get("layers"))
        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-2),
                           metrics=["accuracy", "Precision", "Recall"])

        # fit the training data
        early_stopping_callback = keras.callbacks.EarlyStopping(patience=10)
        self.model.fit(train_X, train_y, batch_size=self.batch_size, validation_split=0.1, epochs=2,
                       callbacks=[early_stopping_callback])

        # print the model summary
        print(self.model.summary())

    def evaluate(self):
        """
        Method for evaluating the model

        :return:
        """
        print("Evaluating")

        # prepare data
        test_X = self.test_X / 255.
        test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)
        test_y = np_utils.to_categorical(self.test_y, 10)

        # evaluate the test data
        self.model.evaluate(test_X, test_y)

    def export_model(self):
        """
        Method for exporting the model

        :return:
        """
        print("Exporting object")

        dump_model(self.model, "LeNet")
