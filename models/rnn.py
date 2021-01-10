import numpy as np
from tensorflow import keras
from keras.utils import np_utils

from models.model import Model
from lib.object_io import load_model
from lib.plotting import plot_confusion_matrix


class RNN(Model):

    def __init__(self, selected_model, pretrained_model=None):
        """
        Constructor

        :param selected_model: string specifying the model to use
        :param pretrained_model: pretrained model, None if not supplied
        """
        print(f"Recurrent Neural Network [{selected_model}]")
        self.selected_model = selected_model
        self.load_data()

        if pretrained_model is not None:
            self.model = load_model(pretrained_model)

            # print the model summary
            print(self.model.summary())

        # CNN architectures
        self.models = {
            "LSTM": {
                "layers": self.__get_lstm_layers(),
                "ref": ""
            }
        }

    def train(self):
        """
        Method for training the model

        :return:
        """
        print("Training")

        # prepare data
        train_X = self.train_X / 255.
        train_y = np_utils.to_categorical(self.train_y, 10)

        # prepare model
        architecture = self.models.get(self.selected_model)
        self.model = keras.models.Sequential(architecture.get("layers"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam",
                           metrics=["accuracy", "Precision", "Recall"])

        # fit the training data
        early_stopping_callback = keras.callbacks.EarlyStopping(patience=10)
        self.model.fit(train_X, train_y, batch_size=self.batch_size, validation_split=0.1, epochs=100,
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
        test_y = np_utils.to_categorical(self.test_y, 10)

        # evaluate the test data
        self.model.evaluate(test_X, test_y)

        # plot a confusion matrix
        pred_y = np.argmax(self.model.predict(test_X), axis=-1)
        plot_confusion_matrix(self.test_y, pred_y, name=f'{self.selected_model}_confusion_matrix', labels=self.labels)

    ###################################################
    # define static methods

    @staticmethod
    def __get_lstm_layers():
        layers = [
            keras.layers.LSTM(64, input_shape=(None, 28), name="LSTM"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation="softmax", name="OUTPUT")
        ]
        return layers