import numpy as np
from tensorflow import keras
from keras.utils import np_utils

from models.model import Model
from lib.object_io import load_model
from lib.layers.residual_unit import ResidualUnit
from lib.plotting import plot_confusion_matrix


class CNN(Model):

    def __init__(self, selected_model, save_name=None, pretrained_model=None):
        """
        Constructor

        :param selected_model: string specifying the model to use
        :param save_name: name to save model to, defaults to None if not supplied
        :param pretrained_model: pretrained model, None if not supplied
        """
        print(f"Convolutional Neural Network [{selected_model}]")
        self.selected_model = selected_model
        self.name = save_name
        self.load_data()

        if pretrained_model is not None:
            self.model = load_model(pretrained_model)

            # print the model summary
            print(self.model.summary())

        # CNN architectures
        self.models = {
            "AlexNet": {
                "layers": self.__get_alexnet_layers(),
                "ref": "Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017)."
                       "Imagenet classification with deep convolutional neural networks."
                       "Communications of the ACM, 60(6), 84-90."
            },
            "LeNet": {
                "layers": self.__get_lenet_layers(),
                "ref": "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998)."
                       "Gradient-based learning applied to document recognition."
                       "Proceedings of the IEEE, 86(11), 2278-2324."
            },
            "ResNet": {
                "layers": self.__get_resnet_layers(),
                "ref": "He, K., Zhang, X., Ren, S., & Sun, J. (2016)."
                       "Deep residual learning for image recognition."
                       "In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778)."
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
        train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
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
        test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)
        test_y = np_utils.to_categorical(self.test_y, 10)

        # evaluate the test data
        self.model.evaluate(test_X, test_y)

        # plot a confusion matrix
        pred_y = np.argmax(self.model.predict(test_X), axis=-1)
        plot_confusion_matrix(self.test_y, pred_y, name=f'{self.selected_model}_confusion_matrix', labels=self.labels)

    ###################################################
    # define static methods

    @staticmethod
    def __get_alexnet_layers():
        layers = [
            keras.layers.experimental.preprocessing.Resizing(227, 227, input_shape=(28, 28, 1), name="INPUT"),
            keras.layers.Conv2D(96, 11, strides=4, padding="valid", activation="relu", input_shape=(227, 227, 1),
                                name="C1"),
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="S2", input_shape=(55, 55, 96)),
            keras.layers.Conv2D(256, 5, strides=1, padding="same", activation="relu", name="C3",
                                input_shape=(27, 27, 96)),
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="S4", input_shape=(27, 27, 256)),
            keras.layers.Conv2D(384, 3, strides=1, padding="same", activation="relu", name="C5",
                                input_shape=(13, 13, 256)),
            keras.layers.Conv2D(384, 3, strides=1, padding="same", activation="relu", name="C6",
                                input_shape=(13, 13, 384)),
            keras.layers.Conv2D(256, 3, strides=1, padding="same", activation="relu", name="C7",
                                input_shape=(13, 13, 384)),
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="S8", input_shape=(13, 13, 256)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu", name="F9"),
            keras.layers.Dense(4096, activation="relu", name="F10"),
            keras.layers.Dense(10, activation="softmax", name="OUTPUT"),
        ]

        return layers

    @staticmethod
    def __get_lenet_layers():
        layers = [
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
        ]

        return layers

    @staticmethod
    def __get_resnet_layers():
        layers = [
            keras.layers.experimental.preprocessing.Resizing(227, 227, input_shape=(28, 28, 1), name="INPUT"),
            keras.layers.Conv2D(64, 7, strides=2, padding="same", input_shape=(227, 227, 1), use_bias=False,
                                name="C1"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="S2", input_shape=(27, 27, 256)),
        ]
        # add the Residual Units
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            layers.append(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        # add the final layers
        layers.append(keras.layers.GlobalAvgPool2D())
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(10, activation="softmax"))

        return layers
