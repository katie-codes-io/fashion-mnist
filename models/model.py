from abc import ABC, abstractmethod
from keras.datasets import fashion_mnist

from lib.plotting import plot_img
from lib.object_io import dump_model


class Model(ABC):
    """
    Model abstract class that other models should inherit
    """

    # declare instance variables
    train_X = None
    train_y = None
    test_X = None
    test_y = None

    selected_model = None
    models = {}
    model = None
    batch_size = 32

    @abstractmethod
    def train(self):
        """
        Abstract method to train the model - must be implemented by classes inheriting Model

        :return:
        """

    @abstractmethod
    def evaluate(self):
        """
        Abstract method to evaluate the model - must be implemented by classes inheriting Model

        :return:
        """

    def export_model(self):
        """
        Method for exporting the model

        :return:
        """
        print("Exporting object")
        dump_model(self.model, self.selected_model)

    def load_data(self):
        """
        Method to load the MNIST data into instant variables

        :return:
        """
        print("Loading data")
        (self.train_X, self.train_y), (self.test_X, self.test_y) = fashion_mnist.load_data()

        # plot a test image
        plot_img(self.train_X[0], name="test_mnist")
