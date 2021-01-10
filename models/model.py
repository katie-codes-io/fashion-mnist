from abc import ABC, abstractmethod
from keras.datasets import fashion_mnist
import numpy as np

from lib.plotting import plot_examples
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
    name = None
    models = {}
    model = None
    batch_size = 32
    labels = ["0: T-shirt", "1: Trousers", "2: Pullover", "3: Dress", "4: Coat", "5: Sandal", "6: Shirt", "7: Sneaker", "8: Bag", "9: Ankle boot"]

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
        dump_model(self.model, self.name)

    def load_data(self):
        """
        Method to load the MNIST data into instant variables

        :return:
        """
        print("Loading data")
        (self.train_X, self.train_y), (self.test_X, self.test_y) = fashion_mnist.load_data()

        # get a test image for each class (just grabbing the first occurrence)
        imgs = []
        for i in range(0, 10):
            index = np.argmax(self.train_y == i)
            imgs.append(self.train_X[index])

        # plot the images
        plot_examples(imgs, name="mnist_examples", labels=self.labels)

