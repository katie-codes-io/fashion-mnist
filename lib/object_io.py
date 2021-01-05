import os
import pickle
import joblib
import keras

from lib.layers.residual_unit import ResidualUnit


def dump_object(obj, name, directory='objects', numpy=False):
    """
    Pickles a Python object

    :param obj: object to pickle
    :param name: filename to save object to
    :param directory: directory to save file to
    :param numpy: boolean specifying whether we're pickling a numpy object
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # check for and append the file suffix
    if not name.endswith('.pkl'):
        name = name + '.pkl'

    # construct filename with path
    filename = os.path.join(directory, name)

    # write the object
    if not numpy:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        joblib.dump(obj, filename)


def load_object(name, directory='objects', numpy=False):
    """
    Unpickles a byte stream

    :param name: name of the file to unpickle
    :param directory: directory to find file
    :param numpy: boolean to specify whether this is a numpy object
    :return: unpickled object
    """

    # check for and append the file suffix
    if not name.endswith('.pkl'):
        name = name + '.pkl'

    # construct filename with path
    filename = os.path.join(directory, name)

    # load the object
    if os.path.exists(filename):
        if not numpy:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            return joblib.load(filename)

    else:
        print(f"{filename} object does not exist. Exiting.")
        exit(0)


def dump_model(model, name, directory='trained_models'):
    """
    Save a Keras machine learning model

    :param model: model to save
    :param name: filename to save object to
    :param directory: directory to save file to
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # check for and append the file suffix
    if not name.endswith('.h5'):
        name = name + '.h5'

    # construct filename with path
    filename = os.path.join(directory, name)

    # write the object
    model.save(filename)


def load_model(name, directory='trained_models'):
    """
    Loads a Keras machine learning model

    :param name: name of the file to load
    :param directory: directory to find file
    :return: Keras Model object
    """

    # check for and append the file suffix
    if not name.endswith('.h5'):
        name = name + '.h5'

    # construct filename with path
    filename = os.path.join(directory, name)

    # load the object
    if os.path.exists(filename):
        return keras.models.load_model(filename, custom_objects={"ResidualUnit": ResidualUnit})

    else:
        print(f"{filename} model does not exist. Exiting.")
        exit(0)
