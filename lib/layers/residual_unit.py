from tensorflow import keras

class ResidualUnit(keras.layers.Layer):
    """
    Custom Keras Layer for ResNet Residual Unit.

    From:
    Géron, A. (2017). Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to
    build intelligent systems. Sebastopol, CA: O'Reilly Media. ISBN: 978-1491962299
    """

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.activation_func = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation_func,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation_func(Z + skip_Z)

    def get_config(self):
        config = super(ResidualUnit, self).get_config()
        config.update({"filters": self.filters, "strides": self.strides, "activation": self.activation})
        return config
