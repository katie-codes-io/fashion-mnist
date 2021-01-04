import os
import matplotlib.pyplot as plt


def plot_img(data, name=None):
    """
    Plot a singular image in greyscale

    :param data: image data to plot
    :param name: optional filename, if None the image will be plotted on screen instead of in file
    :return:
    """
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    if name is not None:
        if not os.path.exists('images'):
            os.makedirs('images')

        if not name.endswith('.png'):
            name = name + '.png'

        filename = os.path.join('images', name)

        plt.tight_layout()
        plt.savefig(filename, dpi=180)
    else:
        plt.show()
