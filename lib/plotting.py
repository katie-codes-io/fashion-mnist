import os, math
import matplotlib.pyplot as plt


def plot_img(data, name=None):
    """
    Plot a singular image in greyscale

    :param data: image data to plot
    :param name: optional filename, if None the image will be plotted on screen instead of in file
    :return:
    """
    print("Plotting image")

    # plot the image in greyscale
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    # print to screen or file
    if name is not None:
        if not os.path.exists('images'):
            os.makedirs('images')

        if not name.endswith('.png'):
            name = name + '.png'

        filename = os.path.join('images', name)

        plt.tight_layout()
        plt.savefig(filename, dpi=100)
    else:
        plt.show()


def plot_examples(data, name=None, labels=None):
    """
    Plot a panel of example images (one per class) in greyscale

    :param data: image data to plot
    :param name: optional filename, if None the image will be plotted on screen instead of in file
    :param labels: optional list with image labels
    :return:
    """
    print("Plotting examples")

    # initialise the figure
    columns = 4
    rows = math.ceil(len(data) / columns)
    fig = plt.figure(figsize=(2 * columns, 2 * rows))

    # plot all the images in data
    for i in range(len(data)):
        ax = fig.add_subplot(rows, columns, i + 1)

        # optionally, add the labels
        if labels is not None:
            ax.title.set_text(labels[i])

        plt.imshow(data[i], cmap='gray')

    # print to screen or file
    if name is not None:
        if not os.path.exists('images'):
            os.makedirs('images')

        if not name.endswith('.png'):
            name = name + '.png'

        filename = os.path.join('images', name)

        plt.tight_layout()
        plt.savefig(filename, dpi=100)
    else:
        plt.show()
