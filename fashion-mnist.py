import sys

from models.cnn import CNN
from models.rnn import RNN


'''
Main file for MNIST data analysis. Use command line parameter to select model to run.
'''

if __name__ == "__main__":
    # available models
    models = {
        # CNN
        "AlexNet": "CNN",
        "LeNet": "CNN",
        "ResNet": "CNN",
        # RNN
        "LSTM": "RNN"
    }

    # get command line args
    args = sys.argv

    # check length of arguments
    if len(args) < 2 or len(args) > 3:
        print("=========================\n"
              "Correct syntax: python fashion-mnist.py <model> [pretrained_model]\n"
              "e.g. python fashion-mnist.py LeNet pretrained_LeNet\n\n"
              "=========================\n"
              "Params\n"
              "=========================\n"
              "model: name of model to use, must match the name of a model in the list below (mandatory)\n"
              "pretrained_model: name of pretrained model to load (optional)\n\n"
              "=========================\n"
              "Models available\n"
              "=========================\n"
              "Convolutional Neural Networks:\n"
              "AlexNet\n"
              "LeNet\n"
              "ResNet\n\n"
              "Recurrent Neural Networks:\n"
              "LSTM\n"
              "\n=========================")
        exit(0)

    else:
        selected_model = args[1]
        if selected_model not in models.keys():
            print(f"Model unknown: {selected_model}. Exiting.")
            exit(0)

        else:
            print(f"Running model: {selected_model}")

            # check for supplied pretrained model
            try:
                pretrained_model = args[2]
            except IndexError:
                pretrained_model = None

            model = None
            if models.get(selected_model) == "CNN":
                model = CNN(selected_model, pretrained_model=pretrained_model)

            elif models.get(selected_model) == "RNN":
                model = RNN(selected_model, pretrained_model=pretrained_model)

            # proceed if we've got a valid model
            if model is not None:

                # train and evaluate
                if pretrained_model is None:
                    model.train()
                model.evaluate()

                # export the model for importing later (save time training from scratch)
                if pretrained_model is None:
                    model.export_model()
