import sys
import tensorflow as tf

from models.cnn import CNN
from models.rnn import RNN


'''
Main file for MNIST data analysis. Use command line parameter to select model to run.
'''

# Firstly, set up GPU for run (sorts out a VRAM-related issue, see
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Now run the main code
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
    args = sys.argv[1:]

    # check length of arguments
    if len(args) == 0:
        print("=========================\n"
              "Correct syntax: python fashion-mnist.py -m [model] -s [model_name] -p [pretrained_model]\n"
              "To train a model and also save it: python fashion-mnist.py -m LeNet -s pretrained_LeNet\n"
              "To load a pre-trained model: python fashion-mnist.py -m LeNet -p pretrained_LeNet\n\n"
              "=========================\n"
              "Params\n"
              "=========================\n"
              "-m [model]: name of model to use, must match the name of a model in the list below (mandatory)\n"
              "-s [model_name]: save trained model by supplying a name (by default the trained model is not saved)."
              "If supplied, -p will be ignored\n"
              "-p [pretrained_model]: name of pretrained model to load\n\n"
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

        # set defaults
        selected_model = None
        save_name = None
        pretrained_model = None

        # check supplied params
        if "-m" in args:
            # check for supplied name
            m_index = args.index("-m") + 1
            try:
                selected_model = args[m_index]
            except IndexError:
                print(f"Supply a model name with -m. Exiting.")
                exit(0)
        else:
            # this is a mandatory param so exit if missing
            print(f"Supply a model name with -m. Exiting.")
            exit(0)

        # parse -s OR -p
        if "-s" in args:
            # check for supplied name
            s_index = args.index("-s") + 1
            try:
                save_name = args[s_index]
            except IndexError:
                print(f"Supply a name for saving model with -s. Exiting.")
                exit(0)

        elif "-p" in args:
            # check for supplied pretrained model
            p_index = args.index("-p") + 1
            try:
                pretrained_model = args[p_index]
            except IndexError:
                print(f"Supply a pretrained model with -p. Exiting.")
                exit(0)

        # check the supplied model
        if selected_model not in models.keys():
            print(f"Model unknown: {selected_model}. Exiting.")
            exit(0)

        else:
            print(f"Running model: {selected_model}")

            model = None
            if models.get(selected_model) == "CNN":
                model = CNN(selected_model, save_name=save_name, pretrained_model=pretrained_model)

            elif models.get(selected_model) == "RNN":
                model = RNN(selected_model, save_name=save_name, pretrained_model=pretrained_model)

            # proceed if we've got a valid model
            if model is not None:

                # train and evaluate
                if pretrained_model is None:
                    model.train()
                model.evaluate()

                # export the model for importing later (save time training from scratch)
                if save_name is not None:
                    model.export_model()
