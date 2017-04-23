# ======================================================================================================================
#                                                  | Weights |
# ======================================================================================================================
# load weights from the weight file
# save weights to the weight file
#
# ======================================================================================================================
# There will be 3 classes of weights:
#    .image recognition weights
#    .speech recognition weights
#    .sentiment analysis & context classification weights.
#
# brain storm on a feature to track updates to weights according to date and time.
# ======================================================================================================================

from numpy import *
import numpy as np


def load_weights(training_class):

    if training_class in "image recognition":
        weights = np.load('image_weights_file.dat')

    elif training_class in "speech recognition":
        weights = np.load('speech_weights_file.dat')

    elif training_class in "sentiment analysis & context classification":
        weights = np.load('language_weights_file.dat')

    else:
        print("training_class "+str(training_class)+" among the available training classes")
        return "load error {1}"

    return weights


def save_weights(training_class, weights):

    if training_class in "image recognition":
        weights.dump("image_weights_file.dat")

    elif training_class in "speech recognition":
        weights.dump("speech_weights_file.dat")

    elif training_class in " sentiment analysis & context classification":
        weights.dump("language_weights_file.dat")

    else:
        print("training_class " + str(training_class) + " among the available training classes")
        return "update error {1}"

    return 'weights updated'


