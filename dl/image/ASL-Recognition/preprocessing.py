
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


# Load pre-shuffled training and test datasets
def one_hot_encode(x_train, y_train, y_test):

    # Number of A's in the training dataset
    num_A_train = sum(y_train==0)
    # Number of B's in the training dataset
    num_B_train = sum(y_train==1)
    # Number of C's in the training dataset
    num_C_train = sum(y_train==2)

    # Number of A's in the test dataset
    num_A_test = sum(y_test==0)
    # Number of B's in the test dataset
    num_B_test = sum(y_test==1)
    # Number of C's in the test dataset
    num_C_test = sum(y_test==2)

    # Print statistics about the dataset
    print("Training set:")
    print("\tA: {}, B: {}, C: {}".format(num_A_train, num_B_train, num_C_train))
    print("Test set:")
    print("\tA: {}, B: {}, C: {}".format(num_A_test, num_B_test, num_C_test))

    # One-hot encode the training labels
    y_train_OH = to_categorical(y_train)

    # One-hot encode the test labels
    y_test_OH = to_categorical(y_test)

    return y_train_OH, y_test_OH

