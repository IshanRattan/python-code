
from keras.layers import Layer

import tensorflow as tf
import numpy as np


# Implementing the following formula :
# PE(pos, 2i) = sin(pos / 10000 ** (2i / d_model) = sin() for even dimension
# PE(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model) = cos() for odd dimension
class PositionalEncoding(Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos : np.ndarray, i : np.ndarray, d_model : int) -> np.ndarray:
        '''Calculate product of position vector(pos) with angles vector(angles)'''

        # dims(params) : pos = (seq_length, 1) , i = (1, d_model), d_model = int

        # dims(angles) : np array of length d_model
        angles = 1 / np.power(10000., (2 * (i//2)) / np.float(d_model))

        # matrix of dims (MAX_LENGTH, d_model)
        return pos * angles

    def call(self, inputs : tf.Tensor) -> tf.Tensor:
        '''Calculate positional encoding and add them to the original inputs'''

        # seq_length same as allowed MAX_LENGTH of sentence
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]

        # np.newaxis here creates a column-wise or row-wise 2-d vector
        # np.arange(seq_length)[:, np.newaxis] added column dimension
        # np.arange(d_model)[np.newaxis, :] added row dimension
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)

        # 0::2 specifies columns starting from 0 and adding a step of 2 i.e columns with even index
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        # 1::2 specifies columns starting from 1 and adding a step of 2 i.e columns with odd index
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        # Next step will create a 3-d vector by adding 1 dim in the start.
        # Now 3-d vector shape will be (1, MAX_LENGTH, d_model)
        pos_encoding = angles[np.newaxis, ...]

        # this step adds pos_encoding weights to each tensor, element wise
        # tensor shape (batch_size, MAX_LENGTH, d_model) & pos_encoding shape (1, MAX_LENGTH, d_model)
        # so if batch size is 2 then 3-d tensor shape : (2, MAX_LENTH, d_model)
        # [...]
        # [...]
        return inputs + tf.cast(pos_encoding, tf.float32)