
from keras.layers import Layer, Dense
import tensorflow as tf


class MultiHeadAttention(Layer):

    def __init__(self, nb_proj):
        super(MultiHeadAttention, self).__init__()
        self.nb_proj = nb_proj

    def build(self, input_shape : tf.TensorShape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.nb_proj == 0
        self.d_proj = self.d_model // self.nb_proj

        self.query_lin = Dense(units=self.d_model)
        self.key_lin = Dense(units=self.d_model)
        self.value_lin = Dense(units=self.d_model)
        self.final_lin = Dense(units=self.d_model)

    def split_proj(self, inputs : tf.Tensor, batch_size : tf.Tensor) -> tf.Tensor:
        # inputs dim : (batch_size, seq_length, d_model)

        shape = (batch_size,
                 -1,
                 self.nb_proj,
                 self.d_proj)

        # splitted_inputs dim : (batch_size, seq_length, nb_proj, d_proj)
        splitted_inputs = tf.reshape(inputs, shape=shape)

        # return dim : (batch_size, nb_proj, seq_length=MAX_LENGTH, d_proj)
        return tf.transpose(splitted_inputs, perm=[0, 2, 1, 3])

    def call(self, queries : tf.Tensor,
             keys : tf.Tensor,
             values : tf.Tensor,
             mask : tf.Tensor) -> tf.Tensor:

        batch_size = tf.shape(queries)[0]

        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        # queries dim : (batch_size, nb_proj, MAX_LENGTH, d_proj)
        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        attention = scaled_dot_product_attention(queries,
                                                 keys,
                                                 values,
                                                 mask)

        # attention dim : (batch_size, MAX_LEN, nb_proj, d_proj)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # concat_attention dim : (batch_size, MAX_LEN, d_model)
        concat_attention = tf.reshape(attention,
                                      shape=(batch_size,
                                             -1,
                                             self.d_model))

        outputs = self.final_lin(concat_attention)
        return outputs


# Implementing attention mechanism : Attention(Q, K, V) = softmax( (Q * K) / sqrt(dk) ) * V
# Q = queries matrix, K = keys matrix, V = values matrix, dk = dimension of keys matrix
def scaled_dot_product_attention(queries : tf.Tensor,
                                 keys : tf.Tensor,
                                 values : tf.Tensor,
                                 mask : tf.Tensor) -> tf.Tensor:

    # how to multiply 4-d tensors?
    product = tf.matmul(queries, keys, transpose_b=True)

    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)

    # scaled_product dim : (batch_size, nb_proj, MAX_LEN, MAX_LEN)
    scaled_product = product / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9)

    # attention dim : (batch_size, nb_proj, MAX_LEN, d_proj)
    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    return attention