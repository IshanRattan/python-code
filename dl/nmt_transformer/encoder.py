
from keras.layers import Layer, Dropout, LayerNormalization, Dense, Embedding
from nmt_transformer.multihead_attention import MultiHeadAttention
from nmt_transformer.pos_encoding import PositionalEncoding

import tensorflow as tf

class EncodeLayer(Layer):

    def __init__(self, FFN_units: int, nb_proj: int, dropout: float):
        super(EncodeLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout = dropout

    def build(self, input_shape: tf.Tensor):
        self.d_model = input_shape[-1]

        self.multi_head_attention = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = Dropout(rate=self.dropout)
        self.norm_1 = LayerNormalization(epsilon=1e-6)

        self.dense_1 = Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = Dense(units=self.d_model)
        self.dropout_2 = Dropout(rate=self.dropout)
        self.norm_2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, training: bool) -> tf.Tensor:
        attention = self.multi_head_attention(inputs,
                                              inputs,
                                              inputs, mask)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)

        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.norm_2(outputs + attention)

        return outputs

class Encoder(Layer):

    def __init__(self,
                 nb_layers: int,
                 FFN_units: int,
                 nb_proj: int,
                 dropout: float,
                 vocab_size: int,
                 d_model: int,
                 name='encoder'):
        super(Encoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = Dropout(rate=dropout)
        self.enc_layers = [EncodeLayer(FFN_units, nb_proj, dropout) for _ in range(nb_layers)]

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, training: bool) -> tf.Tensor:
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.enc_layers[i](outputs, mask, training)
        return outputs
