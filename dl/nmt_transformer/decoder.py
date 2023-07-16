
from keras.layers import Layer, Dropout, LayerNormalization, Dense, Embedding
from nmt_transformer.multihead_attention import MultiHeadAttention
from nmt_transformer.pos_encoding import PositionalEncoding

import tensorflow as tf

class DecoderLayer(Layer):
    def __init__(self, FFN_units, nb_proj, dropout):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.dropout = dropout
        self.nb_proj = nb_proj


    def build(self, input_shape: tf.Tensor):
        self.d_model = input_shape[-1]
        self.multi_head_attention_1 = MultiHeadAttention(self.nb_proj)
        self.dropout_2 = Dropout(rate=self.dropout)
        self.norm_2 = LayerNormalization(epsilon=1e-6)

        self.multi_head_attention_2 = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = Dropout(rate=self.dropout)
        self.norm_1 = LayerNormalization(epsilon=1e-6)

        self.dense_1 = Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = Dense(units=self.d_model)
        self.dropout_3 = Dropout(rate=self.dropout)
        self.norm_3 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor, enc_outputs, mask_1: tf.Tensor, mask_2: tf.Tensor, training: bool) -> tf.Tensor:
        attention = self.multi_head_attention_1(inputs,
                                                inputs,
                                                inputs,
                                                mask_1)
        attention = self.dropout_1(attention, training)
        attention = self.norm_1(attention + inputs)

        attention_2 = self.multi_head_attention_2(attention,
                                                  enc_outputs,
                                                  enc_outputs,
                                                  mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention)

        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)

        return outputs

class Decoder(Layer):
    def __init__(self,
                 nb_layers: int,
                 FFN_units: int,
                 nb_proj: int,
                 dropout: float,
                 vocab_size: int,
                 d_model: int,
                 name='decoder'):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.nb_layers = nb_layers

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = Dropout(rate=dropout)

        self.dec_layers = [DecoderLayer(FFN_units, nb_proj, dropout) for _ in range(nb_layers)]

    def call(self, inputs: tf.Tensor, enc_outputs, mask_1: tf.Tensor, mask_2: tf.Tensor, training: bool) -> tf.Tensor:
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs,
                                         enc_outputs,
                                         mask_1,
                                         mask_2,
                                         training)

        return outputs