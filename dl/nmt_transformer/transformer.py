

from nmt_transformer.encoder import Encoder
from nmt_transformer.decoder import Decoder
from keras.layers import Dense
from keras import Model

import tensorflow as tf

class Transformer(Model):

    def __init__(self,
                 vocab_size_enc: int,
                 vocab_size_dec: int,
                 d_model: int,
                 nb_layers: int,
                 FFN_units: int,
                 nb_proj: int,
                 dropout: float,
                 name='nmt_transformer'):
        super(Transformer, self).__init__(name=name)

        self.encoder = Encoder(nb_layers,
                               FFN_units,
                               nb_proj,
                               dropout,
                               vocab_size_enc,
                               d_model)
        self.decoder = Decoder(nb_layers,
                               FFN_units,
                               nb_layers,
                               dropout,
                               vocab_size_dec,
                               d_model)
        self.last_linear = Dense(units=vocab_size_dec)

    def create_padding_mask(self, seq: tf.Tensor) -> tf.Tensor: # seg : (batch_size, seq_length)
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, enc_inputs: tf.Tensor, dec_inputs: tf.Tensor, training: tf.Tensor) -> tf.Tensor:
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(
            self.create_padding_mask(dec_inputs),
            self.create_look_ahead_mask(dec_inputs)
        )

        dec_mask_2 = self.create_padding_mask(enc_inputs)

        enc_outputs = self.encoder(enc_inputs,
                                   enc_mask,
                                   training)

        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   dec_mask_1,
                                   dec_mask_2,
                                   training)
        outputs = self.last_linear(dec_outputs)
        return outputs
