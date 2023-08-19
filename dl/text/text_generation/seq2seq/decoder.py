
from keras.layers import Input, Embedding, LSTM, Dense
from configs.config import mapping

class Decoder():

    def __init__(self, max_len_target, num_words_output):
        self.max_len_target = max_len_target
        self.num_words_output = num_words_output

    def build(self, encoder_states: list) -> (Input, Dense):
        decoder_inputs = Input(shape=(self.max_len_target,))
        embedding_dec = Embedding(self.num_words_output, mapping['train']['EMBEDDING_DIM'])
        embedded_inputs_dec = embedding_dec(decoder_inputs)
        decoder_lstm = LSTM(mapping['train']['LATENT_DIM'], return_sequences=True, return_state=True, dropout=.2)
        decoder_outputs, _, _ = decoder_lstm(embedded_inputs_dec, initial_state=encoder_states)
        decoder_dense = Dense(self.num_words_output, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)
        return decoder_inputs, decoder_outputs