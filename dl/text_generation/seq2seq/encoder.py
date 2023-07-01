
from keras.layers import Input, Embedding, LSTM
from configs.config import mapping


class Encoder():
    def __init__(self, max_len_input, num_words_input):
        self.max_len_input = max_len_input
        self.num_words_input = num_words_input

    def build(self) -> (Input, list):
        encoder_inputs = Input(shape=(self.max_len_input,))
        embedding_enc = Embedding(self.num_words_input, mapping['train']['EMBEDDING_DIM'])
        embedded_inputs_enc = embedding_enc(encoder_inputs)
        encoder = LSTM(mapping['train']['LATENT_DIM'], return_state=True, dropout=.2)
        encoder_outputs, state_h, state_c = encoder(embedded_inputs_enc)
        encoder_states = [state_h, state_c]
        return encoder_inputs, encoder_states