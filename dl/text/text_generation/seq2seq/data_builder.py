
from keras.preprocessing.text import Tokenizer
from configs.config import mapping

import pandas as pd


class DataBuilder():

    @classmethod
    def build_data(cls, sentences: list) -> (list, list, list):

        encoder_inputs = []
        decoder_inputs = []
        targets = []

        for sentence in sentences:
            tokens = sentence.split(' ')
            for i in range(3, len(tokens)):
                if i < len(tokens) - 2:
                    encoder_inputs.append(" ".join(tokens[ : i + 1]))
                    decoder_inputs.append('<sos> ' + " ".join(tokens[i + 1 : i + 3]))
                    targets.append(" ".join(tokens[i + 1 : i + 3]) + ' <eos>')

        return encoder_inputs, decoder_inputs, targets

    @classmethod
    def filter_data(cls, data: pd.DataFrame) -> list:
        data = data.dropna().drop_duplicates()
        data = data[data['sentence'].apply(lambda x: len(x) <= mapping['inputs']['MIN_SENT_LEN'])]
        data = data[data['sentence'].apply(lambda x: len(x.split(' ')) >= mapping['inputs']['MIN_WORDS'])]

        # For demo : Training on 5000 sentences
        return data.sentence.values[:mapping['inputs']['NUM_ROWS']]

    @classmethod
    def fit_tokenizer(cls, texts: list, max_words: int, filter_spl_char: bool = True) -> Tokenizer:
        if filter_spl_char:
            tokenizer = Tokenizer(num_words=max_words)
        else:
            tokenizer = Tokenizer(num_words=max_words, filters='')
        tokenizer.fit_on_texts(texts)
        return tokenizer

    @classmethod
    def get_sequences(cls, tokenizer : Tokenizer, texts : list) -> list:
        return tokenizer.texts_to_sequences(texts)