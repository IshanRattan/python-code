
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model

from helper import save_config, save_mapping
from data_builder import DataBuilder
from configs.config import mapping
from encoder import Encoder
from decoder import Decoder

import pandas as pd
import numpy as np


# Has single column "sentence" with one sentence per row
sentence_df = pd.read_csv(mapping['inputs']['CSV_PATH'])
sentences = DataBuilder.filter_data(sentence_df)

input_texts, target_texts_inputs, target_texts = DataBuilder.build_data(sentences)
print("num samples:", len(input_texts))

# Tokenize the inputs
tokenizer_inputs = DataBuilder.fit_tokenizer(input_texts, max_words=mapping['train']['MAX_NUM_WORDS'])
input_sequences = DataBuilder.get_sequences(tokenizer_inputs, input_texts)
save_mapping(tokenizer_inputs, mapping['inputs']['save_paths']['INPUTS_TOKENIZER'])

# Word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

num_words_input = len(word2idx_inputs) + 1
max_len_input = max(len(s) for s in input_sequences)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_len_input)

tokenizer_outputs = DataBuilder.fit_tokenizer(target_texts + target_texts_inputs, mapping['train']['MAX_NUM_WORDS'],
                    filter_spl_char=False)
target_sequences_inputs = DataBuilder.get_sequences(tokenizer_outputs, target_texts_inputs)
target_sequences = DataBuilder.get_sequences(tokenizer_outputs, target_texts)
save_mapping(tokenizer_outputs, mapping['inputs']['save_paths']['OUTPUTS_TOKENIZER'])

word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1

max_len_target = max(len(s) for s in target_sequences)
mapping['predict']['MAX_SEQ_LEN_TARGET'] = max_len_target
mapping['predict']['MAX_WORDS_TARGET'] = num_words_output
mapping['predict']['MAX_LEN_INPUT'] = max_len_input

decoder_input_data = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
decoder_target_data = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

decoder_targets_one_hot = np.zeros((len(input_texts), max_len_target, num_words_output), dtype='float32')
for i, d in enumerate(decoder_target_data):
    for t, token in enumerate(d):
        if token != 0: decoder_targets_one_hot[i, t, token] = 1

# Build Encoder
encoder = Encoder(max_len_input, num_words_input)
encoder_inputs, encoder_states = encoder.build()

# Build decoder using `encoder_states` as initial state.
decoder = Decoder(max_len_target, num_words_output)
decoder_inputs, decoder_outputs = decoder.build(encoder_states)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_targets_one_hot,
    batch_size=mapping['train']['BATCH_SIZE'],
    epochs=mapping['train']['EPOCHS'],
    validation_split=mapping['train']['VALIDATION_SPLIT'],
)

# Save model
save_config(mapping, mapping['inputs']['save_paths']['CONFIG_JSON'])
model.save(mapping['inputs']['save_paths']['MODEL_SAVEPATH'])


