
import re

import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.sequence import pad_sequences

from transformer.configs import config


def load_data(path : str) -> str:
    with open(path, 'r', encoding='utf-8') as infile:
        return infile.read()

def clean_corpus(corpus, non_breaking_prefix):
    for prefix in non_breaking_prefix:
        corpus = corpus.replace(prefix, prefix + '###')

    corpus = re.sub(r'\.(?=[0-9]|[a-z]|[A-Z])', '.###', corpus)
    corpus = re.sub(r'\.###', '', corpus)
    corpus = re.sub(r'  +', ' ', corpus)
    return corpus.split('\n')

def preprocess_data():
    europarl_en = load_data(config.path_en_input)
    europarl_fr = load_data(config.path_fr_input)

    non_breaking_prefix_en = load_data(config.path_en_nonbreak)
    non_breaking_prefix_en = [' ' + prefix + '.' for prefix in non_breaking_prefix_en.split('\n')]

    non_breaking_prefix_fr = load_data(config.path_fr_nonbreak)
    non_breaking_prefix_fr = [' ' + prefix + '.' for prefix in non_breaking_prefix_fr.split('\n')]

    corpus_en = clean_corpus(europarl_en, non_breaking_prefix_en)[:100]
    corpus_fr = clean_corpus(europarl_fr, non_breaking_prefix_fr)[:100]

    # init tokenizer for en(input) & fr(output) language
    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus_en, target_vocab_size=2**13)
    tokenizer_fr = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus_fr, target_vocab_size=2**13)

    VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2
    VOCAB_SIZE_FR = tokenizer_fr.vocab_size + 2

    inputs = [[VOCAB_SIZE_EN - 2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_EN - 1]
              for sentence in corpus_en]

    outputs = [[VOCAB_SIZE_FR - 2] + tokenizer_fr.encode(sentence) + [VOCAB_SIZE_FR - 1]
              for sentence in corpus_fr]

    MAX_LENGTH = config.MAX_LENGTH
    idx_to_remove = [count for count, sentence in enumerate(inputs) if len(sentence) > MAX_LENGTH]
    for idx in reversed(idx_to_remove):
        del inputs[idx]
        del outputs[idx]

    idx_to_remove = [count for count, sentence in enumerate(outputs) if len(sentence) > MAX_LENGTH]
    for idx in reversed(idx_to_remove):
        del inputs[idx]
        del outputs[idx]

    # padding
    inputs = pad_sequences(inputs, value=0, padding='post', maxlen=MAX_LENGTH)
    outputs = pad_sequences(outputs, value=0, padding='post', maxlen=MAX_LENGTH)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    dataset = dataset.cache()
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, VOCAB_SIZE_EN, VOCAB_SIZE_FR

