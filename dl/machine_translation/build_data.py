
from keras.preprocessing.sequence import pad_sequences
from configs import config

import tensorflow_datasets as tfds
import tensorflow as tf
import re


def load_data(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as infile:
        return infile.read()

def clean_corpus(corpus: str, non_breaking_prefix: list) -> list:
    for prefix in non_breaking_prefix:
        corpus = corpus.replace(prefix, prefix + '###')

    corpus = re.sub(r'\.(?=[0-9]|[a-z]|[A-Z])', '.###', corpus)
    corpus = re.sub(r'\.###', '', corpus)
    corpus = re.sub(r'  +', ' ', corpus)
    return corpus.split('\n')

def preprocess_data() -> (tf.data.Dataset, int, int):

    input_language_dataset = load_data(config.path_en_input)
    target_language_dataset = load_data(config.path_fr_input)

    non_breaking_prefix_input_lang = load_data(config.path_en_nonbreak)
    non_breaking_prefix_input_lang = [' ' + prefix + '.' for prefix in non_breaking_prefix_input_lang.split('\n')]

    non_breaking_prefix_target_lang = load_data(config.path_fr_nonbreak)
    non_breaking_prefix_target_lang = [' ' + prefix + '.' for prefix in non_breaking_prefix_target_lang.split('\n')]

    input_language_preprocessed = clean_corpus(input_language_dataset, non_breaking_prefix_input_lang)
    target_language_preprocessed = clean_corpus(target_language_dataset, non_breaking_prefix_target_lang)

    # init tokenizer for en(input) & fr(target) language
    tokenizer_input_lang = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(input_language_preprocessed, target_vocab_size=2**13)
    tokenizer_target_lang = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(target_language_preprocessed, target_vocab_size=2**13)

    vocab_size_input_lang = tokenizer_input_lang.vocab_size + 2
    vocab_size_target_lang = tokenizer_target_lang.vocab_size + 2

    model_inputs = [[vocab_size_input_lang - 2] + tokenizer_input_lang.encode(sentence) + [vocab_size_input_lang - 1]
              for sentence in input_language_preprocessed]

    model_outputs = [[vocab_size_target_lang - 2] + tokenizer_target_lang.encode(sentence) + [vocab_size_target_lang - 1]
              for sentence in target_language_preprocessed]

    # remove sentences > MAX_LENGTH in model_inputs
    idx_to_remove = [count for count, sentence in enumerate(model_inputs) if len(sentence) > config.MAX_LENGTH]
    for idx in reversed(idx_to_remove):
        del model_inputs[idx]
        del model_outputs[idx]

    # remove sentences > MAX_LENGTH in model_outputs
    idx_to_remove = [count for count, sentence in enumerate(model_outputs) if len(sentence) > config.MAX_LENGTH]
    for idx in reversed(idx_to_remove):
        del model_inputs[idx]
        del model_outputs[idx]

    # padding
    model_inputs = pad_sequences(model_inputs, value=0, padding='post', maxlen=config.MAX_LENGTH)
    model_outputs = pad_sequences(model_outputs, value=0, padding='post', maxlen=config.MAX_LENGTH)

    dataset = tf.data.Dataset.from_tensor_slices((model_inputs, model_outputs))
    dataset = dataset.cache()
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, vocab_size_input_lang, vocab_size_target_lang

