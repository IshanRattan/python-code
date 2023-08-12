
from nmt_transformer.build_data import load_data, clean_corpus, get_tokenizers
from nmt_transformer.transformer import Transformer
from nmt_transformer.train import CustomSchedule
from nmt_transformer.configs import config

import tensorflow as tf

def evaluate(inp_sentence: str) -> tf.Tensor:
    inp_sentence = \
        [VOCAB_SIZE_EN-2] + tokenizer_en.encode(inp_sentence) + [VOCAB_SIZE_EN-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    output = tf.expand_dims([VOCAB_SIZE_FR-2], axis=0)

    for _ in range(config.MAX_LENGTH):
        predictions = transformer(enc_input, output, False) # (1, seq_length, vocab_size_fr)

        prediction = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        if predicted_id == VOCAB_SIZE_FR-1:
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def translate(sentence: str):
    output = evaluate(sentence).numpy()

    predicted_sentence = tokenizer_fr.decode(
        [i for i in output if i < VOCAB_SIZE_FR-2]
    )

    print("Input: {}".format(sentence))
    print("Predicted translation: {}".format(predicted_sentence))

# Hyper-parameters
D_MODEL = 128 # 512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8, Attention heads
DROPOUT_RATE = 0.1 # 0.1

leaning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

input_language_dataset = load_data(config.path_en_input)
target_language_dataset = load_data(config.path_fr_input)

non_breaking_prefix_input_lang = load_data(config.path_en_nonbreak)
non_breaking_prefix_input_lang = [' ' + prefix + '.' for prefix in non_breaking_prefix_input_lang.split('\n')]

non_breaking_prefix_target_lang = load_data(config.path_fr_nonbreak)
non_breaking_prefix_target_lang = [' ' + prefix + '.' for prefix in non_breaking_prefix_target_lang.split('\n')]

input_language_preprocessed = clean_corpus(input_language_dataset, non_breaking_prefix_input_lang)[:config.TRAINING_DATA_POINTS]
target_language_preprocessed = clean_corpus(target_language_dataset, non_breaking_prefix_target_lang)[:config.TRAINING_DATA_POINTS]

# init tokenizer for en(input) & fr(target) language
tokenizer_en, tokenizer_fr = get_tokenizers(input_language_preprocessed, target_language_preprocessed)

VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2
VOCAB_SIZE_FR = tokenizer_fr.vocab_size + 2


transformer = Transformer(vocab_size_enc=VOCAB_SIZE_EN,
                          vocab_size_dec=VOCAB_SIZE_FR,
                          d_model=D_MODEL,
                          nb_layers=NB_LAYERS,
                          FFN_units=FFN_UNITS,
                          nb_proj=NB_PROJ,
                          dropout=DROPOUT_RATE)

checkpoint_path = config.checkpoint_path
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")

sentences = load_data(config.path_en_input).split('\n')
sentences = sentences[:10]
for sentence in sentences:
    translate(sentence)
    print('')

# Input: new jersey is sometimes quiet during autumn , and it is snowy in april .
# Predicted translation: new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
#
# Input: the united states is usually chilly during july , and it is usually freezing in november .
# Predicted translation: les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
#
# Input: california is usually quiet during march , and it is usually hot in june .
# Predicted translation: california est généralement calme en mars , et il est généralement chaud en juin .
#
# Input: the united states is sometimes mild during june , and it is cold in september .
# Predicted translation: les états-unis est parfois légère en juin , et il est froid en septembre .
#
# Input: your least liked fruit is the grape , but my least liked is the apple .
# Predicted translation: votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .