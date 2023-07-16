
# data processing
MAX_LENGTH = 20

BATCH_SIZE = 15
BUFFER_SIZE = 20000

TRAINING_DATA_POINTS = 50000 # -1 to use complete training data

path_en_input = '/Users/ishanrattan/Desktop/Study/text/machinetranslation/old/data/en.txt' #'/Users/ishanrattan/Desktop/Study/text/machinetranslation/nmt_transformer/data/fr-en/europarl-v7.fr-en.en'
path_fr_input = '/Users/ishanrattan/Desktop/Study/text/machinetranslation/old/data/fr.txt' #'/Users/ishanrattan/Desktop/Study/text/machinetranslation/nmt_transformer/data/fr-en/europarl-v7.fr-en.fr'

path_en_nonbreak = '/Users/ishanrattan/Desktop/Study/text/machinetranslation/nmt_transformer/data/fr-en/P85-Non-Breaking-Prefix.en'
path_fr_nonbreak = '/Users/ishanrattan/Desktop/Study/text/machinetranslation/nmt_transformer/data/fr-en/P85-Non-Breaking-Prefix.fr'


# model
checkpoint_path = "/Users/ishanrattan/Desktop/Study/text/machinetranslation/nmt_transformer/ckpt/"

