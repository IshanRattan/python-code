

mapping = {'train' :
               {'BATCH_SIZE' : 32,
                'EPOCHS' : 20,
                'LATENT_DIM' : 128,
                'MAX_NUM_WORDS' : 20000,
                'EMBEDDING_DIM' : 200,
                'VALIDATION_SPLIT' : .2},

          'predict' :
              {'MAX_SEQ_LEN_TARGET' : None,
               'MAX_WORDS_TARGET' : None,
               'MAX_LEN_INPUT' : None},

          'inputs' :
               {'save_paths' :
                    {'INPUTS_TOKENIZER' : 'tokenizers/input_tokens.json',
                     'OUTPUTS_TOKENIZER' : 'tokenizers/output_tokens.json',
                     'CONFIG_JSON' : 'tokenizers/config.json',
                     'MODEL_SAVEPATH' : 'models/seq2seq'},
                'CSV_PATH': 'datasets/sentences.csv',
                'NUM_ROWS' : 5000,
                'MIN_WORDS' : 8,
                'MIN_SENT_LEN' : 70}}

