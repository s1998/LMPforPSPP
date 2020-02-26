from lmcodes.allimports import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import *
from random import random

### Data Retrieval
import numpy as np
import pandas as pd
import copy
import random

def get_data(train_data,
             test_data,
             lm_pretrain=False,
             tokenizer_encoder=None,
             maxlen_seq=700, # protein residues padded to 700
             ratio = 0.1,
             r = 700,
             f = 57):  # number of features for each residue

    cb6133filtered = train_data
    cb513          = test_data
    print(cb6133filtered.shape)
    print(cb513.shape)

    residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq', 'B']
    q8_list      = list('LBEGIHST') + ['NoSeq']

    columns = ["id", "len", "input", "profiles", "expected"]

    def get_data(arr, bounds=None):
        if bounds is None: bounds = range(len(arr))
        
        data = [None for i in bounds]
        for i in bounds:
            seq, q8, profiles = '', '', []
            for j in range(r):
                jf = j*f
                
                # Residue convert from one-hot to decoded
                residue_onehot = arr[i,jf+0:jf+22]
                residue = residue_list[np.argmax(residue_onehot)]

                # Q8 one-hot encoded to decoded structure symbol
                residue_q8_onehot = arr[i,jf+22:jf+31]
                residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

                if residue == 'NoSeq': break      # terminating sequence symbol
                if residue_q8 == 'NoSeq':
                    print("error")
                nc_terminals = arr[i,jf+31:jf+33] # nc_terminals = [0. 0.]
                sa = arr[i,jf+33:jf+35]           # sa = [0. 0.]
                profile = arr[i,jf+35:jf+57]      # profile features
                
                seq += residue # concat residues into amino acid sequence
                q8  += residue_q8 # concat secondary structure into secondary structure sequence
                profiles.append(profile)
            
            data[i] = [str(i+1), len(seq), seq, np.array(profiles), q8]
        
        return pd.DataFrame(data, columns=columns)

    ### Train-test Specification
    train_df = get_data(cb6133filtered)
    test_df  = get_data(cb513)

    # Maps the sequence to a one-hot encoding
    def onehot_to_seq(oh_seq, index):
        s = ''
        for o in oh_seq:
            i = np.argmax(o)
            if i != 0:
                s += index[i]
            else:
                break
        return s

    def seq2onehot(seq, n):
        out = np.zeros((len(seq), maxlen_seq, n))
        for i in range(len(seq)):
            for j in range(maxlen_seq):
                out[i, j, seq[i, j]] = 1
        return out

    # Computes and returns the n-grams of a particualr sequence, defaults to trigrams
    def seq2ngrams(seqs, n = 1):
        return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

    # Loading and converting the inputs to trigrams
    train_input_seqs, train_target_seqs = \
        train_df[['input', 'expected']][(train_df.len.astype(int) <= maxlen_seq)].values.T
    train_input_grams = seq2ngrams(train_input_seqs)

    # Same for test
    test_input_seqs = test_df['input'].values.T
    test_input_grams = seq2ngrams(test_input_seqs)

    # Initializing and defining the tokenizer encoders and decoders based on the train set
    if lm_pretrain:
        tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(train_input_grams)

    if lm_pretrain:
        train_target_seqs = copy.deepcopy(train_input_grams)

    tokenizer_decoder = Tokenizer(char_level = True)
    tokenizer_decoder.fit_on_texts(train_target_seqs)

    # Using the tokenizer to encode and decode the sequences for use in training
    # Inputs
    train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
    train_input_data = sequence.pad_sequences(train_input_data,
                                              maxlen = maxlen_seq, padding='post')

    # Targets
    if lm_pretrain:
        train_target_data = tokenizer_encoder.texts_to_sequences(train_target_seqs)
        train_target_data = sequence.pad_sequences(train_target_data,
                                                   maxlen = maxlen_seq, padding='post')

        train_input_data = copy.deepcopy(train_input_data)
        skip = 1.0 / ratio
        curr = int(random.random() * skip)
        for i in range(len(train_input_data)):
            for t in range(maxlen_seq):
                if t != curr:
                    train_target_data[i, t] = 0
                elif train_input_data[i, t] != 0:
                    train_input_data[i, t] = len(tokenizer_encoder.word_index) + 1
                    curr += skip
        train_target_data[0][0] = 22
        train_target_data = to_categorical(train_target_data)
    else:
        train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
        train_target_data = sequence.pad_sequences(train_target_data,
                                                   maxlen = maxlen_seq, padding='post')
        train_target_data = to_categorical(train_target_data)

    # Use the same tokenizer defined on train for tokenization of test
    test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
    test_input_data = sequence.pad_sequences(test_input_data,
                                             maxlen = maxlen_seq, padding='post')

    # Computing the number of words and number of tags for the keras model
    if lm_pretrain:
        n_words = len(tokenizer_encoder.word_index) + 2
        n_tags = len(tokenizer_encoder.word_index) + 2
    else:
        n_words = len(tokenizer_encoder.word_index) + 2
        n_tags = len(tokenizer_decoder.word_index) + 1

    train_input_data_alt = train_input_data
    train_input_data = seq2onehot(train_input_data, n_words)
    train_profiles = train_df.profiles.values

    test_input_data_alt = test_input_data
    test_input_data = seq2onehot(test_input_data, n_words)
    test_profiles = test_df.profiles.values

    train_profiles_np = np.zeros((len(train_profiles), maxlen_seq, n_words))
    for i, profile in enumerate(train_profiles):
        for j in range(profile.shape[0]):
            for k in range(profile.shape[1]):
                train_profiles_np[i, j, k] = profile[j, k]

    test_profiles_np = np.zeros((len(test_profiles), maxlen_seq, n_words))
    for i, profile in enumerate(test_profiles):
        for j in range(profile.shape[0]):
            for k in range(profile.shape[1]):
                test_profiles_np[i, j, k] = profile[j, k]

    if lm_pretrain:
        train_profiles_np *= 0
        test_profiles_np *= 0

    return ([train_input_data, train_input_data_alt, train_profiles_np],
            train_target_data, 
            [test_input_data, test_input_data_alt, test_profiles_np],
            test_df['expected'],
            tokenizer_encoder,
            tokenizer_decoder,
            n_words,
            n_tags)

    # cb6133filtered = np.load(train_path)
    # cb513          = np.load(test_path)

    # cb6133filtered = train_data
    # cb513          = test_data

    # print(cb6133filtered.shape)
    # print(cb513.shape)

    # residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq', 'B']
    # q8_list      = list('LBEGIHST') + ['NoSeq']

    # columns = ["id", "len", "input", "profiles", "expected"]

    # def get_data(arr, bounds=None):
        
    #     if bounds is None: bounds = range(len(arr))
        
    #     data = [None for i in bounds]
    #     for i in bounds:
    #         seq, q8, profiles = '', '', []
    #         for j in range(r):
    #             jf = j*f
                
    #             # Residue convert from one-hot to decoded
    #             residue_onehot = arr[i,jf+0:jf+22]
    #             residue = residue_list[np.argmax(residue_onehot)]

    #             # Q8 one-hot encoded to decoded structure symbol
    #             residue_q8_onehot = arr[i,jf+22:jf+31]
    #             residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

    #             if residue == 'NoSeq': break      # terminating sequence symbol

    #             nc_terminals = arr[i,jf+31:jf+33] # nc_terminals = [0. 0.]
    #             sa = arr[i,jf+33:jf+35]           # sa = [0. 0.]
    #             profile = arr[i,jf+35:jf+57]      # profile features
                
    #             seq += residue # concat residues into amino acid sequence
    #             q8  += residue_q8 # concat secondary structure into secondary structure sequence
    #             profiles.append(profile)
            
    #         data[i] = [str(i+1), len(seq), seq, np.array(profiles), q8]
        
    #     return pd.DataFrame(data, columns=columns)

    # ### Train-test Specification
    # train_df = get_data(cb6133filtered)
    # test_df  = get_data(cb513)

    # # Maps the sequence to a one-hot encoding
    # def onehot_to_seq(oh_seq, index):
    #     s = ''
    #     for o in oh_seq:
    #         i = np.argmax(o)
    #         if i != 0:
    #             s += index[i]
    #         else:
    #             break
    #     return s

    # def seq2onehot(seq, n):
    #     out = np.zeros((len(seq), maxlen_seq, n))
    #     for i in range(len(seq)):
    #         for j in range(maxlen_seq):
    #             out[i, j, seq[i, j]] = 1
    #     return out

    # # Computes and returns the n-grams of a particualr sequence, defaults to trigrams
    # def seq2ngrams(seqs, n = 1):
    #     return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

    # # Loading and converting the inputs to trigrams
    # train_input_seqs, train_target_seqs = \
    #     train_df[['input', 'expected']][(train_df.len.astype(int) <= maxlen_seq)].values.T
    # train_input_grams = seq2ngrams(train_input_seqs)

    # # Same for test
    # test_input_seqs = test_df['input'].values.T
    # test_input_grams = seq2ngrams(test_input_seqs)

    # # Initializing and defining the tokenizer encoders and decoders based on the train set
    # print(train_input_grams[0])
    # print(train_input_grams)
    # tokenizer_encoder = Tokenizer()
    # tokenizer_encoder.fit_on_texts(train_input_grams)
    # tokenizer_decoder = Tokenizer(char_level = True)
    # tokenizer_decoder.fit_on_texts(train_target_seqs)

    # # Using the tokenizer to encode and decode the sequences for use in training
    # # Inputs
    # train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
    # train_input_data = sequence.pad_sequences(train_input_data,
    #                                           maxlen = maxlen_seq, padding='post')

    # # Targets
    # train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
    # train_target_data = sequence.pad_sequences(train_target_data,
    #                                            maxlen = maxlen_seq, padding='post')
    # train_target_data = to_categorical(train_target_data)

    # # Use the same tokenizer defined on train for tokenization of test
    # test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
    # test_input_data = sequence.pad_sequences(test_input_data,
    #                                          maxlen = maxlen_seq, padding='post')

    # # Computing the number of words and number of tags for the keras model
    # n_words = len(tokenizer_encoder.word_index) + 1
    # n_tags = len(tokenizer_decoder.word_index) + 1

    # train_input_data_alt = train_input_data
    # train_input_data = seq2onehot(train_input_data, n_words)
    # train_profiles = train_df.profiles.values

    # test_input_data_alt = test_input_data
    # test_input_data = seq2onehot(test_input_data, n_words)
    # test_profiles = test_df.profiles.values

    # train_profiles_np = np.zeros((len(train_profiles), maxlen_seq, 22))
    # for i, profile in enumerate(train_profiles):
    #     for j in range(profile.shape[0]):
    #         for k in range(profile.shape[1]):
    #             train_profiles_np[i, j, k] = profile[j, k]

    # test_profiles_np = np.zeros((len(test_profiles), maxlen_seq, 22))
    # for i, profile in enumerate(test_profiles):
    #     for j in range(profile.shape[0]):
    #         for k in range(profile.shape[1]):
    #             test_profiles_np[i, j, k] = profile[j, k]

    # return ([train_input_data, train_input_data_alt, train_profiles_np],
    #         train_target_data, 
    #         [test_input_data, test_input_data_alt, test_profiles_np],
    #         test_df['expected'],
    #         tokenizer_encoder,
    #         tokenizer_decoder)
