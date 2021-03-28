import numpy as np
import pandas as pd
import os
import re
import json
from tqdm import tqdm
import nltk

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

def load_data():
    question = []
    answer = []

    f = open("./data/wmt/train_en.m.txt", 'r')
    for i in range(4000000):
        question.append(f.readline())
    f.close()
    f = open("./data/wmt/train_de.txt", 'r')
    for i in range(4000000):
        answer.append(f.readline())
    f.close()

    return question, answer

def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, '', sentence)
        for word in sentence.split():
            words.append(word)

    return [word for word in words if word]

def prepro_like_morphlized(data):
    pattern = r'''(?x) ([A-Z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`]'''
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(nltk.regexp_tokenize(seq.replace(' ', ''), pattern))
        result_data.append(morphlized_seq)

    return result_data

def load_vocabulary(vocab_path, tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        question, answer = load_data()
        if tokenize_as_morph:
            question = prepro_like_morphlized(question)
            answer = prepro_like_morphlized(answer)

        data = []
        data.extend(question)
        data.extend(answer)
        words = data_tokenizer(data)
        words = list(set(words))

        words[:0] = MARKER

        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    char2idx, idx2char = make_vocabulary(vocabulary_list)

    return char2idx, idx2char, len(char2idx)

def make_vocabulary(vocabulary_list):
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    return char2idx, idx2char

def enc_processing(value, dictionary, tokenize_as_morph = False):
    sequences_input_index = []
    sequences_length = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []

        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]

        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length

def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)

    return np.asarray(sequences_output_index), sequences_length

def dec_target_processing(value, dictionary, tokenize_as_morph = False):
    sequences_target_index = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)



VOCAB_PATH = 'data_in/vocabulary.txt'
inputs, outputs = load_data()
char2idx, idx2char, vocab_size = load_vocabulary(VOCAB_PATH, False)

index_inputs, input_seq_len = enc_processing(inputs, char2idx, False)
index_outputs, output_seq_len = dec_output_processing(outputs, char2idx, False)
index_targets = dec_target_processing(outputs, char2idx, False)

data_configs = {}
data_configs['char2idx'] = char2idx
data_configs['idx2char'] = idx2char
data_configs['vocab_size'] = vocab_size
data_configs['pad_symbol'] = PAD
data_configs['std_symbol'] = STD
data_configs['end_symbol'] = END
data_configs['unk_symbol'] = UNK

DATA_IN_PATH = './data_in/'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIGS = 'data_configs.json'

np.save(open(DATA_IN_PATH+TRAIN_INPUTS, 'wb'), index_inputs)
np.save(open(DATA_IN_PATH+TRAIN_OUTPUTS, 'wb'), index_inputs)
np.save(open(DATA_IN_PATH+TRAIN_TARGETS, 'wb'), index_inputs)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'))
