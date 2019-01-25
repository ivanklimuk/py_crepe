import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


def load_train_data(path, labels_path=None):
    '''
    Load the train dataset with the labels:
    - either as the second value in each row in the read_csv
    - or as a separate file
    '''
    train_text = np.array(pd.read_csv(path, header=None))
    if labels_path:
        train_labels = np.array(pd.read_csv(labels_path, header=None))
    else:
        train_labels, train_text = train_text[:, 0], train_text[:, 1]
    train_labels = to_categorical(train_labels)

    return (train_text, train_labels)


def text_to_array(text, maxlen, vocabulary):
    '''
    Iterate over the loaded text and create a matrix of size (len(text), maxlen)
    Each character is encoded into a one-hot array later at the lambda layer.
    Chars not in the vocab are encoded as -1, into an all zero vector.
    '''

    text_array = np.zeros((len(text), maxlen), dtype=np.int)
    for row, line in enumerate(text):
        for column in range(min([len(line), maxlen])):
            text_array[row, column] = vocabulary.get(line[column], -1)  # if not in vocabulary_size, return -1

    return text_array


def create_vocabulary_set(ascii=True, digits=True, punctuation=True):
    '''
    This alphabet is 69 chars vs. 70 reported in the paper since they include two
    '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.
    '''

    alphabet = []
    if ascii:
        alphabet += list(string.ascii_lowercase)
    if digits:
        alphabet += list(string.digits)
    if punctuation:
        alphabet += list(string.punctuation) + ['\n']
    alphabet = set(alphabet)
    vocabulary_size = len(alphabet)
    vocabulary = {}
    reverse_vocabulary = {}
    for ix, t in enumerate(alphabet):
        vocabulary[t] = ix
        reverse_vocabulary[ix] = t

    return vocabulary, reverse_vocabulary, vocabulary_size, alphabet
