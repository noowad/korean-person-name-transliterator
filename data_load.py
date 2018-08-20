# -*- coding: utf-8 -*-
# /usr/bin/python2
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
import codecs
import os


# E for empty, S for end of Sentence
def load_vocab():
    def _make_dicts(fpath):
        tokens = [line for line in codecs.open(fpath, 'r', 'utf-8').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        return token2idx, idx2token

    eng_w2i, eng_i2w = _make_dicts('datas/eng_voca.txt')
    kor_w2i, kor_i2w = _make_dicts('datas/kor_voca.txt')
    return eng_w2i, eng_i2w, kor_w2i, kor_i2w


# text->index and padding
def create_data(X_txt, Y_txt):
    x_w2i, _, y_w2i, _, = load_vocab()
    # Index
    x_list, y_list = [], []

    for eng_name, kor_name in zip(X_txt, Y_txt):
        x = [x_w2i.get(eng, 1) for eng in eng_name + "E"]
        y = [y_w2i.get(kor, 1) for kor in kor_name + "E"]
        x_list.append(np.array(x))
        y_list.append(np.array(y))

    # Pad
    X_index = np.zeros([len(x_list), hp.max_len], np.int32)
    Y_index = np.zeros([len(y_list), hp.max_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X_index[i] = np.lib.pad(x, [0, hp.max_len - len(x)], 'constant', constant_values=(0, 0))
        Y_index[i] = np.lib.pad(y, [0, hp.max_len - len(y)], 'constant', constant_values=(0, 0))
    return X_index, Y_index


def load_train_data(k=0):
    '''Loads vectorized input training data
    '''
    X_txt, Y_txt = [], []
    for names in codecs.open('datas/korean_train.txt', 'r', 'utf-8').read().splitlines():
        X_txt.append(names.split('\t')[0])
        Y_txt.append(names.split('\t')[1])

    X_index, Y_index = create_data(X_txt, Y_txt)

    return X_txt, X_index, Y_txt, Y_index


def load_evaluate_data(eval_mode="test"):
    '''Embeds and vectorize words in input corpus'''
    x_w2i, _, y_w2i, _ = load_vocab()
    X_txt, Y_txt = [], []
    if eval_mode == "validate":
        for names in codecs.open('datas/korean_validate.txt', 'r', 'utf-8').read().splitlines():
            X_txt.append(names.split('\t')[0])
            Y_txt.append(names.split('\t')[1])
    if eval_mode == "test":
        for names in codecs.open('datas/korean_test.txt', 'r', 'utf-8').read().splitlines():
            if len(names.split('\t')[0]) < 20 and len(names.split('\t')[1]) < 20:
                X_txt.append(names.split('\t')[0])
                Y_txt.append(names.split('\t')[1])

    X_index, Y_index = create_data(X_txt, Y_txt)

    return X_txt, X_index, Y_txt, Y_index
