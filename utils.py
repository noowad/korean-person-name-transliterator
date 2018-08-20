# coding:utf-8
from hyperparams import Hyperparams as hp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_load import load_vocab
import os
import codecs


def dec_input(labels):
    x = tf.fill([tf.shape(labels)[0], 1], hp.START_TOKEN)
    x = tf.to_int32(x)
    return tf.concat([x, labels[:, :-1]], 1)


def plot_alignment(alignment, epoch, eng_name, kor_name):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    epoch: epochs
    """
    _, x_i2w, _, y_i2w = load_vocab()
    non_padded_eng_name = eng_name[np.nonzero(eng_name)]
    non_padded_kor_name = kor_name[np.nonzero(kor_name)]
    txt_eng_name = " ".join(x_i2w[idx] for idx in non_padded_eng_name).encode('utf-8').split('E')[0]
    txt_kor_name = " ".join(y_i2w[idx] for idx in non_padded_kor_name).encode('utf-8')
    txt_kor_name = txt_kor_name.replace('S', '').replace('E', '')
    fig, ax = plt.subplots()
    im = ax.imshow(alignment[:non_padded_eng_name.shape[0] - 1, :non_padded_kor_name.shape[0] - 1], cmap='Greys')

    fig.colorbar(im)
    plt.title('{} epochs \n {} \n {}'.format(epoch, txt_eng_name, txt_kor_name))
    plt.savefig('{}/alignment_{}k.png'.format(hp.logdir + '/' + hp.modelname, epoch), format='png')
    plt.close()


def make_oversampling():
    files = [f for f in os.listdir('./datas/nations') if not f.startswith('.')]
    for f in files:
        lines = codecs.open('./datas/nations/' + f, 'r', 'utf-8').read().splitlines()
        with codecs.open('./datas/50_oversampled_nations/' + f, 'w', 'utf-8') as fout:
            if len(lines) < 50:
                flag = True
                oversampled_lines = []
                while flag:
                    for line in lines:
                        oversampled_lines.append(line)
                        if len(oversampled_lines) == 50:
                            flag = False
                            break
                for line in oversampled_lines:
                    fout.write(line + '\n')
            else:
                for line in lines:
                    fout.write(line + '\n')
