from __future__ import print_function
import tensorflow as tf
import os
from tqdm import tqdm
from data_load import load_vocab, load_train_data
from hyperparams import Hyperparams as hp
from utils import plot_alignment
from data_load import *
from graph import Graph
import argparse

x_w2i, x_i2w, y_w2i, y_i2w = load_vocab()


def train():
    print("Graph loading......Model name:{}".format(hp.modelname))
    g = Graph()
    print("Data loading...")
    _, eng_names, _, kor_names = load_train_data()
    _, val_eng_names, _, val_kor_names = load_evaluate_data(eval_mode="validate")

    early_stopping_count = 0
    data_list = list(range(len(eng_names)))
    with g.graph.as_default():
        sv = tf.train.Saver()
        with tf.Session() as sess:
            # Initialize
            sess.run(tf.global_variables_initializer())
            best_valid_loss = 100000.
            for epoch in range(1, hp.num_epochs + 1):
                np.random.shuffle(data_list)
                # # Attention Plot per epochs
                # al = sess.run(g.alignments, {g.x: eng_names[data_list][:1],
                #                              g.y: kor_names[data_list][:1]})
                # plot_alignment(al[0], epoch - 1, eng_names[data_list][:1], kor_names[data_list][:1])
                # Train
                train_loss = 0
                num_batch = len(eng_names) / hp.batch_size
                for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    name_ids = data_list[step * hp.batch_size:step * hp.batch_size + hp.batch_size]
                    loss, gs = sess.run([g.train_op, g.global_step],
                                        {g.x: eng_names[name_ids],
                                         g.y: kor_names[name_ids]})
                    train_loss += loss
                    if step % 20 == 0:
                        print('\t step:{} train_loss:{:.3f}'.format(gs, loss))
                train_loss /= num_batch

                # Validation
                valid_loss = 0.
                for idx in range(0, len(val_eng_names), hp.batch_size):
                    v_loss = sess.run(g.mean_loss, {g.x: val_eng_names[idx:idx + hp.batch_size],
                                                    g.y: val_kor_names[idx:idx + hp.batch_size]})
                    valid_loss += v_loss
                valid_loss /= len(val_eng_names) / hp.batch_size
                print("[epoch{}] train_loss={:.3f} validate_loss={:.3f} ".format(epoch, train_loss, valid_loss))
                # Stopping
                if valid_loss <= best_valid_loss * 0.999:
                    best_valid_loss = valid_loss
                    sv.save(sess, "logdir/" + hp.modelname + '/model.ckpt')
                else:
                    if hp.is_earlystopping:
                        early_stopping_count += 1
                        if early_stopping_count == 3:
                            print("Early Stopping...")
                            break


if __name__ == '__main__':
    if not os.path.exists('./logdir/' + hp.modelname):
        os.makedirs('./logdir/' + hp.modelname)
    train()
    print("Done")
