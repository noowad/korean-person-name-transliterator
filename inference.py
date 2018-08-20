import tensorflow as tf
from data_load import load_evaluate_data, load_vocab
from hyperparams import Hyperparams as hp
from graph import Graph
import os
import hangul
import codecs
import argparse
import numpy as np


def eval():
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Load graph
    print("Graph loaded")
    print("Model name:{}".format(hp.modelname))
    # Load data
    print("Testing Data...")
    txt_src_names, idx_src_names, txt_tgt_names, _ = load_evaluate_data(eval_mode="test")

    x_w2i, x_i2w, y_w2i, y_i2w = load_vocab()

    g = Graph(is_training=False)
    with g.graph.as_default(), tf.Session() as sess:
        sv = tf.train.Saver()
        # Restore parameters
        print("Parameter Restoring...")
        sv.restore(sess, tf.train.latest_checkpoint(hp.logdir + '/' + hp.modelname))
        # Inference
        count = 0
        with open('./results/' + hp.modelname + '_result.txt', "w") as fout:
            for i in range(0, len(txt_src_names), hp.batch_size):
                batch_txt_src_names = txt_src_names[i:i + hp.batch_size]
                batch_idx_src_names = idx_src_names[i:i + hp.batch_size]
                batch_txt_tgt_names = txt_tgt_names[i:i + hp.batch_size]
                batch_predicted_ids = sess.run(g.pred_outputs,
                                               {g.x: batch_idx_src_names}).predicted_ids[:, :, :]

                for source, target, predicted_ids in zip(batch_txt_src_names, batch_txt_tgt_names, batch_predicted_ids):
                    print(str(count) + '\t' + source + '\t' + hangul.join_jamos(target))
                    count += 1
                    candidates = []
                    predicted_ids = predicted_ids.transpose(1, 0)
                    for pred in predicted_ids:
                        candidate = "".join(y_i2w[idx] for idx in pred).split("E")[0]
                        candidate = hangul.join_jamos(candidate)
                        candidates.append(candidate)

                    fout.write(source + '\t')
                    fout.write(hangul.join_jamos(target))
                    for candidate in candidates:
                        fout.write('\t')
                        fout.write(candidate.encode('utf-8'))
                    fout.write('\n')
                    fout.flush()


if __name__ == '__main__':
    eval()
