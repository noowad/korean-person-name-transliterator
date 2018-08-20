import tensorflow as tf
from module import *
from network import *
from utils import *
from data_load import load_vocab
from rnn_wrappers import *

x_w2i, x_i2w, y_w2i, y_i2w = load_vocab()


class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len,))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len,))

            # it means sequence lengths without masking
            self.x_seq_len = tf.count_nonzero(self.x, 1, dtype=tf.int32)
            self.y_seq_len = tf.count_nonzero(self.y, 1, dtype=tf.int32)

            # Encoder
            with tf.variable_scope("enc-embed", initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)):
                # Embedding table
                self.x_embeddings = embed(len(x_w2i), hp.embed_size)
                self.x_embed = tf.nn.embedding_lookup(self.x_embeddings, self.x)
            with tf.variable_scope("encoder"):
                self.enc_outputs, self.enc_states = encode(self.x_embed, self.x_seq_len, is_training=is_training)

            # Decoder
            with tf.variable_scope("dec-embed", initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)):
                # Embedding table
                self.dec_embeddings = embed(len(y_w2i), hp.embed_size)
                self.dec_embed = tf.nn.embedding_lookup(self.dec_embeddings, dec_input(self.y))

            with tf.variable_scope("decoder"):
                if is_training:
                    # Training helper
                    self.helper = tf.contrib.seq2seq.TrainingHelper(self.dec_embed,
                                                                    self.y_seq_len)
                    # Decoder for training
                    self.train_outputs, self.alignments = training_decode(self.enc_outputs,
                                                                          self.x_seq_len,
                                                                          self.helper,
                                                                          len(y_w2i))

                    # loss
                    # for matching length
                    self.y_ = self.y[:, :tf.reduce_max(self.y_seq_len, -1)]
                    self.istarget = tf.to_float(tf.not_equal(self.y_, tf.zeros_like(self.y_)))  # masking
                    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_,
                                                                               logits=self.train_outputs)
                    self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                    # optimizing
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(hp.lr)
                    self.train_op = tf.contrib.training.create_train_op(self.mean_loss, self.optimizer,
                                                                        global_step=self.global_step)

                else:
                    # Decoder for inference (I use BeamSearchDecoder)
                    self.pred_outputs = inference_decode(self.enc_outputs,
                                                         self.x_seq_len,
                                                         self.dec_embeddings,
                                                         len(y_w2i))
