# coding:utf-8
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BeamSearchDecoder, BahdanauAttention, AttentionWrapper
from rnn_wrappers import *
from module import *


def encode(encoder_inputs, seq_len, is_training=None):
    prenet_out = prenet(encoder_inputs,
                        num_units=[hp.embed_size, hp.embed_size // 2],
                        dropout_rate=hp.dropout,
                        is_training=is_training)
    # Conv1D bank
    enc = conv1d_banks(prenet_out,
                       K=hp.encoder_banks,
                       num_units=hp.embed_size // 2,
                       norm_type=hp.norm_type,
                       is_training=is_training)

    # Max pooling
    enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")

    # Conv1D projections
    enc = conv1d(enc, hp.embed_size // 2, 3, scope="conv1d_1")
    enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                    activation_fn=tf.nn.relu, scope="norm1")
    enc = conv1d(enc, hp.embed_size // 2, 3, scope="conv1d_2")
    enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                    activation_fn=None, scope="norm2")
    enc += prenet_out  # residual connections
    enc_outputs, enc_states = gru(encoder_inputs, hp.embed_size // 2, seq_len, bidirection=True)
    return enc_outputs, enc_states


def training_decode(enc_outputs, seq_len, helper, out_dim):
    dec_prenet_outputs = DecoderPrenetWrapper(GRUCell(hp.embed_size),
                                              is_training=True,
                                              prenet_sizes=hp.embed_size,
                                              dropout_prob=hp.dropout)
    attention_mechanism = BahdanauAttention(hp.embed_size,
                                            enc_outputs,
                                            normalize=True,
                                            memory_sequence_length=seq_len,
                                            probability_fn=tf.nn.softmax)
    attn_cell = AttentionWrapper(dec_prenet_outputs,
                                 attention_mechanism,
                                 alignment_history=True,
                                 output_attention=False)
    concat_cell = ConcatOutputAndAttentionWrapper(attn_cell)
    decoder_cell = MultiRNNCell([OutputProjectionWrapper(concat_cell, hp.embed_size),
                                 ResidualWrapper(GRUCell(hp.embed_size)),
                                 ResidualWrapper(GRUCell(hp.embed_size))], state_is_tuple=True)

    output_cell = OutputProjectionWrapper(decoder_cell, out_dim)
    initial_state = output_cell.zero_state(batch_size=tf.shape(enc_outputs)[0], dtype=tf.float32)

    decoder = BasicDecoder(cell=output_cell,
                           helper=helper,
                           initial_state=initial_state)

    (outputs, _), last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=hp.max_len)
    # for attention plot
    alignments = tf.transpose(last_state[0].alignment_history.stack(), [1, 2, 0])
    return outputs, alignments


def inference_decode(enc_outputs, seq_len, embeddings, out_dim):
    tiled_enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, hp.beam_width)
    tiled_seq_len = tf.contrib.seq2seq.tile_batch(seq_len, hp.beam_width)

    beam_batch_size = tf.shape(tiled_enc_outputs)[0]
    # start tokens, end token
    start_tokens = tf.tile([hp.START_TOKEN], [beam_batch_size // hp.beam_width])
    end_token = hp.END_TOKEN

    dec_prenet_outputs = DecoderPrenetWrapper(GRUCell(hp.embed_size),
                                              is_training=False,
                                              prenet_sizes=hp.embed_size,
                                              dropout_prob=hp.dropout)
    attention_mechanism = BahdanauAttention(hp.embed_size,
                                            tiled_enc_outputs,
                                            normalize=True,
                                            memory_sequence_length=tiled_seq_len,
                                            probability_fn=tf.nn.softmax)
    attn_cell = AttentionWrapper(dec_prenet_outputs,
                                 attention_mechanism,
                                 alignment_history=True,
                                 output_attention=False)
    concat_cell = ConcatOutputAndAttentionWrapper(attn_cell)
    decoder_cell = MultiRNNCell([OutputProjectionWrapper(concat_cell, hp.embed_size),
                                 ResidualWrapper(GRUCell(hp.embed_size)),
                                 ResidualWrapper(GRUCell(hp.embed_size))], state_is_tuple=True)

    output_cell = OutputProjectionWrapper(decoder_cell, out_dim)
    initial_state = output_cell.zero_state(batch_size=beam_batch_size, dtype=tf.float32)

    decoder = BeamSearchDecoder(cell=output_cell,
                                embedding=embeddings,
                                start_tokens=start_tokens,
                                end_token=end_token,
                                initial_state=initial_state,
                                beam_width=hp.beam_width)
    outputs, t1, t2 = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                        maximum_iterations=hp.max_len)
    return outputs
