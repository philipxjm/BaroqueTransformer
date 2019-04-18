import tensorflow as tf
import hyper_params as hp
from modules import ff, positional_encoding, \
                    multihead_attention, get_token_embeddings


class Model:
    def __init__(self, inputs, labels, dropout, token2idx, idx2token):
        self.inputs = inputs
        self.labels = labels
        self.dropout = dropout
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.embeddings = get_token_embeddings(hp.VOCAB_SIZE,
                                               hp.HIDDEN_SIZE,
                                               zero_pad=True)

        self.logits = self.time_encode(inputs)
        self.optimize, self.loss = self.train(self.inputs, self.labels)

    def time_encode(self, encoder_inputs):
        '''
        Returns
        memory: encoder outputs. (BATCH, SEQ_LEN, HIDDEN_SIZE)
        '''
        with tf.variable_scope("time_encoder", reuse=tf.AUTO_REUSE):

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, encoder_inputs)
            enc *= hp.HIDDEN_SIZE**0.5

            enc += positional_encoding(enc, hp.MAX_LEN)
            enc = tf.nn.dropout(enc, self.dropout)

            # Blocks
            for i in range(hp.NUM_BLOCKS):
                with tf.variable_scope("num_blocks_{}".format(i),
                                       reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=hp.NUM_HEADS,
                                              dropout=self.dropout,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[hp.FF_SIZE, hp.HIDDEN_SIZE])

        output = tf.reshape(enc, (-1, hp.MAX_LEN, hp.HIDDEN_SIZE))
        logits = tf.layers.dense(output, hp.VOCAB_SIZE)
        return logits

    def loss_function(self, logits, labels):
        nonpadding = tf.to_float(tf.not_equal(labels, self.token2idx[hp.PAD]))
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
        loss = tf.reduce_sum(ce*nonpadding) / (tf.reduce_sum(nonpadding)+1e-7)
        return loss

    def train(self, inputs, labels):
        loss = self.loss_function(self.logits, labels)
        return tf.train.AdamOptimizer(1e-4).minimize(loss), loss
