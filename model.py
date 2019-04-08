import tensorflow as tf
import hyper_params as hp
from modules import ff, positional_encoding, \
                    multihead_attention, get_token_embeddings


class Model:
    def __init__(self, inputs, labels, token2idx, idx2token):
        self.inputs = inputs
        self.labels = labels
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.embeddings = get_token_embeddings(hp.VOCAB_SIZE,
                                               hp.HIDDEN_SIZE,
                                               zero_pad=True)
        self.optimize, self.loss = self.train(self.inputs, self.labels)
        self.prediction = self.eval(self.inputs, self.labels)

    def time_encode(self, encoder_inputs, training=True):
        '''
        Returns
        memory: encoder outputs. (BATCH, SEQ_LEN, HIDDEN_SIZE)
        '''
        with tf.variable_scope("time_encoder", reuse=tf.AUTO_REUSE):

            # embedding
            # (batch, time, note * 2)
            enc = tf.nn.embedding_lookup(self.embeddings, encoder_inputs)
            enc *= hp.HIDDEN_SIZE**0.5

            enc += positional_encoding(enc, hp.MAX_LEN)
            enc = tf.layers.dropout(enc,
                                    hp.DROPOUT,
                                    training=training)

            # Blocks
            for i in range(hp.NUM_BLOCKS):
                with tf.variable_scope("num_blocks_{}".format(i),
                                       reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=hp.NUM_HEADS,
                                              dropout_rate=hp.DROPOUT,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[hp.FF_SIZE, hp.HIDDEN_SIZE])

        # output = tf.reshape(enc, (-1, hp.MAX_LEN * hp.HIDDEN_SIZE))
        output = tf.layers.dense(enc, hp.HIDDEN_SIZE, activation=tf.nn.relu)
        logits = tf.layers.dense(output, hp.VOCAB_SIZE)
        return logits

    def loss_function(self, logits, labels):
        nonpadding = tf.to_float(tf.not_equal(labels, self.token2idx[hp.PAD]))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                        labels=labels)
        loss = tf.reduce_mean(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        return loss

    def train(self, inputs, labels):
        logits = self.time_encode(inputs, training=True)
        loss = self.loss_function(logits, labels)
        return tf.train.AdamOptimizer(1e-3).minimize(loss), loss

    def eval(self, inputs, labels):
        logits = self.time_encode(inputs, training=False)
        return logits

    # def decode(self, decoder_inputs, memory, training=True):
    #     '''
    #     memory: encoder outputs. (BATCH, SEQ_LEN, HIDDEN_SIZE)
    #     Returns
    #     logits: (N, T2, V). float32.
    #     y_hat: (N, T2). int32
    #     y: (N, T2). int32
    #     sents2: (N,). string.
    #     '''
    #     with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    #         # embedding
    #         dec = decoder_inputs
    #         dec = tf.einsum('ntd,dk->ntk', dec, self.embeddings)
    #         dec *= hp.HIDDEN_SIZE ** 0.5  # scale
    #
    #         dec += positional_encoding(dec, self.hp.maxlen2)
    #         dec = tf.layers.dropout(dec, hp.DROPOUT, training=training)
    #
    #         # Blocks
    #         for i in range(self.hp.num_blocks):
    #             with tf.variable_scope("num_blocks_{}".format(i),
    #                                    reuse=tf.AUTO_REUSE):
    #                 # Masked self-attention (Causality is True at this time)
    #                 dec = multihead_attention(queries=dec,
    #                                           keys=dec,
    #                                           values=dec,
    #                                           num_heads=hp.NUM_HEADS,
    #                                           dropout_rate=hp.DROPOUT,
    #                                           training=training,
    #                                           causality=True,
    #                                           scope="self_attention")
    #
    #                 # Vanilla attention
    #                 dec = multihead_attention(queries=dec,
    #                                           keys=memory,
    #                                           values=memory,
    #                                           num_heads=hp.NUM_HEADS,
    #                                           dropout_rate=hp.DROPOUT,
    #                                           training=training,
    #                                           causality=False,
    #                                           scope="vanilla_attention")
    #                 # Feed Forward
    #                 dec = ff(dec, num_units=[hp.FF_SIZE, hp.HIDDEN_SIZE])
    #
    #     # Final linear projection (embedding weights are shared)
    #     logits = tf.layers.dense(dec, hp.SEQ_LEN * 2)
    #
    #     return logits
