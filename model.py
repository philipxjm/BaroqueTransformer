import tensorflow as tf
from tensorflow.contrib.framework import nest

BATCH_SIZE = 10
SEQ_LEN = 128
NOTE_LEN = 78
EPS = 1e-10


class Model:
    def __init__(self,
                 inputs,
                 labels,
                 keep_prob,
                 time_sizes=[300, 300],
                 note_sizes=[100, 50]):
        self.inputs = inputs  # input shape (batch, time, note, feature)
        self.labels = labels  # label shape (batch, time, note, out)
        self.keep_prob = keep_prob
        self.time_sizes = time_sizes
        self.note_sizes = note_sizes

        self.time_lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(sz, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
             for sz in self.time_sizes],
            state_is_tuple=True
        )
        self.time_state = nest.map_structure(
            lambda x: tf.placeholder_with_default(x, x.shape, x.op.name),
            self.time_lstm_cell.zero_state(BATCH_SIZE * NOTE_LEN, tf.float32))
        for tensor in nest.flatten(self.time_state):
            tf.add_to_collection('time_state_input', tensor)

        self.note_lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(sz, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
             for sz in self.note_sizes],
            state_is_tuple=True
        )
        self.note_state = nest.map_structure(
            lambda x: tf.placeholder_with_default(x, x.shape, x.op.name),
            self.note_lstm_cell.zero_state(BATCH_SIZE * SEQ_LEN, tf.float32))
        for tensor in nest.flatten(self.note_state):
            tf.add_to_collection('note_state_input', tensor)

        self.final_time_state, self.final_note_state, self.prediction \
            = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()

    def forward_pass(self):
        x = tf.transpose(self.inputs, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [BATCH_SIZE * NOTE_LEN, SEQ_LEN, -1])

        # time model
        with tf.variable_scope('time_model'):
            time_out, time_state \
                = tf.nn.dynamic_rnn(cell=self.time_lstm_cell,
                                    inputs=x,
                                    initial_state=self.time_state,
                                    dtype=tf.float32)
            for tensor in nest.flatten(time_state):
                tf.add_to_collection('time_state_output', tensor)

        # reshape from note invariant to time invariant
        hidden = tf.reshape(time_out, [BATCH_SIZE, NOTE_LEN, SEQ_LEN, -1])
        hidden = tf.transpose(hidden, perm=[0, 2, 1, 3])
        hidden = tf.reshape(hidden, [BATCH_SIZE * SEQ_LEN, NOTE_LEN, -1])
        # start_label = tf.zeros([BATCH_SIZE * SEQ_LEN, 1, 2])
        # correct_choices, _ = tf.split(self.labels, [NOTE_LEN - 1, 1], 2)
        # correct_choices = tf.reshape(correct_choices,
        #                              [BATCH_SIZE * SEQ_LEN, NOTE_LEN - 1, -1])
        # correct_choices = tf.concat([start_label, correct_choices], 1)
        # hidden = tf.concat([hidden, correct_choices], 2)

        # note model
        with tf.variable_scope('note_model'):
            note_out, note_state \
                = tf.nn.dynamic_rnn(cell=self.note_lstm_cell,
                                    inputs=hidden,
                                    initial_state=self.note_state,
                                    dtype=tf.float32)
            for tensor in nest.flatten(note_state):
                tf.add_to_collection('note_state_output', tensor)

        # dense layer
        W = tf.Variable(tf.random_normal([self.note_sizes[-1], 2],
                                         stddev=0.01,
                                         dtype=tf.float32))
        b = tf.Variable(tf.random_normal([2], stddev=0.01, dtype=tf.float32))
        note_out = tf.tensordot(note_out, W, axes=[[2], [0]]) + b
        note_out = tf.reshape(note_out, [BATCH_SIZE, SEQ_LEN, NOTE_LEN, 2])

        return time_state, note_state, tf.nn.sigmoid(note_out)

    def optimizer(self):
        return tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def loss_function(self):
        active_notes = tf.expand_dims(self.labels[:, :, :, 0], 3)
        mask = tf.concat([tf.ones_like(active_notes), active_notes], 3)
        loglikelihoods = mask * tf.log(tf.clip_by_value(2 * self.prediction
                                                        * self.labels
                                                        - self.prediction
                                                        - self.labels + 1,
                                                        EPS, 1.0))
        return -tf.reduce_mean(loglikelihoods)
