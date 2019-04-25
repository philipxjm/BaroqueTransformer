import model
import tensorflow as tf
from data import load_pieces, get_training_batch, build_vocab, \
                 tokenize, get_test_batch
import numpy as np
import sys
from tqdm import tqdm
import hyper_params as hp
from midi_handler import noteStateMatrixToMidi

np.set_printoptions(threshold=sys.maxsize)


def train(model, pieces, epochs, save_name, start=0):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    pbar = tqdm(range(start, start+epochs))
    for i in pbar:
        x, y = get_training_batch(pieces)
        l, _ = sess.run([model.loss, model.optimize],
                        feed_dict={model.inputs: x,
                                   model.labels: y,
                                   model.dropout: hp.KEEP_PROB})
        pbar.set_description("epoch {}, loss={}".format(i, l))
        if i % 100 == 0:
            print("epoch {}, loss={}".format(i, l))
        if i % 500 == 0:
            print("Saving at epoch {}, loss={}".format(i, l))
            saver.save(sess,
                       save_name + str(l),
                       global_step=i)
        if i % 1000 == 0:
            total_correct = 0
            total_symbols = 0
            for piece in pieces["test"]:
                x = np.expand_dims(piece[:-1], axis=0)
                y = np.expand_dims(piece[1:], axis=0)
                prediction = sess.run(model.logits,
                                      feed_dict={model.inputs: x,
                                                 model.dropout: 1.0})
                activation = np.argmax(prediction, axis=2)
                # print("act: ", activation)
                # print("lab: ", y)
                total_correct += np.sum(y == activation)
                total_symbols += activation.shape[1]
            print(total_correct / total_symbols)
    final_loss = sess.run([model.loss],
                          feed_dict={model.inputs: x,
                                     model.labels: y})
    saver.save(sess, save_name + str(final_loss[0]))


def test(model, pieces, save_name):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    total_correct = 0
    total_symbols = 0
    for piece in pieces["test"]:
        x = np.expand_dims(piece[:-1], axis=0)
        y = np.expand_dims(piece[1:], axis=0)
        prediction = sess.run(model.logits,
                              feed_dict={model.inputs: x,
                                         model.dropout: 1.0})
        activation = np.argmax(prediction, axis=2)
        print("act: ", activation)
        print("lab: ", y)
        total_correct += np.sum(y == activation)
        total_symbols += activation.shape[1]
    print(total_correct / total_symbols)


def generate(model,
             pieces,
             save_name,
             token2idx,
             idx2token,
             batch_size=10,
             length=1000):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    x, _ = get_test_batch(pieces, 1)

    time_input = np.copy(x)
    for i in range(16, time_input.shape[1]):
        time_input[0][i] = token2idx[hp.PAD]
    # (batch_size, max_len, pitch_sz)
    composition = np.zeros((time_input.shape[0],
                            int(time_input.shape[1] / 4),
                            hp.NOTE_LEN, 2))
    real_comp = np.zeros((time_input.shape[0],
                          int(time_input.shape[1] / 4),
                          hp.NOTE_LEN, 2))
    previous = np.zeros((hp.NOTE_LEN, 2))
    real_previous = np.zeros((hp.NOTE_LEN, 2))
    # pbar = tqdm(range(length))
    # print(time_input)
    # int(time_input.shape[1] / 4)
    for i in range(4, int(time_input.shape[1] / 4)):
        for j in range(4):
            # (batch_size, max_len, vocab_size)
            prediction = sess.run(model.logits,
                                  feed_dict={model.inputs: time_input,
                                             model.dropout: 1.0})
            # (batch_size, max_len)
            activation = np.argmax(prediction, axis=2)
            pitch = idx2token[activation[0][i*4 + j-1]] - 24
            if pitch < hp.NOTE_LEN:
                composition[0][i][pitch][0] = 1
                if previous[pitch][0] == 1:
                    composition[0][i][pitch][1] = 0
                else:
                    composition[0][i][pitch][1] = 1

            real_pitch = idx2token[x[0][i*4 + j]] - 24
            if real_pitch < hp.NOTE_LEN:
                real_comp[0][i][real_pitch][0] = 1
                if real_previous[real_pitch][0] == 1:
                    real_comp[0][i][real_pitch][1] = 0
                else:
                    real_comp[0][i][real_pitch][1] = 1

            time_input[0][i*4 + j] = activation[0][i*4 + j-1]
            print(time_input)
        previous = composition[0][i]
        real_previous = real_comp[0][i]
    print(composition.shape)
    for song_idx in range(composition.shape[0]):
        noteStateMatrixToMidi(composition[song_idx],
                              'output/sample_' + str(song_idx))
    for song_idx in range(real_comp.shape[0]):
        noteStateMatrixToMidi(real_comp[song_idx],
                              'output/real_sample_' + str(song_idx))


if __name__ == '__main__':
    inputs = tf.placeholder(tf.int32, shape=[None, hp.MAX_LEN])
    labels = tf.placeholder(tf.int32, shape=[None, hp.MAX_LEN])
    dropout = tf.placeholder(tf.float32, shape=())

    pieces, seqlens = load_pieces("data/roll/jsb8.pkl")
    token2idx, idx2token = build_vocab(pieces)
    pieces = tokenize(pieces, token2idx, idx2token)
    m = model.Model(inputs=inputs,
                    labels=labels,
                    dropout=dropout,
                    token2idx=token2idx,
                    idx2token=idx2token)
    # train(m, pieces, 500000, "model/jsb8/model_")
    # test(m, pieces, "model/jsb8/model")
    generate(m, pieces, "model/jsb8/model", token2idx, idx2token)
