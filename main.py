import model
import tensorflow as tf
import os
import random
from midi_handler import midiToNoteStateMatrix, noteStateMatrixToMidi
from data import noteStateMatrixToInputForm
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 10
SEQ_LEN = 128
NOTE_LEN = 78
DIVISION_LEN = 16


def loadPieces(dirpath):
    # loads midi pieces and converts them each to midi state matrices.
    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid', '.MID'):
            continue

        name = fname[:-4]

        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        if len(outMatrix) < SEQ_LEN+1:
            # Skip if piece is too short or is not 4/4 time.
            continue

        pieces[name] = outMatrix
        print("Loaded {}".format(name))

    return pieces


def getPieceSegment(pieces):
    # randomly pick from pieces a batch_len long segment
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, len(piece_output)-SEQ_LEN+1, DIVISION_LEN)

    # (batch_len, pitch_sz, 2)
    seg_out = piece_output[start:start+SEQ_LEN+1]
    # (batch_len, pitch_sz, 80)
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in[:-1], seg_out[1:]


def getPieceBatch(pieces):
    # (batch_size, batch_len, pitch_sz, 80)
    # (batch_size, batch_len, pitch_sz, 2)
    i, o = zip(*[getPieceSegment(pieces) for _ in range(BATCH_SIZE)])
    return np.array(i, dtype=float), np.array(o, dtype=float)


def train(model, pieces, epochs, save_name, start=0):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    pbar = tqdm(range(start, start+epochs))
    for i in pbar:
        x, y = getPieceBatch(pieces)
        l, _ = sess.run([model.loss, model.optimize],
                        feed_dict={model.inputs: x,
                                   model.labels: y,
                                   model.keep_prob: 0.5})
        # if i % 100 == 0:
        pbar.set_description("epoch {}, loss={}".format(i, l))
        if i % 100 == 0:
            print("epoch {}, loss={}".format(i, l))
            # p = sess.run(model.prediction,
            #              feed_dict={model.inputs: x,
            #                         model.labels: y,
            #                         model.keep_prob: 1.0})
        if i % 500 == 0:
            print("Saving at epoch {}, loss={}".format(i, l))
            saver.save(sess,
                       save_name + str(l),
                       global_step=i)
    final_loss = sess.run([model.loss],
                          feed_dict={model.inputs: x,
                                     model.labels: y,
                                     model.keep_prob: 1.0})
    saver.save(sess, save_name + str(final_loss[0]))


def generate(model, pieces, save_name):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    x, y = getPieceBatch(pieces)
    ts, ns = sess.run([model.final_time_state, model.final_note_state],
                      feed_dict={model.inputs: x,
                                 model.labels: y,
                                 model.keep_prob: 1.0})
    print(ts[0].shape)
    print(ts[1].shape)
    print(ns[0].shape)
    print(ns[1].shape)
    # p = sess.run(model.prediction,
    #              feed_dict={model.inputs: x,
    #                         model.labels: y,
    #                         model.keep_prob: 1.0})
    # p = p[0]
    # r = np.random.random(p.shape)
    # p = np.greater(p, r).astype(int)
    # p[:, :, 1] = np.multiply(p[:, :, 0], p[:, :, 1])
    # noteStateMatrixToMidi(p, 'output/sample')
    # print(p)


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN,
                                               NOTE_LEN, 80])
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN,
                                               NOTE_LEN, 2])
    time_state = tf.placeholder(tf.float32, shape=[2, BATCH_SIZE*NOTE_LEN, 300])
    keep_prob = tf.placeholder(tf.float32)
    pcs = loadPieces("data/midi/nocturne")
    # print(getPieceBatch(pcs)[0].shape)
    m = model.Model(inputs, labels, keep_prob, [300, 300], [100, 50])
    # train(m, pcs, 20000, "model/327/model_")
    generate(m, pcs, "model/327/model_0.0023756323")
