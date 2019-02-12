import pickle
import model
import tensorflow as tf
import os
import random
from midi_handler import midiToNoteStateMatrix, noteStateMatrixToMidi
from data import noteStateMatrixToInputForm
import numpy as np

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
        if len(outMatrix) < SEQ_LEN:
            # Skip if piece is too short or is not 4/4 time.
            continue

        pieces[name] = outMatrix
        print("Loaded {}".format(name))

    return pieces


def getPieceSegment(pieces):
    # randomly pick from pieces a batch_len long segment
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, len(piece_output)-SEQ_LEN, DIVISION_LEN)

    # (batch_len, pitch_sz, 2)
    seg_out = piece_output[start:start+SEQ_LEN]
    # (batch_len, pitch_sz, 80)
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out


def getPieceBatch(pieces):
    # (batch_size, batch_len, pitch_sz, 80)
    # (batch_size, batch_len, pitch_sz, 2)
    i, o = zip(*[getPieceSegment(pieces) for _ in range(BATCH_SIZE)])
    return np.array(i, dtype=float), np.array(o, dtype=float)


def train(model, pieces, epochs, start=0):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(start, start+epochs):
        x, y = getPieceBatch(pieces)
        l, _ = sess.run([model.loss, model.optimize],
                        feed_dict={model.inputs: x,
                                   model.labels: y})
        # if i % 100 == 0:
        print("epoch {}, loss={}".format(i, l))
        if i % 100 == 0:
            p = sess.run(model.prediction,
                         feed_dict={model.inputs: x,
                                    model.labels: y})
            print(np.array(p)[0])
            noteStateMatrixToMidi(np.array(p)[0],
                                  'output/sample{}'.format(i))
        #     pickle.dump(model.learned_config,
        #                 open('output/params{}.p'.format(i), 'wb'))


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN,
                                               NOTE_LEN, 80])
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN,
                                               NOTE_LEN, 2])
    pcs = loadPieces("data/midi/chopin")
    print(getPieceBatch(pcs)[0].shape)
    m = model.Model(inputs, labels, 0.5, [300, 300], [100, 50])
    train(m, pcs, 10000)
