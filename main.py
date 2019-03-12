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

TIME_SIZE_0 = 300
TIME_SIZE_1 = 300
NOTE_SIZE_0 = 100
NOTE_SIZE_1 = 100


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


def generate(model, pieces, save_name, batch_size=10, length=500):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    x, y = getPieceBatch(pieces)

    ts_0_c = np.zeros((batch_size*NOTE_LEN, TIME_SIZE_0), dtype=np.float32)
    ts_0_h = np.zeros((batch_size*NOTE_LEN, TIME_SIZE_0), dtype=np.float32)
    ts_1_c = np.zeros((batch_size*NOTE_LEN, TIME_SIZE_1), dtype=np.float32)
    ts_1_h = np.zeros((batch_size*NOTE_LEN, TIME_SIZE_1), dtype=np.float32)
    ns_0_c = np.zeros((batch_size*SEQ_LEN, NOTE_SIZE_0), dtype=np.float32)
    ns_0_h = np.zeros((batch_size*SEQ_LEN, NOTE_SIZE_0), dtype=np.float32)
    ns_1_c = np.zeros((batch_size*SEQ_LEN, NOTE_SIZE_1), dtype=np.float32)
    ns_1_h = np.zeros((batch_size*SEQ_LEN, NOTE_SIZE_1), dtype=np.float32)

    composition = y[:, -1:, :, :]  # (batch_size, 1, pitch_sz, 2)

    pbar = tqdm(range(length))
    for i in pbar:
        # print("generating note" + str(i))
        hi, ts = sess.run([model.time_out, model.final_time_state],
                          feed_dict={model.inputs: x,
                                     model.time_state[0].c: ts_0_c,
                                     model.time_state[0].h: ts_0_h,
                                     model.time_state[1].c: ts_1_c,
                                     model.time_state[1].h: ts_1_h,
                                     model.keep_prob: 1.0})

        ts_0_c = ts[0].c
        ts_0_h = ts[0].h
        ts_1_c = ts[1].c
        ts_1_h = ts[1].h

        print(hi.shape)

        step_composition = np.zeros((batch_size, 1, 2))
        for j in range(NOTE_LEN):
            note_input = np.append(hi[:, :j+1, :], step_composition, axis=2)
            no, ns = sess.run([model.prediction, model.final_note_state],
                              feed_dict={model.note_input: note_input,
                                         model.note_state[0].c: ns_0_c,
                                         model.note_state[0].h: ns_0_h,
                                         model.note_state[1].c: ns_1_c,
                                         model.note_state[1].h: ns_1_h,
                                         model.keep_prob: 1.0})
            ns_0_c = ns[0].c
            ns_0_h = ns[0].h
            ns_1_c = ns[1].c
            ns_1_h = ns[1].h
            step_composition = np.append(step_composition,
                                         no[0, :, -1:, :],
                                         axis=1)

        r = np.random.random(step_composition.shape)
        step_composition = np.greater(step_composition, r).astype(int)
        step_composition[:, :, 1] = np.multiply(step_composition[:, :, 0],
                                                step_composition[:, :, 1])

        composition = np.append(composition,
                                np.expand_dims(step_composition, axis=1),
                                axis=1)

        x = np.array([noteStateMatrixToInputForm(p_) for p_ in np.expand_dims(step_composition, axis=1)])

    for song_idx in range(composition.shape[0]):
        noteStateMatrixToMidi(composition[song_idx],
                              'output/sample_' + str(song_idx))


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN,
                                               NOTE_LEN, 80])
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN,
                                               NOTE_LEN, 2])
    keep_prob = tf.placeholder(tf.float32)

    pcs = loadPieces("data/midi/nocturne")
    # print(getPieceBatch(pcs)[0].shape)
    m = model.Model(inputs=inputs,
                    labels=labels,
                    keep_prob=keep_prob,
                    time_sizes=[TIME_SIZE_0, TIME_SIZE_1],
                    note_sizes=[NOTE_SIZE_0, NOTE_SIZE_1])
    # train(m, pcs, 25000, "model/new/model_")
    generate(m, pcs, "model/new/model_0.3516726-0")
