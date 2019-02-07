import os
import random
from midi_handler import midiToNoteStateMatrix, noteStateMatrixToMidi
from data import noteStateMatrixToInputForm
import pickle as pickle
import numpy as np

import signal

batch_width = 10  # number of sequences in a batch
batch_len = 16*8  # length of each sequence
division_len = 16  # interval between possible start locations


def loadPieces(dirpath):
    # loads midi pieces and converts them each to midi state matrices.
    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid', '.MID'):
            continue

        name = fname[:-4]

        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        if len(outMatrix) < batch_len:
            # Skip if piece is too short or is not 4/4 time.
            continue

        pieces[name] = outMatrix
        print("Loaded {}".format(name))

    return pieces


def getPieceSegment(pieces):
    # randomly pick from pieces a batch_len long segment
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, len(piece_output)-batch_len, division_len)

    # (batch_len, pitch_sz, 2)
    seg_out = piece_output[start:start+batch_len]
    # (batch_len, pitch_sz, 80)
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out


def getPieceBatch(pieces):
    # (batch_size, batch_len, pitch_sz, 80)
    # (batch_size, batch_len, pitch_sz, 2)
    i, o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return np.array(i), np.array(o)


def trainPiece(model, pieces, epochs, start=0):
    stopflag = [False]

    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start, start+epochs):
        if stopflag[0]:
            break
        error = model.update_fun(*getPieceBatch(pieces))
        if i % 100 == 0:
            print("epoch {}, error={}".format(i, error))
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xIpt, xOpt = map(np.array, getPieceSegment(pieces))
            noteStateMatrixToMidi(np.concatenate(
                                     (np.expand_dims(xOpt[0], 0),
                                      model.predict_fun(batch_len,
                                                        1,
                                                        xIpt[0])),
                                      axis=0),
                                  'output/sample{}'.format(i))
            pickle.dump(model.learned_config,
                        open('output/params{}.p'.format(i), 'wb'))
    signal.signal(signal.SIGINT, old_handler)


print(getPieceBatch(loadPieces("data/midi/chopin"))[1].shape)
