import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.utils import np_utils

def get_notes(dir):
    notes = []

    for file in glob.glob("data/midi/" + dir + "*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes/' + dir[:-1], 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def get_sequence(notes, seq_len, vocab_size):
    vocab = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(vocab))
    train_input = []
    train_output = []

    for i in range(0, len(notes) - seq_len, 1):
        seq_in = notes[i:i+seq_len]
        seq_out = notes[i+seq_len]
        train_input.append([note_to_int[char] for char in seq_in])
        train_output.append(note_to_int[seq_out])

    train_input = np.reshape(train_input, [len(train_input), seq_len, 1])
    train_input = train_input / float(vocab_size)
    train_output = np_utils.to_categorical(train_output)

    print(train_input.shape)
    print(train_output.shape)

    return train_input, train_output
