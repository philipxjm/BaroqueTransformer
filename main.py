import pickle
from preprocess import get_notes, get_sequence
from model import build_model, train, generate_notes
from music21 import instrument, note, stream, chord

seq_len = 50


def train_model(dir):
    notes = get_notes(dir)
    vocab_size = len(set(notes))
    train_input, train_output = get_sequence(notes, seq_len, vocab_size)
    train_input = train_input / float(vocab_size)
    model = build_model(seq_len, 1, vocab_size)
    train(model, train_input, train_output, dir)


def generate(notes_path, weights):
    with open('data/notes/' + notes_path, 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    vocab_size = len(set(notes))

    # print(pitchnames)

    train_input, train_output = get_sequence(notes, seq_len, vocab_size)

    # print(train_input.shape)
    model = build_model(seq_len, 1, vocab_size)
    model.fit(train_input, train_output, epochs=0)
    print(model.summary())
    model.load_weights(weights)
    generated_notes = generate_notes(
        model, weights, train_input, pitchnames, vocab_size)
    print(generated_notes)
    create_midi(generated_notes)


def create_midi(logits):
    offset = 0
    output_notes = []
    for pattern in logits:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')


if __name__ == '__main__':
    train_model("bach_test/")
    # generate('bach_test', 'weights/bach_test-54-0.0643.hdf5')
