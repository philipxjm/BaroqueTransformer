import pickle
from preprocess import get_notes, get_sequence
from model import build_model, train, generate_notes


def train_model(dir):
    notes = get_notes(dir)
    vocab_size = len(set(notes))
    train_input, train_output = get_sequence(notes, 100, vocab_size)
    train_input = train_input / float(vocab_size)
    model = build_model(100, 1, vocab_size)
    train(model, train_input, train_output, dir)


def generate(notes_path, weights):
    with open('data/notes/' + notes_path, 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    vocab_size = len(set(notes))

    train_input, _ = get_sequence(notes, 100, vocab_size)
    model = build_model(100, 1, vocab_size)
    generated_notes = generate_notes(
        model, weights, train_input, pitchnames, vocab_size)
    print(generated_notes)
    create_midi(generated_notes)


def create_midi(notes):
    pass


if __name__ == '__main__':
    # train_model("chopin/")
    generate('chopin', 'weights/chopin-07-4.7131.hdf5')
