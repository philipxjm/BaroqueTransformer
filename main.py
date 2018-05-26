import argparse
from preprocess import get_notes, get_sequence
from model import build_model, train

if __name__ == '__main__':
    dir = 'bach_test/'

    notes = get_notes(dir)
    vocab_size = len(set(notes))
    train_input, train_output = get_sequence(notes, 100, vocab_size)
    model = build_model(100, 1, vocab_size)
    train(model, train_input, train_output, dir)
