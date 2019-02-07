import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Lambda
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras import backend as k


def build_model(seq_size, channel_size, vocab_size):
    model = Sequential()
    # model.add(Lambda(lambda x: k.squeeze(x, 2)))
    # model.add(Embedding(vocab_size, 100, input_length=seq_size))
    model.add(LSTM(
        512,
        input_shape=(seq_size, channel_size),
        return_sequences=True
    ))
    model.add(Dropout(0.5))
    model.add(LSTM(
        512
    ))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train(model, train_input, train_output, dir):
    path = "weights/" + dir[:-1] + "-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        path,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    model.fit(
        train_input,
        train_output,
        epochs=500,
        batch_size=64,
        callbacks=[checkpoint]
    )


def generate_notes(model, weights, network_input, pitchnames, vocab_size):
    model.load_weights(weights)
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict(
        (number, note) for number, note in enumerate(pitchnames))

    # print(int_to_note)

    pattern = list(network_input[start])
    prediction_output = []

    for note_index in range(500):
        # print(pattern)
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab_size)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
