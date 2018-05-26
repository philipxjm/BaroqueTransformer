import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint

def build_model(seq_size, channel_size, vocab_size):
    model = Sequential()
    model.add(LSTM(
        1024,
        input_shape=(seq_size, channel_size),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(1024))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

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
        epochs=200,
        batch_size=64,
        callbacks=[checkpoint]
    )
