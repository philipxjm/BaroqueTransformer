import model
import tensorflow as tf
from data import load_pieces, get_batch, build_vocab, tokenize
import numpy as np
from tqdm import tqdm
import hyper_params as hp


def train(model, pieces, epochs, save_name, start=0):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    pbar = tqdm(range(start, start+epochs))
    for i in pbar:
        x, y = get_batch(pieces)
        l, _ = sess.run([model.loss, model.optimize],
                        feed_dict={model.inputs: x,
                                   model.labels: y})
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
                                     model.labels: y})
    saver.save(sess, save_name + str(final_loss[0]))


def generate(model, pieces, save_name, batch_size=10, length=1000):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    x, y = get_batch(pieces)

    time_input = x
    composition = x  # (batch_size, 1, pitch_sz, 2)

    pbar = tqdm(range(length))
    for i in pbar:
        # print("generating note" + str(i))
        prediction = sess.run(model.prediction,
                              feed_dict={model.inputs: time_input})

        r = np.random.random(prediction.shape)
        step_composition = np.greater(prediction, r).astype(int)
        step_composition[:, :, 1] = np.multiply(step_composition[:, :, 0],
                                                step_composition[:, :, 1])

        composition = np.append(composition,
                                np.expand_dims(step_composition, axis=1),
                                axis=1)

        time_input = composition[:, -hp.SEQ_LEN:, :, :]

    composition = composition[:, hp.SEQ_LEN:, :, :]

    # for song_idx in range(composition.shape[0]):
    #     noteStateMatrixToMidi(composition[song_idx],
    #                           'output/sample_' + str(song_idx))


if __name__ == '__main__':
    inputs = tf.placeholder(tf.int32, shape=[None, hp.MAX_LEN])
    labels = tf.placeholder(tf.int32, shape=[None])

    pieces, seqlens = load_pieces("data/roll/jsb16.pkl")
    token2idx, idx2token = build_vocab(pieces)
    pieces = tokenize(pieces, token2idx, idx2token)
    m = model.Model(inputs=inputs,
                    labels=labels)
    train(m, pieces, 500000, "model/transformer/model_")
    # generate(m, pieces, "model/transformer/model_0.07783421")
