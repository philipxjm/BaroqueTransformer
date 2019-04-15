import numpy as np
import pickle
import hyper_params as hp


def load_pieces(dirpath):
    # loads piano roll
    file = open(dirpath, "rb")
    pieces = pickle.load(file, encoding="latin1")
    pieces = clean_pieces(pieces)
    pieces, seqlens = pad_pieces_to_max(pieces)
    return pieces, seqlens


def clean_pieces(pieces):
    def pad(chord):
        # pad to 4 voices
        padded = np.array(list(chord))
        while len(padded) < 4:
            padded = np.append([hp.REST], padded)
        return padded

    def clean_piece(piece):
        # pad and serialize
        return np.array([pad(chord) for chord in piece]).flatten()
    pieces["train"] = np.array([clean_piece(p) for p in pieces["train"]])
    pieces["test"] = np.array([clean_piece(p) for p in pieces["test"]])
    pieces["valid"] = np.array([clean_piece(p) for p in pieces["valid"]])
    return pieces


def pad_pieces_to_max(pieces):
    def pad_piece_to_max(piece):
        while len(piece) < hp.MAX_LEN + 1:
            piece = np.append(piece, [hp.PAD])
        return piece

    def seperate_long_piece(pieces_list):
        new_pieces = []
        for i in range(len(pieces_list)):
            piece = np.append(pieces_list[i], [hp.STOP])
            if len(pieces_list[i]) > hp.MAX_LEN + 1:
                new_pieces = new_pieces \
                             + [piece[j:j+hp.MAX_LEN+1] for j in
                                range(0, len(piece), hp.SEPERATION)]
            else:
                new_pieces.append(pad_piece_to_max(piece))
        new_pieces = list(
            filter(lambda x: len(x) == hp.MAX_LEN + 1, new_pieces)
        )
        return np.array(new_pieces)

    pieces["train"] = seperate_long_piece(pieces["train"])
    pieces["test"] = seperate_long_piece(pieces["test"])
    pieces["valid"] = seperate_long_piece(pieces["valid"])

    seqlens = {
        "train": np.zeros(len(pieces["train"])),
        "test": np.zeros(len(pieces["test"])),
        "valid": np.zeros(len(pieces["valid"]))
    }

    for i in range(len(pieces["train"])):
        seqlens["train"][i] = len(pieces["train"][i])
    for i in range(len(pieces["test"])):
        seqlens["test"][i] = len(pieces["test"][i])
    for i in range(len(pieces["valid"])):
        seqlens["valid"][i] = len(pieces["valid"][i])
    return pieces, seqlens


def build_vocab(pieces):
    total_notes = np.hstack((pieces["train"].astype(int).flatten(),
                             pieces["test"].astype(int).flatten(),
                             pieces["valid"].astype(int).flatten()))
    vocabs = set(total_notes)
    if hp.PAD in vocabs:
        vocabs.remove(hp.PAD)
    idx2token = {i+1: w for i, w in enumerate(vocabs)}
    token2idx = {w: i+1 for i, w in enumerate(vocabs)}
    idx2token[0] = hp.PAD
    token2idx[hp.PAD] = 0

    # print(idx2token[0])
    # print(token2idx[hp.PAD])

    return token2idx, idx2token


def tokenize(pieces, token2idx, idx2token):
    pieces["train"] = np.array(
        [[token2idx[w] for w in p] for p in pieces["train"]]
    )
    pieces["test"] = np.array(
        [[token2idx[w] for w in p] for p in pieces["test"]]
    )
    pieces["valid"] = np.array(
        [[token2idx[w] for w in p] for p in pieces["valid"]]
    )
    return pieces


def get_batch(pieces, batch_size=hp.BATCH_SIZE):
    batch_indices = np.random.choice(len(pieces["train"]),
                                     size=batch_size,
                                     replace=True)
    # print(batch_indices)
    # print(pieces["train"][batch_indices][:, :-1].shape)
    x = pieces["train"][batch_indices][:, 1:]
    y = pieces["train"][batch_indices][:, 1:]
    # seqlens = seqlens["train"][batch_indices]
    # print(x[:, -5:])
    # print(y)
    return x.astype(int), y.astype(int)


# pieces, seqlens = load_pieces("data/roll/jsb8.pkl")
# print(pieces["test"].shape)
# get_batch(pieces)
token2idx, idx2token = build_vocab(load_pieces("data/roll/jsb8.pkl")[0])
print(idx2token)
# print(len(token2idx))
# print(tokenize(load_pieces("data/roll/jsb16.pkl")[0], token2idx, idx2token)["train"])
