NOTE_LEN = 78  # length of note sequence
DIVISION_LEN = 16  # 1 bar of music

EPS = 1e-10  # episilon for log math
KEEP_PROB = 0.5  # dropout rate

# big
BATCH_SIZE = 10  # size of batches
HIDDEN_SIZE = 512  # size of encoder decoder hidden dimension
FF_SIZE = 2048  # size of feed forward dimension
NUM_HEADS = 8  # number of attention head
NUM_BLOCKS = 8  # number of encoder decoder blocks
VOCAB_SIZE = 49  # jsb4-55, jsb8-49
MAX_LEN = 512

# small
# BATCH_SIZE = 10  # size of batches
# HIDDEN_SIZE = 256  # size of encoder decoder hidden dimension
# FF_SIZE = 1024  # size of feed forward dimension
# NUM_HEADS = 4  # number of attention head
# NUM_BLOCKS = 4  # number of encoder decoder blocks
# VOCAB_SIZE = 55  # jsb4-55, jsb8-49
# MAX_LEN = 128

REST = 420
STOP = 690
PAD = 999
SEPERATION = 16
