import pprint

LABEL_FILE = 'data/trainLabels.csv'
MEAN_FILE = 'data/mean.npy'

TRAIN_DIR = 'data/train_res'
TEST_DIR = 'data/test_res'

# cache files for neural net feature extraction
TRAIN_FEATURES = 'data/X_train.npy'
TEST_FEATURES = 'data/X_test.npy'
TRAIN_LABELS = 'data/y_train.npy'
TEST_LABELS = 'data/y_test.npy'
ESTIMATOR_FILENAME = 'estimator.pickle'
WEIGHTS = 'weights.pickle'

# image dimensions
C = 3
W = 256
H = W

RANDOM_STATE = 999

MAX_PIXEL_VALUE = 255
MAX_ITER = 500
PATIENCE = 50
BATCH_SIZE = 128
INITIAL_LEARNING_RATE = 0.01
INITIAL_MOMENTUM = 0.9

SUBMISSION = 'data/sub'

# print the config to the terminal
print("############### CONFIG ##################")
pprint.pprint({k: v for k, v in locals().items() if k.isupper()})
print("#########################################")
