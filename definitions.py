import pprint

LABEL_FILE = 'data/trainLabels.csv'
PSEUDO_LABEL_FILE = 'data/testLabels.csv'
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
CACHE_DIR = 'data/cache.dat'

# image dimensions
C = 3
W = 256
H = 256
N_TARGETS = 1
N_CLASSES = 5
REGRESSION = True

CUSTOM_SCORE_NAME = 'kappa'

RANDOM_STATE = 42

MAX_PIXEL_VALUE = 255
MAX_ITER = 500
PATIENCE = 50
BATCH_SIZE = 128
INITIAL_LEARNING_RATE = 0.01
INITIAL_MOMENTUM = 0.9

SUBMISSION = 'data/sub'

# number of class samples in training set
#CLASSES = [25810, 5292, 2443, 873, 708]

# print the config to the terminal
print("############################# CONFIG #################################")
pprint.pprint({k: v for k, v in locals().items() if k.isupper()})
print("######################################################################")
