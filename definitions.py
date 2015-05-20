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
TRANSFORM_DIR = 'data/transform'


# image dimensions
C = 3
W = 224
H = 224
N_TARGETS = 1
N_CLASSES = 5
REGRESSION = True

CUSTOM_SCORE_NAME = 'kappa'

RANDOM_STATE = 9

MAX_PIXEL_VALUE = 255
TEST_ITER = 1
MAX_ITER = 1000
PATIENCE = 40
BATCH_SIZE = 128
INITIAL_LEARNING_RATE = 0.005
DECAY_FACTOR = 0.1
INITIAL_MOMENTUM = 0.9
STD = [70.57616967, 51.79715616,43.08553464]
MEAN = [108.7016983, 75.91925049, 54.36722183]
SUBMISSION = 'data/sub'
BALANCE_WEIGHT = 0.2

# number of class samples in training set
#CLASSES = [25810, 5292, 2443, 873, 708]
OCCURENCES = {0: 25810, 2: 5292, 1: 2443, 3: 873, 4: 708}

# print the config to the terminal
print("############################# CONFIG #################################")
pprint.pprint({k: v for k, v in locals().items() if k.isupper()})
print("######################################################################")
