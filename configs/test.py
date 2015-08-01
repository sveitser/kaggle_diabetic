from layers import *

from lasagne import layers 

from config import Config

cnf = {
    'name': 'test',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size': 128,
    'rotate': False,
    'learning_rate': 0.005,
    'balance': 0.5,
    'balance_ratio': 0.5,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size'], C, cnf['w'], cnf['h'])}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                         'nonlinearity': rectify if REGRESSION else softmax}),
]

config = Config(layers=layers, cnf=cnf)
