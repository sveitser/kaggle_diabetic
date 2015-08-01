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
    'balance_ratio': 0.5,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size'], 3, cnf['w'], cnf['h'])}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
