from layers import *

from lasagne import layers 

from config import Config

cnf = {
    'name': '448_2x2',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 40,
    'batch_size_test': 8,
    'mean': [ 108.73683167, 75.54026794,  53.80962753],
    'std': [ 70.44262987, 51.35997035, 42.51656026],
    'rotate': True,
    'learning_rate': 0.005,
    'balance': 0.2,
    'patience': 100,
    'regression': True,
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (2, 2),
        'untie_biases': False,
    }
    args.update(kwargs)
    return conv_params(**args)


layers = [
    (InputLayer, {'shape': (None, C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(32, filter_size=(3, 3), stride=(2, 2))),
    #(Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(64, filter_size=(3, 3), stride=(2, 2))),
    #(Conv2DLayer, cp(64)),
    #(Conv2DLayer, cp(64, border_mode='full')),
    #(Conv2DLayer, cp(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(128, border_mode='full')),
    #(Conv2DLayer, cp(128)),
    #(Conv2DLayer, cp(128, border_mode='full')),
    #(Conv2DLayer, cp(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(256)),
    #(Conv2DLayer, cp(256, border_mode='full')),
    #(Conv2DLayer, cp(256)),
    #(Conv2DLayer, cp(256, border_mode='full')),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(512, border_mode='full')),
    #(Conv2DLayer, cp(512)),
    #(Conv2DLayer, cp(512, border_mode='full')),
    #(Conv2DLayer, cp(512)),
    (RMSPoolLayer, pool_params()),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if cnf['regression'] else N_CLASSES,
                  'nonlinearity': rectify if cnf['regression'] else softmax}),
]

config = Config(layers=layers, cnf=cnf)
