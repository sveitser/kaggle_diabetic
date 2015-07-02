from layers import *

from model import Model

cnf = {
    'name': __name__.split('.')[-1],
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size_train': 128,
    'batch_size_test': 32,
    #'mean': [112.26],
    #'std': [26.63],
    'mean': [ 108.73683167, 75.54026794,  53.80962753],
    'std': [ 70.44262987, 51.35997035, 42.51656026],
    'learning_rate': 0.003,
    'patience': 40,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 1.0,
    'balance_weights':  np.array([1, 20, 4, 40, 40], dtype=float),
    'balance_ratio': 0.95,
    'final_balance_weights':  np.array([1, 3, 2, 4, 5], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.4, 1.4),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-20, 20),
        'do_flip': True,
        'allow_stretch': True,
    },
    'color': True,
    'sigma': 0.1,
    'schedule': {
        0: 0.003,
        150: 0.0003,
        200: 'stop',
    },
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3),
        'nonlinearity': very_leaky_rectify,
    }
    args.update(kwargs)
    return conv_params(**args)


layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(24, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, cp(24)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, cp(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192)),
    (RMSPoolLayer, pool_params(pool_size=(3, 3), stride=(2, 2))), # pad to get even x/y
    #(MaxPool2DLayer, pool_params()),
    #(Conv2DLayer, cp(512)),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                  'nonlinearity': None if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
