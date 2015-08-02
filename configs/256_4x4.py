from layers import *

from config import Config

cnf = {
    'name': '256_4x4',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size_train': 128,
    'batch_size_test': 16,
    'balance_weights':  np.array([1, 25, 8, 50, 60], dtype=float),
    'balance_ratio': 0.9,
    'final_balance_weights':  np.array([1, 3, 2.0, 4, 5], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.1, 1.1),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-20, 20),
        'do_flip': True,
        'allow_stretch': True,
    }
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (4, 4),
        'untie_biases': False,
    }
    args.update(kwargs)
    return conv_params(**args)


layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(24, stride=(2, 2))),
    (Conv2DLayer, cp(24, border_mode=None, pad=2)),
    #Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48, stride=(2, 2))),
    (Conv2DLayer, cp(48, border_mode=None, pad=2)),
    (Conv2DLayer, cp(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(96, border_mode=None, pad=2)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96, border_mode=None, pad=2)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192, border_mode=None, pad=2)),
    (Conv2DLayer, cp(192)),
    (RMSPoolLayer, pool_params(pool_size=(3, 3), stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1024}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1024}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
