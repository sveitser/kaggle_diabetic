from layers import *

from model import Model

cnf = {
    'name': __name__.split('.')[-1],
    'w': 236,
    'h': 236,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 32,
    'batch_size_test': 8,
    'mean': [108.64628601, 75.86886597, 54.34005737],
    'std': [70.53946096, 51.71475228, 43.03428563],
    #'learning_rate': 0.001,
    'patience': 40,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 1.0,
    #'balance_weights':  np.array([1, 2, 2, 3, 3.5], dtype=float),
    'balance_weights': np.array(CLASS_WEIGHTS),
    'balance_ratio': 0.95,
    'final_balance_weights':  np.array([1, 2, 2, 3, 3.5], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.2, 1.2),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-20, 20),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0005,
    'color': True,
    'sigma': 0.1,
    #'update': 'adam',
    'schedule': {
        0: 0.002,
        120: 0.0002,
        170: 0.00002,
        201: 'stop',
    },
    'slope': theano.shared(np.cast['float32'](0.5)),
    'slope_decay': 0.975,
}

#nonlinearity = SharedRelu(cnf['slope'])
nonlinearity = LeakyRectify(cnf['slope'])

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3),
        'pad': 0,
        'border_mode': None,
        'nonlinearity': nonlinearity,
    }
    args.update(kwargs)
    return conv_params(**args)

def dp(num_units, *args, **kwargs):
    args = {
        'num_units': num_units,
        'nonlinearity': nonlinearity,
    }
    args.update(kwargs)
    return dense_params(**args)

n = 16

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(n)),
    (Conv2DLayer, cp(n)),
    #Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(2 * n)),
    (Conv2DLayer, cp(2 * n)),
    (Conv2DLayer, cp(2 * n)),
    #(Conv2DLayer, cp(2 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(4 * n)),
    (Conv2DLayer, cp(4 * n)),
    (Conv2DLayer, cp(4 * n)),
    #(MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(8 * n)),
    (Conv2DLayer, cp(8 * n)),
    (Conv2DLayer, cp(8 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(16 * n)),
    (Conv2DLayer, cp(16 * n)),
    (Conv2DLayer, cp(16 * n)),
    (RMSPoolLayer, pool_params(stride=(3, 3), pool_size=(4, 4), pad=0)),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dp(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dp(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                  'nonlinearity': rectify if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
