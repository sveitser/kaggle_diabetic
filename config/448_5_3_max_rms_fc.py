from layers import *

from lasagne import layers 

from model import Model

cnf = {
    'name': '448_5_3_max_rms_fc',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 48,
    'batch_size_test': 8,
    'rotate': True,
    'learning_rate': 0.005,
    'balance': 0.2,
    'patience': 100,
    'regression': True,
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3)
    }
    args.update(kwargs)
    return conv_params(**args)


layers = [
    (InputLayer, {'shape': (None, C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(16, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, conv_params(16)),
    (Conv2DLayer, conv_params(16)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48, stride=(2, 2))),
    (Conv2DLayer, cp(48)),
    (MaxPool2DLayer, pool_params()),
    (LocalResponseNormalization2DLayer, {}),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(512)),
    (Conv2DLayer, cp(512)),
    #(Conv2DLayer, cp(384)),
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

model = Model(layers=layers, cnf=cnf)
