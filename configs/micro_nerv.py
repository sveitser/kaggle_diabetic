from layers import *

from config import Config

cnf = {
    'name': 'micro',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_medium',
    'batch_size': 128,
    #'mean': [112.26],
    #'std': [26.63],
    'mean': [ 108.73683167, 75.54026794,  53.80962753],
    'std': [ 70.44262987, 51.35997035, 42.51656026],
    'learning_rate': 0.005,
    'regression': True,
    'n_classes': 3,
    'rotate': True,
    'balance': 0.4,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size'], C, cnf['w'], cnf['h'])}),
    (cuda_convnet.ShuffleBC01ToC01BLayer, {}),
    (Conv2DLayer, conv_params(32, stride=(2, 2))),
    (Conv2DLayer, conv_params(32)),
    #Conv2DLayer, conv_params(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(48, stride=(2, 2))),
    (Conv2DLayer, conv_params(48)),
    (Conv2DLayer, conv_params(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (MaxPool2DLayer, pool_params(stride=(1, 1))),
    (cuda_convnet.ShuffleC01BToBC01Layer, {}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                         'nonlinearity': rectify if REGRESSION else softmax}),
]

#layers = [
#    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
#    (Conv2DLayer, conv_params(36, filter_size=(3, 3), stride=(2, 2))),
#    #(Conv2DLayer, conv_params(24)),
#    #Conv2DLayer, conv_params(32)),
#    (MaxPool2DLayer, pool_params()),
#    (Conv2DLayer, conv_params(64, stride=(2, 2))),
#    (MaxPool2DLayer, pool_params()),
#    (Conv2DLayer, conv_params(128)),
#    (MaxPool2DLayer, pool_params()),
#    (Conv2DLayer, conv_params(256)),
#    (MaxPool2DLayer, pool_params()),
#    (DropoutLayer, {'p': 0.5}),
#    (DenseLayer, {'num_units': 1024}),
#    (FeaturePoolLayer, {'pool_size': 2}),
#    (DropoutLayer, {'p': 0.5}),
#    (DenseLayer, {'num_units': 1024}),
#    (FeaturePoolLayer, {'pool_size': 2}),
#    (DenseLayer, {'num_units': N_TARGETS if cnf['regression'] 
#                                      else cnf.get('n_classes', N_CLASSES),
#                         'nonlinearity': rectify if cnf['regression'] 
#                                         else softmax}),
#]

config = Config(layers=layers, cnf=cnf)
