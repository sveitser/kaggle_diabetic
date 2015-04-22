import os

import pandas as pd
import numpy as np

from lasagne import layers
from lasagne import init
import lasagne.layers.cuda_convnet
import lasagne.layers.dnn
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify
from lasagne import updates
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
    
from sklearn.utils import shuffle

from PIL import Image

LABEL_FILE = 'data/trainLabels.csv'
TRAIN_DIR = 'data/res'
C = 3
W = 192
H = W
MAX_PIXEL_VALUE = 255
N_CLASSES = 4

Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
#Conv2DLayer = layers.dnn.Conv2DDNNLayer
#MaxPool2DLayer = layers.dnn.MaxPool2DDNNLayer

FeaturePoolLayer = layers.pool.FeaturePoolLayer
DropoutLayer = layers.DropoutLayer

def float32(k):
    return np.cast['float32'](k)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        files, labels = super(FlipBatchIterator, self).transform(Xb, yb)

        Xb = load_images(files) 

        # convert batch to float and bring values in range of [0, 1]
        Xb /= MAX_PIXEL_VALUE
        
        # remove the mean per image and channel
        #Xb -= Xb.mean(axis=(2, 3))[:, :, np.newaxis, np.newaxis]

        # remove mean pixels per channel
        Xb -= Xb.mean(axis=0)

        # Flip half of the images in this batch at random in both dimensions
        bs = Xb.shape[0]

        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]

        return Xb, labels[:, np.newaxis]

def get_net():
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            #('drop1', DropoutLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            #('drop2', DropoutLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            #('drop3', DropoutLayer),
            ('hidden4', layers.DenseLayer),
            #('drop4', DropoutLayer),
            ('hidden5', layers.DenseLayer),
            #('drop5', DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, C, W, H),
        conv1_num_filters=32, conv1_filter_size=(7, 7), 
        conv1_border_mode='same',
        conv1_strides=(2, 2),
        conv1_W=init.GlorotUniform(0.01),

        pool1_ds=(4, 4), pool1_strides=(2, 2),
        #drop1_p=0.2,

        conv2_num_filters=64, conv2_filter_size=(5, 5), 
        conv2_border_mode='same',
        conv2_strides=(2, 2),
        conv2_nonlinearity=rectify,
        conv2_W=init.GlorotUniform(0.01),

        pool2_ds=(3, 3),
        #drop2_p=0.2,

        conv3_num_filters=128, conv3_filter_size=(4, 4),
        conv3_border_mode='full',
        #conv3_strides=(2, 2),
        conv3_nonlinearity=rectify,
        conv3_W=init.GlorotUniform(0.01),
        
        pool3_ds=(3, 3),
        #drop3_p=0.3,

        hidden4_num_units=1024, hidden4_nonlinearity=rectify,

        hidden5_num_units=1024, hidden5_nonlinearity=rectify,
        
        output_num_units=1, 
        output_nonlinearity=None,

        batch_iterator_train=FlipBatchIterator(batch_size=128),
        batch_iterator_test=FlipBatchIterator(batch_size=128),

        update=updates.nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.02)),
        update_momentum=theano.shared(float32(0.9)),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.02, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],

        use_label_encoder=False,

        regression=True,
        max_epochs=200,
        verbose=2,
    )
    return net


def get_labels(names):
    return np.array(pd.read_csv(LABEL_FILE, index_col=0).loc[names]).flatten()


def get_image_files(datadir):
    fs = [os.path.join(dp, f) for dp, dn, fn in os.walk(datadir) for f in fn]
    return [x for x in fs if x.endswith('.tiff')]


def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]


def load_images(files):
    images = np.array([np.array(Image.open(f)).transpose(2, 1, 0) 
                       for f in files])
    return images.astype(np.float32)

def main():

    print('loading data...')
    files = np.array(get_image_files(TRAIN_DIR))
    names = get_names(files)
    y = get_labels(names).astype(np.float32)
    files, y = shuffle(files, y)

    print('fitting ...')
    net = get_net()
    net.fit(files, y)


if __name__ == '__main__':
    main()
