from lasagne import layers, nonlinearities
import lasagne
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split

import theano
import theano.tensor as T
import sys
from numpy import random
import os
from PIL import Image
from skimage import transform
import numpy as np
import cPickle as pickle
from nolearn.lasagne import NeuralNet, BatchIterator
from skimage.feature import hog
from skimage import color
from skimage.transform import rotate

sys.setrecursionlimit(10000)


def float32(k):
    return np.cast['float32'](k)


def int32(k):
    return np.cast['int32'](k)


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


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class LoadDataCNN:
    def load_single_class(self, X, y, pixels, path_class, total_number_of_images, class_nr):
        #  path_class0 = '/media/stephan/12C3-91C8/Kaggle.com/diabetic_retinopathy/kaggle diabetic_/Class0_Resized_1024_1024'

        imgs = os.listdir(path_class)
        number_of_images_per_class = total_number_of_images/2
        readout = range(0, len(imgs))
        random.shuffle(readout)

        for i in range(0, number_of_images_per_class):
            number_of_current_image = i
            img = Image.open(path_class + '/' + imgs[readout[number_of_current_image]])
            img = np.asarray(img, dtype='float32')/255.
            img = transform.resize(img, (pixels, pixels))
            img = img.transpose(2, 0, 1).reshape(3, pixels, pixels)
            X[number_of_current_image] = img
            y[number_of_current_image] = class_nr

        return X, y


class TrainIterator(BatchIterator):
    transformation_probability = 0

    def transform(self, X_batch, y_batch):
        number_of_epochs = 3000
        pixels = 256
        print('Transforming data for fitting')
        X_batch, y_batch = super(TrainIterator, self).transform(X_batch, y_batch)
        batch_size = X_batch.shape[0]
        print(batch_size)

        # The training data is shuffled.
        indices = np.random.permutation(y_batch.size)
        y_batch = y_batch[indices]
        X_batch = X_batch[indices, :, :, :]

        probability_augmentation = random.rand(1)
        rand_number_augmentation = 0.0 + TrainIterator.transformation_probability/float(number_of_epochs)
        TrainIterator.transformation_probability = TrainIterator.transformation_probability + 1/float(number_of_epochs)

        if probability_augmentation <= rand_number_augmentation:
            augmentation_type = random.randinit(1, 7)

            if augmentation_type == 1:
                # Flip the images from left to right
                image_number = 0
                for current_image in X_batch[:, 1, 1, 1]:
                    X_batch[image_number, 0, :, :] = np.fliplr(X_batch[image_number, 0, :, :])
                    X_batch[image_number, 1, :, :] = np.fliplr(X_batch[image_number, 1, :, :])
                    X_batch[image_number, 2, :, :] = np.fliplr(X_batch[image_number, 2, :, :])
                    image_number = image_number+1

            elif augmentation_type == 2:
                # Flip the images upside down.
                image_number = 0
                for current_image in X_batch[:, 1, 1, 1]:
                    X_batch[image_number, 0, :, :] = np.flipud(X_batch[image_number, 0, :, :])
                    X_batch[image_number, 1, :, :] = np.flipud(X_batch[image_number, 1, :, :])
                    X_batch[image_number, 2, :, :] = np.flipud(X_batch[image_number, 2, :, :])
                    image_number = image_number+1

            elif augmentation_type == 3:
                # Only use Greenchannel
                X_batch[:, 0, :, :] = 0
                X_batch[:, 2, :, :] = 0

            elif augmentation_type == 4:
                number_of_pictures = X_batch.shape[0]
                image_number = 0
                for current_image_number in X_batch[:, 1, 1, 1]:
                    current_image = color.rgb2gray(X_batch[image_number, :, :, :].reshape(pixels, pixels, 3))
                    fd, hog_image = hog(current_image, orientations=8, pixels_per_cell=(pixels, pixels),
                    cells_per_block=(1, 1), visualise=True)
                    X_batch[image_number, 0, :, :] = 0
                    X_batch[image_number, 1, :, :] = hog_image
                    X_batch[image_number, 2, :, :] = 0
                    image_number = image_number+1

            elif augmentation_type == 5:
                #  rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
                scale_factor = np.around(random.rand(1)*1.6, decimals=1)
                if scale_factor < 1:
                    scale_factor = 1
                scale_factor = float(1.2)

                middle_point_new_size = np.round(transform.rescale(X_batch[0, 0, :, :], scale_factor).shape[0]/2)
                boundary_down_readout = middle_point_new_size-pixels/2
                boundary_up_readout = middle_point_new_size+pixels/2

                image_number=0
                for current_image in X_batch[:, 1, 1, 1]:
                    scaled_picture = transform.rescale(X_batch[image_number, 0, :, :], scale_factor)
                    X_batch[image_number, 0, :, :] = scaled_picture[boundary_down_readout:boundary_up_readout, boundary_down_readout:boundary_up_readout]
                    scaled_picture = transform.rescale(X_batch[image_number, 1, :, :], scale_factor)
                    X_batch[image_number, 1, :, :] = scaled_picture[boundary_down_readout:boundary_up_readout, boundary_down_readout:boundary_up_readout]
                    scaled_picture = transform.rescale(X_batch[image_number, 2, :, :], scale_factor)
                    X_batch[image_number, 2, :, :] = scaled_picture[boundary_down_readout:boundary_up_readout, boundary_down_readout:boundary_up_readout]
                    image_number = image_number+1
                X_batch = float32(X_batch)

            elif augmentation_type == 6:
                # Shift between -10 and 10
                shift_strength = random.randint(-50, 50)
                shift_right = False
                if shift_strength >= 0:
                    shift_right = True

                image_number = 0
                for current_image in X_batch[:, 1, 1, 1]:
                    if shift_right:
                        shifted_image = np.hstack((np.zeros([pixels, shift_strength]), np.squeeze(X_batch[image_number, 0, :, :])))
                        X_batch[image_number, 0, :, :] = shifted_image[0:pixels, 0:pixels]
                        shifted_image = np.hstack((np.zeros([pixels, shift_strength]), np.squeeze(X_batch[image_number, 1, :, :])))
                        X_batch[image_number, 1, :, :] = shifted_image[0:pixels, 0:pixels]
                        shifted_image = np.hstack((np.zeros([pixels, shift_strength]), np.squeeze(X_batch[image_number, 2, :, :])))
                        X_batch[image_number, 2, :, :] = shifted_image[0:pixels, 0:pixels]

                    else:
                        shift_strength = abs(shift_strength)
                        shifted_image = np.hstack((np.squeeze(X_batch[image_number, 0, :, :]), np.zeros([pixels, shift_strength])))
                        X_batch[image_number, 0, :, :] = shifted_image[0:pixels, shift_strength : (shift_strength+pixels)]
                        shifted_image = np.hstack((np.squeeze(X_batch[image_number, 1, :, :]), np.zeros([pixels, shift_strength])))
                        X_batch[image_number, 1, :, :] = shifted_image[0:pixels, shift_strength:]
                        shifted_image = np.hstack((np.squeeze(X_batch[image_number, 2, :, :]), np.zeros([pixels, shift_strength])))
                        X_batch[image_number, 2, :, :] = shifted_image[0:pixels, shift_strength:]
                    image_number = image_number + 1

            elif augmentation_type == 7:
                image_number = 0
                angle = random.randint(0, 360)
                for current_image in X_batch[:, 1, 1, 1]:
                    X_batch[image_number, 0, :, :] = rotate(np.squeeze(X_batch[image_number, 0, :, :]), angle)
                    X_batch[image_number, 1, :, :] = rotate(np.squeeze(X_batch[image_number, 1, :, :]), angle)
                    X_batch[image_number, 2, :, :] = rotate(np.squeeze(X_batch[image_number, 2, :, :]), angle)
                    image_number = image_number + 1

        return X_batch, y_batch


class neural_network:
    def prepare_data_cnn(self, pixels):
        total_number_of_images = 1000
        number_of_colour_channels = 3
        X = np.zeros((total_number_of_images, number_of_colour_channels, pixels, pixels), dtype='float32')
        y = np.zeros(total_number_of_images)

        load_data_class = LoadDataCNN()

        #  path_class0 = os.getcwd() + '/Class0_Resized_1024_1024'
        path_class0 = '/media/stephan/12C3-91C8/Kaggle.com/diabetic_retinopathy/kaggle diabetic_/Class0_Resized_1024_1024'

        X, y = load_data_class.load_single_class(X, y, pixels, path_class0, total_number_of_images, class_nr=0)

        #  path_class1 = os.getcwd() + '/Class1_Resized_1024_1024'
        path_class1 = '/media/stephan/12C3-91C8/Kaggle.com/diabetic_retinopathy/kaggle diabetic_/Class1_Resized_1024_1024'

        X, y = load_data_class.load_single_class(X, y, pixels, path_class1, total_number_of_images, class_nr=1)

        y = int32(y)
        return X, y

    def save_results(self, net1):
        with open('net1.pickle', 'wb') as f:
            pickle.dump(net1, f, -1)

    def return_attribute(self, net1):
        return net1.get_all_params_values()

    def create_cnn(self, pixels):
        net1 = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('hidden4', layers.DenseLayer),
                ('hidden5', layers.DenseLayer),
                ('output', layers.DenseLayer),
            ],


            batch_iterator_train=TrainIterator(batch_size=50),
            #  config, batch_size=config.get('batch_size_train', BATCH_SIZE),
            #  deterministic=False, resample=True),

            input_shape=(None, 3, pixels, pixels),
            conv1_num_filters=128, conv1_filter_size=(3, 3), pool1_pool_size=(2,2),
            dropout1_p=0.3,
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
            dropout2_p=0.2,
            conv3_num_filters=32, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
            hidden4_num_units=500,
            hidden5_num_units=300,
            output_num_units=2,
            output_nonlinearity=nonlinearities.softmax,

            # learning rate parameters
            update_learning_rate= theano.shared(float32(0.03)),  #0.01,
            update_momentum=theano.shared(float32(0.9)),
            regression=False,
            #  classification=True,

            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
                EarlyStopping(patience=200),
            ],

            # We only train for 10 epochs
            max_epochs = 3000,
            verbose = 1,

            # Training test-set split
            eval_size = 0.2
        )
        return net1

    def run_cnn(self, pretrain):
        if pretrain is not None:
            try:
                print('Loading already established weights from previous network.')
                with open('net1.pickle', 'rb') as f:
                    net_pretrain = pickle.load(f)
            except:
                print('No network weights could be loaded.')
                net_pretrain = None
        else:
            net_pretrain = None

        print("Loading data...")
        pixels = 256
        X, y = self.prepare_data_cnn(pixels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Create neural network config (depending on first command line parameter)
        print("Building config and compiling functions...")

        network = self.create_cnn(pixels)

        if net_pretrain is not None:
            network.load_weights_from(net_pretrain)

        print("fitting ...")
        network.fit(X_train, y_train)

c_neural_net = neural_network()
pretrain = True
c_neural_net.run_cnn(pretrain)
