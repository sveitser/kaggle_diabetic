from __future__ import print_function
from collections import Counter
from datetime import datetime
import pprint
from time import time
import sys

import pandas as pd
import numpy as np

import lasagne
import lasagne.layers
from lasagne import init

from lasagne.updates import nesterov_momentum
from lasagne import updates
from lasagne.objectives import (MaskedObjective, Objective,
                                categorical_crossentropy)
from lasagne.layers import get_output
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano import tensor as T

from definitions import *
from quadratic_weighted_kappa import quadratic_weighted_kappa
import util
from iterator import SingleIterator

import augment


#def ordistic_loss(y_true, y_pred):
#    zero = T.fvector('zero')
#    T.maximum(T.abs_(y_true - y_pred) - 1, zero)

#def mask_from_labels(target):
#    # coeffs for polynomial  balanced weights for training set vs range(5)
#    #w = [1, 5, 3, 8, 10]
#    #w = np.array([ -0.16940318,   0.63021986,   2.52366008, -11.51726516,
#    #              11.13624402,   0.27218907], dtype=np.float32)
#    w = np.array([ -0.20611039,   0.76943727,   3.03613623, -14.40281366,
#                   14.80335056,   1.        ], dtype=np.float32) / 10.0
#    return w[-1] + w[-2] * target + w[-3] * target**2 + w[-4] * target**3 \
#           + w[-5] * target**4

def loss(y_true, y_pred):
    return abs(y_true - y_pred)

#
#   TODO make this take arguments
#
def create_net(model):
    net = Net(
        layers=model.layers,
        batch_iterator_train=SingleIterator(
            model, batch_size=model.get('batch_size_train', BATCH_SIZE),
            deterministic=False, resample=True),
        # TODO pass deterministic argument
        batch_iterator_test=SingleIterator(
            model, batch_size=model.get('batch_size_test', BATCH_SIZE), 
            deterministic=False, resample=False),
        update=updates.nesterov_momentum,
        update_learning_rate=theano.shared(
            float32(model.get('learning_rate', INITIAL_LEARNING_RATE))),
        update_momentum=theano.shared(float32(INITIAL_MOMENTUM)),
        on_epoch_finished=[
            #AdjustVariable('update_learning_rate', start=INITIAL_LEARNING_RATE,
            #               stop=0.0001),
            AdjustVariable('update_momentum', start=INITIAL_MOMENTUM,
                            stop=0.999),
            EarlyStopping(loss=CUSTOM_SCORE_NAME, greater_is_better=True),
            AdjustLearningRate('update_learning_rate',
                                loss=CUSTOM_SCORE_NAME, 
                                greater_is_better=True,
                                patience=PATIENCE // 2),
        ],
        custom_score=(CUSTOM_SCORE_NAME, util.kappa),
        #loss_mask=util.get_mask(y),
        objective=RegularizedObjective,
        #objective=MyMasked,
        #objective_loss_function=loss,
        use_label_encoder=False,
        eval_size=0.1,
        regression=model.get('regression', REGRESSION),
        max_epochs=MAX_ITER,
        verbose=2,
    )
    return net


def float32(k):
    return np.cast['float32'](k)


class RegularizedObjective(Objective):

    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):

        loss = super(RegularizedObjective, self).get_loss(
            input=input, target=target, deterministic=deterministic, **kwargs)
        if not deterministic:
            return loss \
                + 0.0005 * lasagne.regularization.l2(self.input_layer)
        else:
            return loss


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


class ScoreMonitor(object):
    def __init__(self, patience=PATIENCE, loss='valid_loss',
                 greater_is_better=False):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.loss = loss
        self.greater_is_better = greater_is_better

    def _act(self):
        raise NotImplementedError

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.loss] \
            * (-1.0 if self.greater_is_better else 1.0)
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
            nn.save_params_to(WEIGHTS)
        elif self.best_valid_epoch + self.patience < current_epoch:
            self._act(nn, train_history)


class LossThreshold(object):
    """Stop when loss reaches a certain threshold (when retraining)."""
    def __init__(self, threshold, loss='train_loss'):
        self.threshold = threshold
        self.loss = loss

    def __call__(self, nn, train_history):
        current_loss = train_history[1][self.loss]
        if current_loss < self.threshold:
            print("Train loss threshold {} reached. Stopping."
                  "".format(self.threshold))
            raise StopIteration


class EarlyStopping(ScoreMonitor):
    def _act(self, nn, train_history):
        print("Early stopping.")
        print("Best valid loss was {:.6f} at epoch {}.".format(
            self.best_valid, self.best_valid_epoch))
        #nn.load_params_from(self.best_weights)
        raise StopIteration


class AdjustLearningRate(ScoreMonitor):
    def __init__(self, name='update_learning_rate', factor=DECAY_FACTOR, 
                 *args, **kwargs):
        self.name = name
        self.factor = factor
        super(AdjustLearningRate, self).__init__(*args, **kwargs)

    def _act(self, nn, train_history):
        self.best_valid = np.inf
        old_value = getattr(nn, self.name).get_value()
        new_value = float32(old_value * self.factor)
        print("decreasing {} from {} to {}".format(self.name, old_value,
                                                   new_value))
        getattr(nn, self.name).set_value(new_value)



class QueueIterator(BatchIterator):
    """BatchIterator with seperate thread to do the image reading."""
    def __iter__(self):
        queue = Queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for Xb, yb in super(QueueIterator, self).__iter__():
                queue.put((np.array(Xb), np.array(yb)))
            queue.put(end_marker)

        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not end_marker:
            yield item
            queue.task_done()
            item = queue.get()


class Net(NeuralNet):

    def train_test_split(self, X, y, eval_size):
        if eval_size:
            kf = util.cross_validation.StratifiedShuffleSplit(
                    y, test_size=eval_size, n_iter=1, 
                    random_state=RANDOM_STATE)

            train_indices, valid_indices = next(iter(kf))
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = X[len(X):], y[len(y):]

        return X_train, X_valid, y_train, y_valid

    def initialize(self):
        if getattr(self, '_initialized', False):
            return

        out = getattr(self, '_output_layer', None)
        if out is None:
            out = self._output_layer = self.initialize_layers()
        self._check_for_unused_kwargs()
        if self.verbose:
            self._print_layer_info(self.layers_.values())

        if self.X_tensor_type is None:
            types = {
                2: T.matrix,
                3: T.tensor3,
                4: T.tensor4,
                }
            first_layer = list(self.layers_.values())[0]
            self.X_tensor_type = types[len(first_layer.shape)]

        iter_funcs = self._create_iter_funcs(
            self.layers_, self.objective, self.update,
            self.X_tensor_type,
            self.y_tensor_type,
            )
        for name, fun in iter_funcs.items():
            setattr(self, name, fun)
        #self.train_iter_, self.eval_iter_, self.predict_iter_ = iter_funcs
        self._initialized = True

    def _create_iter_funcs(self, layers, objective, update, input_type,
                           output_type):
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        output_layer = list(layers.values())[-1]
        objective_params = self._get_params_for('objective')
        obj = objective(output_layer, **objective_params)
        if not hasattr(obj, 'layers'):
            # XXX breaking the Lasagne interface a little:
            obj.layers = layers


        loss_params = self._get_params_for('loss')
        loss_train = obj.get_loss(X_batch, y_batch, **loss_params)
        loss_eval = obj.get_loss(X_batch, y_batch, deterministic=True,
                                 **loss_params)
        predict_proba = output_layer.get_output(X_batch, deterministic=True)
        transform = list(layers.values())[-2].get_output(X_batch,
                                                         deterministic=True)
        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = self.get_all_params()
        update_params = self._get_params_for('update')
        updates = update(loss_train, all_params, **update_params)

        train_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_train],
            updates=updates,
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        eval_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_eval, accuracy],
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        predict_iter = theano.function(
            inputs=[theano.Param(X_batch)],
            outputs=predict_proba,
            givens={
                X: X_batch,
                },
            )
        transform_iter = theano.function(
            inputs=[theano.Param(X_batch)],
            outputs=transform,
            givens={
                X: X_batch,
                },
            )

        return {
            'train_iter_': train_iter,
            'eval_iter_': eval_iter,
            'predict_iter_': predict_iter,
            'transform_iter_': transform_iter
        }

    def _check_for_unused_kwargs(self):
        names = list(self.layers_.keys()) + ['update', 'objective', 'loss']
        for k in self._kwarg_keys:
            for n in names:
                prefix = '{}_'.format(n)
                if k.startswith(prefix):
                    break
            else:
                raise ValueError("Unused kwarg: {}".format(k))

    def fit(self, X, y):
        self.objective.mask = util.get_mask(y)
        return super(Net, self).fit(X, y)

    def transform(self, X):
        features = []
        for Xb, yb in self.batch_iterator_test(X):
            features.append(self.transform_iter_(Xb))
        return np.vstack(features)
    
    def predict_proba(self, X):
        probas = []
        for Xb, yb in self.batch_iterator_test(X):

            # add dummy data for nervana kernels that need batch_size % 8 = 0
            missing = (8 - len(Xb) % 8) % 8
            if missing != 0:
                tiles = np.ceil(float(missing) / len(Xb)).astype(int)
                Xb = np.tile(Xb, [tiles] + [1] * (Xb.ndim - 1))[:8]

            preds = self.predict_iter_(Xb)

            if missing != 0:
                preds = preds[-missing:]

            probas.append(preds)

        return np.vstack(probas)

    def predict(self, X):
        if self.regression:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            if self.use_label_encoder:
                y_pred = self.enc_.inverse_transform(y_pred)
            return y_pred

    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            #custom_score = []
            y_pred, y_true = [], []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)
                y_true.append(yb)
                if self.custom_score:
                    y_prob = self.predict_iter_(Xb)
                    y_pred.append(y_prob)
                    #custom_score.append(self.custom_score[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if self.custom_score:
                #avg_custom_score = np.mean(custom_score)
                avg_custom_score = self.custom_score[1](np.vstack(y_true),
                                                        np.vstack(y_pred))

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
            if self.custom_score:
                info[self.custom_score[0]] = avg_custom_score
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

#class MyMasked(MaskedObjective):
#    def get_loss(self, input=None, target=None, mask=None,
#                 aggregation=None, deterministic=False, **kwargs):
#        """
#        Get loss scalar expression
#
#        :parameters:
#            - input : (default `None`) an expression that results in the
#                input data that is passed to the network
#            - target : (default `None`) an expression that results in the
#                desired output that the network is being trained to generate
#                given the input
#            - mask : None for no mask, or a soft mask that is the same shape
#                as - or broadcast-able to the shape of - the result of
#                applying the loss function. It selects/weights the
#                contributions of the resulting loss values
#            - aggregation : None to use the value passed to the
#                constructor or a value to override it
#            - kwargs : additional keyword arguments passed to `input_layer`'s
#                `get_output` method
#
#        :returns:
#            - output : loss expressions
#        """
#        print(input, input.shape)
#        print(target, target.shape)
#        print(mask, mask.shape)
#        print(np.unique(mask))
#
#        network_output = get_output(self.input_layer, input, **kwargs)
#        if target is None:
#            target = self.target_var
#        if mask is None:
#            mask = self.mask_var
#
#        if aggregation not in self._valid_aggregation:
#            raise ValueError('aggregation must be \'mean\', \'sum\', '
#                             '\'normalized_sum\' or None, '
#                             'not {0}'.format(aggregation))
#
#        # Get aggregation value passed to constructor if None
#        if aggregation is None:
#            aggregation = self.aggregation
#
#        if deterministic:
#            masked_losses = self.loss_function(network_output, target)
#        else:
#            masked_losses = self.loss_function(network_output, target) \
#                    * mask_from_labels(target)
#
#        if aggregation is None or aggregation == 'mean':
#            return masked_losses.mean()
#        elif aggregation == 'sum':
#            return masked_losses.sum()
#        elif aggregation == 'normalized_sum':
#            return masked_losses.sum() / mask.sum()
#        else:
#            raise RuntimeError('This should have been caught earlier')
#
