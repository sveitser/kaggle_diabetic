from __future__ import print_function
from time import time

import pandas as pd
import numpy as np

import lasagne
import lasagne.layers
from lasagne import init

from lasagne.updates import nesterov_momentum
from lasagne import updates
from lasagne.objectives import Objective
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano import tensor as T

from definitions import *
from quadratic_weighted_kappa import quadratic_weighted_kappa
import util
from iterator import SingleIterator

import augment


def create_net(mean, layers, tta=False):
    net = AggNet(
        n_eval=TEST_ITER,
        layers=layers,
        batch_iterator_train=SingleIterator(batch_size=BATCH_SIZE,
                                            mean=mean, deterministic=False,
                                            resample=True),
        batch_iterator_test=SingleIterator(batch_size=BATCH_SIZE, mean=mean, 
                                           deterministic=False if tta else True,
                                           resample=False,
                                           iterations=TEST_ITER),
        update=updates.nesterov_momentum,
        update_learning_rate=theano.shared(float32(INITIAL_LEARNING_RATE)),
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
        objective=RegularizedObjective,
        use_label_encoder=False,
        eval_size=0.1,
        regression=REGRESSION,
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
    def __init__(self, name='update_learning_rate', factor=0.1, 
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


class AggNet(NeuralNet):
    def __init__(self, n_eval=1, *args, **kwargs):
        self.n_eval = n_eval
        super(AggNet, self).__init__(*args, **kwargs)

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

        loss_train = obj.get_loss(X_batch, y_batch)
        loss_eval = obj.get_loss(X_batch, y_batch, deterministic=True)
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


    def transform(self, X):
        features = []
        for Xb, yb in self.batch_iterator_test(X):
            features.append(self.transform_iter_(Xb))
        return np.vstack(features)


    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

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
        first_iteration = True
        num_epochs_past = len(self.train_history_)

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            custom_score = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)


            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

            if self.custom_score:
                y_probs = []
                y_trues = []
                for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                    y_probs.append(self.predict_iter_(Xb))
                    y_trues.append(yb)

                y_prob = np.vstack(y_probs)
                y_true = np.vstack(y_trues)
                
                y_true = y_true.ravel().reshape(self.n_eval, -1).mean(axis=0)
                y_prob = y_prob.ravel().reshape(self.n_eval, -1).mean(axis=0)

                custom_score = self.custom_score[1](y_true, y_prob)

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)

            if custom_score:
                avg_custom_score = np.mean(custom_score)

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
                info[self.custom_score[0]] = custom_score
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)
    
    def predict_proba(self, X):
        probas = []
        for Xb, yb in self.batch_iterator_test(X):
            probas.append(self.predict_iter_(Xb))
        return np.vstack(probas)

    def predict(self, X):
        if self.regression:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            if self.use_label_encoder:
                y_pred = self.enc_.inverse_transform(y_pred)
            return y_pred
