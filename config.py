from datetime import datetime
import pprint
import os

import numpy as np

from util import mkdir

TRANSFORM_DIR = 'data/transform'

class Config(object):
    def __init__(self, layers, cnf=None):
        self.layers = layers
        self.cnf = cnf
        pprint.pprint(cnf)

    def get(self, k, default=None):
        return self.cnf.get(k, default)

    @property
    def weights_epoch(self):
        path = "weights/{}/epochs".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    @property
    def weights_best(self):
        path = "weights/{}/best".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    @property
    def weights_file(self):
        path = "weights/{}".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    @property
    def retrain_weights_file(self):
        path = "weights/{}/retrain".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    @property
    def final_weights_file(self):
        path = "weights/{}".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights_final.pkl')

    @property
    def logfile(self):
        mkdir('log')
        t = datetime.now().replace(microsecond=0).isoformat()
        return 'log/{}_{}.log'.format(self.get('name'), t)

    def get_transform_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_mean_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'),  n_iter, skip)
        return os.path.join(TRANSFORM_DIR, fname)

    def get_std_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_std_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'), n_iter, skip)
        return os.path.join(TRANSFORM_DIR, fname)

    def save_transform(self, X, n_iter, skip=0, test=False):
        mkdir(TRANSFORM_DIR)
        np.save(open(self.get_transform_fname(n_iter, skip=skip, 
                                              test=test), 'wb'), X)

    def save_std(self, X, n_iter, skip=0, test=False):
        mkdir(TRANSFORM_DIR)
        np.save(open(self.get_std_fname(n_iter, skip=skip,
                                        test=test), 'wb'), X)

    def load_transform(self, test=False):
        return np.load(open(self.get_transform_fname(test=test)))

