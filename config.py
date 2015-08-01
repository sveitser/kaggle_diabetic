from copy import deepcopy
from datetime import datetime
import pprint
import os

import numpy as np

import augment
import util
from definitions import *

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


class Config(object):
    def __init__(self, layers, cnf=None):
        self.layers = layers
        self.cnf = cnf if cnf is not None else {}
        self.setup(cnf)
        pprint.pprint(cnf)
        #pprint.pprint([(l.__name__, p) for l, p in layers])


    def setup(self, cnf):
        cnf = deepcopy(cnf)
        cnf['mean'] = np.array(cnf.get('mean', MEAN), dtype=np.float32)
        cnf['std'] = np.array(cnf.get('std', STD), dtype=np.float32)
        self.cnf = cnf

    def load(self, fname, *args, **kwargs):
        img = util.load_image_uint_one(fname)
        img -= np.array(MEAN, dtype=np.float32)[:, np.newaxis, np.newaxis]
        img /= np.array(STD, dtype=np.float32)[:, np.newaxis, np.newaxis]
        if kwargs.get('deterministic') is True:
            return augment.crop(img, w=self.cnf['w'], h=self.cnf['h'])
        elif kwargs.get('rotate') is True:
            return augment.perturb(img / 255.0) * 255.0
        else:
            return augment.crop_random(img, w=self.cnf['w'], h=self.cnf['h'])

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

