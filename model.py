from datetime import datetime
import pprint
import os

import numpy as np

import augment
import util
from definitions import *

class Model(object):
    def __init__(self, layers, cnf=None):
        self.layers = layers
        self.cnf = cnf if cnf is not None else {}
        self.setup(cnf)
        pprint.pprint(cnf)
        pprint.pprint([(l.__name__, p) for l, p in layers])


    def setup(self, cnf):
        cnf['mean'] = cnf.get('mean', MEAN)
        cnf['std'] = cnf.get('std', STD)
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
    def logfile(self):
        try:
            os.mkdir('log')
        except OSError:
            pass
        t = datetime.now().replace(microsecond=0).isoformat()
        return 'log/{}_{}.log'.format(self.get('name'), t)

