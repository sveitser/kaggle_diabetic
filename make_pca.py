from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import mahotas as mh
import numpy as np
from sklearn.decomposition import PCA

import augment
import util


def process(fname):
    img = util.load_image_uint_one(fname)
    return np.hstack([mh.features.haralick(channel).ravel() 
                      for channel in img])

@click.command()
@click.option('--directory', default='data/train_medium')
def main(directory):

    filenames = data.get_image_files(directory)
    n = len(filenames)

    bs = 1000
    batches = [filenames[i * bs : (i + 1) * bs] 
               for i in range(int(len(filenames) / bs) + 1)]
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)

    Us, evs = [], []
    for batch in batches:
        #images = util.load_image(batch)
        images = np.array([augment.load(f, w=512, h=512, deterministic=True,
                                        mean=mean, std=std)
                           for f in batch])
        X = images.transpose(0, 2, 3, 1).reshape(-1, 3)
        cov = np.dot(X.T, X) / X.shape[0]
        print(cov)
        U, S, V = np.linalg.svd(cov)
        ev = np.sqrt(S)
        Us.append(U)
        evs.append(ev)
        print(U)
        print(ev)

    print(np.mean(Us, axis=0))
    print(np.mean(evs, axis=0))

if __name__ == '__main__':
    main()
