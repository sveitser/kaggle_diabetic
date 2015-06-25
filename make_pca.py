from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import mahotas as mh
import numpy as np
from sklearn.decomposition import PCA

import util

def process(fname):
    img = util.load_image_uint_one(fname)
    return np.hstack([mh.features.haralick(channel).ravel() 
                      for channel in img])

@click.command()
@click.option('--directory', default='data/train_res')
def main(directory):

    filenames = util.get_image_files(directory)
    n = len(filenames)

    batches = np.split(filenames, 14)

    for batch in batches:
        images = util.load_image(batch)
        X = images.transpose(0, 2, 3, 1).reshape(-1, 3)
        pca = PCA()
        pca.fit(X)
        print(pca.components_)
        print(pca.explained_variance_ratio_/np.linalg.norm(pca.explained_variance_ratio_))



    
    #fname = 'data/haralick_{}.npy'.format(directory.split('/')[-1])

    #np.save(open(fname, 'wb'), features)

    #print('saved {} features to {}'.format(features.shape, fname))


if __name__ == '__main__':
    main()
