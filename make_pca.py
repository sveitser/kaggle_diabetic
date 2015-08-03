from __future__ import division, print_function

import click
import numpy as np

import data
import util

def process(fname):
    img = util.load_image_uint_one(fname)
    return np.hstack([mh.features.haralick(channel).ravel() 
                      for channel in img])

@click.command()
@click.option('--directory', default='data/train_tiny')
def main(directory):

    filenames = data.get_image_files(directory)

    bs = 1000
    batches = [filenames[i * bs : (i + 1) * bs] 
               for i in range(int(len(filenames) / bs) + 1)]

    Us, evs = [], []
    for batch in batches:
        images = np.array([data.load_augment(f, 128, 128) for f in batch])
        X = images.transpose(0, 2, 3, 1).reshape(-1, 3)
        cov = np.dot(X.T, X) / X.shape[0]
        U, S, V = np.linalg.svd(cov)
        ev = np.sqrt(S)
        Us.append(U)
        evs.append(ev)

    print('U')
    print(np.mean(Us, axis=0))
    print('eigenvalues')
    print(np.mean(evs, axis=0))

if __name__ == '__main__':
    main()
