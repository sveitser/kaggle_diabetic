import click
import numpy as np

import util

@click.command()
@click.option('--directory', default='data/train_res')
def aux(directory):
    files = data.get_image_files(directory)
    means = []
    maxes = []
    stds = []
    for i, f in enumerate(files):
        if i % 1000 == 0:
            print(" {} / {}".format(i, len(files)))
        img = util.load_image_uint_one(f)
        means.append(img.mean(axis=(1, 2)))
        maxes.append(img.max(axis=(1, 2)))
        stds.append(img.std(axis=(1, 2)))

    means = np.vstack(means)
    maxes = np.vstack(maxes)
    stds = np.vstack(stds)

    feats = np.hstack([means, maxes, stds])

    fname = 'data/transform/aux' + ('_test' if 'test' in directory else '') \
            + '.npy'
    np.save(open(fname, 'wb'), feats)

if __name__ == '__main__':
    aux()
