"""Resize and crop images to square, save as tiff.

Assumes the training data is in ./data/train
"""
from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import numpy as np
from PIL import Image

# 2015 04 21
# the images seem to be 3:2 width:height, but didn't verify for all of them
RESIZE_H = 256
RESIZE_W = RESIZE_H // 2 * 3
CROP_SIZE = 256


def process(args):
    directory, convert_directory, fname = args
    img = Image.open(fname)
    img = img.resize([RESIZE_W, RESIZE_H])

    # crop center square
    left = (RESIZE_W - CROP_SIZE) // 2
    top = (RESIZE_H - CROP_SIZE) // 2
    right = (RESIZE_W + CROP_SIZE) // 2
    bottom = (RESIZE_H + CROP_SIZE) // 2

    img = img.crop([left, top, right, bottom])

    img.save(fname.replace('jpeg', 'tiff').replace(directory,
                                                   convert_directory))


@click.command()
@click.option('--directory', default='data/train')
def main(directory):

    convert_directory = directory + '_res'

    try:
        os.mkdir(convert_directory)
    except OSError:
        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpeg')]

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool
    batchsize = 500
    batches = n // batchsize + 1

    pool = Pool()

    args = zip([directory] * n, [convert_directory] * n, filenames)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        m = pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')


if __name__ == '__main__':
    main()
