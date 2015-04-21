"""Resize and crop images to square, save as tiff.

Assumes the training data is in ./data/train
"""
from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import numpy as np

from PIL import Image

# 2015 04 21
# the images seem to be 3:2 width:height, but didn't verify for all of them
SIZE = 128
WIDTH = SIZE // 2 * 3 
FOLDER = 'data/train'


def process(fname):
    img = Image.open(fname)
    img = img.resize([WIDTH, SIZE])

    # crop center square
    box = [(WIDTH - SIZE) // 2, 0, WIDTH - (WIDTH - SIZE) // 2, SIZE]
    img = img.crop(box)

    img.save(fname.replace('jpeg', 'tiff').replace('train', 'res'))


def main(folder=FOLDER):
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(folder) 
                 for f in fn if f.endswith('jpeg')]

    # process in batches, sometimes weird things happen with Pool
    batchsize = 500
    batches = len(filenames) // batchsize + 1

    pool = Pool()

    print("resizing images, this takes a while")

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        m = pool.map(process, filenames[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')


if __name__ == '__main__':
    main()
