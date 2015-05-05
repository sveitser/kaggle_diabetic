"""Resize and crop images to square, save as tiff.

Assumes the training data is in ./data/train
"""
from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import numpy as np
from PIL import Image, ImageChops, ImageFilter

# 2015 04 21
# the images seem to be 3:2 width:height, but didn't verify for all of them
RESIZE_H = 256
RESIZE_W = RESIZE_H // 2 * 3
CROP_SIZE = 256


def process(args):
    directory, convert_directory, fname = args
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    _, w, _ = ba.shape

    left_max = ba[:, : w / 64, :].max(axis=(0, 1)).astype(int)
    right_max = ba[:, - w / 64:, :].max(axis=(0, 1)).astype(int)
    max_bg = np.maximum(left_max, right_max)

    foreground = (ba > max_bg + 10).astype(np.uint8)
    bbox = Image.fromarray(foreground).getbbox()
    cropped = img.crop(bbox)
    resized = cropped.resize([RESIZE_H, RESIZE_H])

    resized.save(fname.replace('jpeg', 'tiff').replace(directory,
                                                       convert_directory))


@click.command()
@click.option('--directory', default='data/train')
@click.option('--convert_directory', default='data/train_res')
@click.option('--test', is_flag=True, default=False)
def main(directory, convert_directory, test):

    try:
        os.mkdir(convert_directory)
    except OSError:
        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpeg')]
    filenames = sorted(filenames)

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool
    batchsize = 500
    batches = n // batchsize + 1

    pool = Pool(8)

    args = zip([directory] * n, [convert_directory] * n, filenames)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    if test:
        args = args[:batchsize]

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        m = pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')


if __name__ == '__main__':
    main()
