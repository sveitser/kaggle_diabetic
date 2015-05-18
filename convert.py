"""Resize and crop images to square, save as tiff.

Assumes the training data is in ./data/train
"""
from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

from skimage import exposure
from skimage.exposure import equalize_adapthist
from skimage.filters import rank
from skimage.morphology import disk

import click
import numpy as np
from PIL import Image, ImageChops, ImageFilter

import util
from definitions import *

# 2015 04 21
# the images seem to be 3:2 width:height, but didn't verify for all of them


def convert(fname, crop_size):
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
    resized = cropped.resize([crop_size, crop_size])
    return resized


def convert_micro(fname, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    _, w, _ = ba.shape

    left_max = ba[:, : w / 64, :].max(axis=(0, 1)).astype(int)
    right_max = ba[:, - w / 64:, :].max(axis=(0, 1)).astype(int)
    max_bg = np.maximum(left_max, right_max)

    foreground = (ba > max_bg + 10).astype(np.uint8)
    bbox = Image.fromarray(foreground).getbbox()

    cropped = img.crop(bbox).resize([crop_size, crop_size], 
                                    resample=Image.LANCZOS)

    green = np.array(cropped)[:, :, 1]
    green = equalize_adapthist(np.array(green, dtype=np.float32) / 255)

    med = rank.median(green, disk(20))

    result = med.astype(np.float32) - 255 * green.astype(np.float32)

    result -= np.min(result)
    result *= 255 / np.max(result)

    return Image.fromarray(result.astype(np.uint8))


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    img = fun(fname, crop_size)
    save(img, fname, extension, directory, convert_directory) 


def save(img, fname, extension, directory, convert_directory):
    img.save(fname.replace('jpeg', extension).replace(directory,
                                                      convert_directory),
                                            quality=97)

@click.command()
@click.option('--directory', default='data/train')
@click.option('--convert_directory', default='data/train_res')
@click.option('--test', is_flag=True, default=False)
@click.option('--crop_size', default=256)
@click.option('--extension', default='tiff')
@click.option('--micro', is_flag=True, default=False)
def main(directory, convert_directory, test, crop_size, extension, micro):

    try:
        os.mkdir(convert_directory)
    except OSError:
        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpeg')]
    filenames = sorted(filenames)

    if test:
        fun = convert_micro if micro else convert
        names = util.get_names(filenames)
        y = util.get_labels(names)
        for f, level in zip(filenames, y):
            if level == 1:
                try:
                    img = fun(f, crop_size)
                    img.show()
                    Image.open(f).show()
                    real_raw_input = vars(__builtins__).get('raw_input',input)
                    real_raw_input('enter for next')
                except KeyboardInterrupt:
                    exit(0)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool
    batchsize = 500
    batches = n // batchsize + 1
    pool = Pool()

    args = []

    fun = convert_micro if micro else convert

    for f in filenames:
        args.append((fun, (directory, convert_directory, f, crop_size, 
                           extension)))

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        m = pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')

if __name__ == '__main__':
    main()
