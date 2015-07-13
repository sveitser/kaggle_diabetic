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

N_PROC = 2

def convert(fname, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original 
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
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


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpeg', extension).replace(directory, 
                                                    convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory, 
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname) 


def save(img, fname):
    img.save(fname, quality=97)

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
    pool = Pool(N_PROC)

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
