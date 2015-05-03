import time

import numpy as np
from scipy.ndimage.interpolation import rotate
import skimage
import skimage.transform

import util

from definitions import MAX_PIXEL_VALUE


def crop_random(img, size):
    x_offset, y_offset = [np.random.randint(dim - size) for dim in img.shape]
    return img[x_offset: x_offset + size, y_offset: y_offset + size]


def rgb_mix(img):
    r = 1.0 + 0.3 * (np.random.rand(3).astype(np.float32) - 0.5)
    return img * r[:, np.newaxis, np.newaxis]


def rotate_uniform(img):
    return rotate(img, 360 * np.random.rand(), axes=(1, 2),
                  reshape=False, order=0)


def augment(img):
    return rgb_mix(img)


def load_transform(args):
    fname, mean, deterministic = args
    img = (util.load_image(fname) - mean) / MAX_PIXEL_VALUE
    if deterministic:
        return img
    else:
        return augment(img)

