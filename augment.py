import time

import numpy as np
from scipy.ndimage.interpolation import rotate
import skimage
import skimage.transform

import util

from definitions import *

def crop_random(img, w=W, h=H):
    x_offset, y_offset = [np.random.randint(dim - s)
                          for dim, s in zip(img.shape[1:], [w, h])]
    return img[:, x_offset: x_offset + w, y_offset: y_offset + h]


def crop(img, w=W, h=H):
    lx, ly = img.shape[1:]
    x0, x1 = (lx - w) // 2, lx - (lx - w + 1) // 2
    y0, y1 = (ly - h) // 2, ly - (ly - h + 1) // 2
    cropped = img[:, x0: x1, y0: y1]
    return cropped


def rgb_mix(img):
    r = 1.0 + 0.3 * (np.random.rand(3).astype(np.float32) - 0.5)
    return img * r[:, np.newaxis, np.newaxis]


def rotate_uniform(img):
    return rotate(img, 360 * np.random.rand(), axes=(1, 2),
                  reshape=False, order=0)


def augment(img):
    return crop_random(img)


def load_transform(fname, mean, deterministic):
    img = (util.load_image(fname) - mean) / MAX_PIXEL_VALUE
    #fname, mean, deterministic = args
    #img = (util.load_image(fname) - mean) / MAX_PIXEL_VALUE
    if deterministic:
        return crop(img)
    else:
        return crop_random(img)

