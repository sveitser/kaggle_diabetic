from __future__ import division, print_function
from collections import Counter
from datetime import datetime
import importlib
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn import cross_validation
from tempfile import mkdtemp
from joblib import Memory

from quadratic_weighted_kappa import quadratic_weighted_kappa
from definitions import *

MEMORY = Memory(cachedir=CACHE_DIR, verbose=4, compress=0)


def timeit(f):
    def wrapped_f(*args, **kwargs):
        t_in = time.time()
        ret_val = f(*args, **kwargs)
        print("{}.{} took {:.5f} seconds".format(
            getattr(f, 'im_class', '.').split('.', 1)[1],
            f.__name__, time.time() - t_in))
        return ret_val
    return wrapped_f


def compute_mean(files, batch_size=BATCH_SIZE, axis=0):
    """Load images in files in batches and compute mean."""
    first_image = load_image_uint_one(files[0])
    m = np.zeros(first_image.shape)
    for i in range(0, len(files), batch_size):
        images = load_image(files[i : i + batch_size])
        m += images.sum(axis=axis)
    return (m / len(files)).astype(np.float32)


def std(files, batch_size=BATCH_SIZE):
    #data = np.array(load_image_uint(files), dtype=np.uint8)
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print("done with {:>3} / {} images".format(i, len(files)))
        images = np.array(load_image_uint(files[i : i + batch_size]),
                          dtype=np.float64)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var)


def get_mean(files=None, cached=True):
    """Computes mean image per channel of files or loads from cache."""
    if cached:
        try:
            return np.load(open(MEAN_FILE, 'rb)'))
        except IOError:
            if files is None:
                raise ValueError("couldn't load from cache and no files given")
    print("couldn't load mean from file, computing mean images")
    m = compute_mean(files)
    np.save(open(MEAN_FILE, 'wb'), m)
    print("meanfile saved to {}".format(MEAN_FILE))
    return m


def get_labels(names, label_file=LABEL_FILE, per_patient=False):

    labels = pd.read_csv(label_file, index_col=0).loc[names].values.flatten()
    if per_patient:
        left = np.array(['left' in n for n in names])
        return np.vstack([labels[left], labels[~left]]).T
    else:
        return labels


def get_image_files(datadir, left_only=False):
    fs = [os.path.join(dp, f) for dp, dn, fn in os.walk(datadir) for f in fn]

    if left_only:
        fs = [f for f in fs if 'left' in f]

    return np.array(sorted([x for x in fs  if 
                            any([x.endswith(ext) 
                                 for ext in ['tiff', 'jpeg']])]))


def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]


def load_image_uint_one(filename):
    img = np.array(Image.open(filename), dtype=np.uint8)
    if len(img.shape) == 3:
        img = img.transpose(2, 1, 0)
    else:
        img = np.array([img])

    #black = np.sum(img, axis=0) < np.mean(img)
    #for c in [0, 1, 2]:
    #    ch = img[c]
    #    ch[black] = np.mean(ch[~black])
    return img


def load_image_uint(filename):
    if not hasattr(filename, '__iter__'):
        return load_image_uint_one(filename)
    else:
        return [load_image_uint_one(f) for f in filename]


def load_image(filename):
    return np.array(load_image_uint(filename), dtype=np.float32)


def load_normalized(files):
    # hack so we don't need to pass the arguments for mean and std
    mean = get_mean()
    X = load_image(files) - mean
    X /= np.array(STD, dtype=np.float32)[np.newaxis, :, np.newaxis,
                                         np.newaxis]
    return X


def normalize(batch):
    mean = get_mean().mean(axis=(1, 2))
    X = np.array(batch, np.float32) - np.array(mean, dtype=np.float32)[
            np.newaxis, :, np.newaxis, np.newaxis]
    X /= np.array(STD, dtype=np.float32)[np.newaxis, :, np.newaxis,
                                         np.newaxis]
    return X


def load_patient(name, left=True, right=True, path=TRAIN_DIR):
    files = get_filenames(name, path)
    return merge_left_right(*load_image(files))


def get_filename(name, left=True, path=TRAIN_DIR):
    return '{}/{}_{}.tiff'.format(path, name, 'left' if left else 'right')


def get_filenames(name, path=TRAIN_DIR):
    return [get_filename(name, left, path) for left in [True, False]]


def merge_left_right(l, r):
    return np.concatenate([l, r], axis=1)


def get_submission_filename():
    sha = get_commit_sha()
    return "{}_{}_{}.csv".format(SUBMISSION, sha,
                                 datetime.now().replace(microsecond=0))


def get_commit_sha():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                         stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.strip().decode('utf-8')


def balance_shuffle_indices(y, random_state=None, weight=BALANCE_WEIGHT):
    y = np.asarray(y)
    counter = Counter(y)
    max_count = np.max(counter.values())
    indices = []
    for cls, count in counter.items():
        ratio = weight * max_count / count + (1 - weight)
        idx = np.tile(np.where(y == cls)[0], 
                      np.ceil(ratio).astype(int))
        np.random.shuffle(idx)
        indices.append(idx[:max_count])
    return shuffle(np.hstack(indices), random_state=random_state)


def split_indices(y, test_size=0.1, random_state=RANDOM_STATE):
    spl = cross_validation.StratifiedShuffleSplit(y, test_size=test_size, 
                                                  random_state=random_state,
                                                  n_iter=1)
    return next(iter(spl))

def split(X, y, test_size=0.1, random_state=RANDOM_STATE):
    train, test = split_indices(y, test_size, random_state)
    return X[train], X[test], y[train], y[test]


def kappa(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = y_true.dot(range(y_true.shape[1]))
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.dot(range(y_pred.shape[1]))
    try:
        return quadratic_weighted_kappa(y_true, y_pred)
    except IndexError:
        return np.nan


def kappa_from_proba(w, p, y_true):
    return kappa(y_true, p.dot(w))


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').strip('.py'))


def get_mask(y):
    counts = Counter(y)
    mask = np.zeros(y.shape, dtype=np.float32)
    for k, v in counts.items():
        mask[y == k] = 1.0 / v
    mask /= np.mean(mask)
    return mask

def grid_search_score_dataframe(gs):
    dicts = [s._asdict() for s in gs.grid_scores_]
    df = pd.DataFrame([d['parameters'] for d in dicts])
    scores = np.array([d['cv_validation_scores'] for d in dicts])
    df.loc[:, 'mean'] = scores.mean(axis=1)
    df.loc[:, 'std'] = scores.std(axis=1)
    return df.iloc[np.argsort(df['mean'])]
