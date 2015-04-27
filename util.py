from datetime import datetime
try:
    import cPickle as pickle
except ImportError: 
    import pickle
import os
import subprocess

import numpy as np
import pandas as pd

from PIL import Image

from definitions import *

def compute_mean(files, batch_size=BATCH_SIZE):
    """Load images in files in batches and compute mean."""
    m = np.zeros([C, W, H])
    for i in range(0, len(files), batch_size):
        images = load_images(files[i : i + batch_size])
        m += images.sum(axis=0)
    return (m / len(files)).astype(np.float32)


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


def get_labels(names):
    return np.array(pd.read_csv(LABEL_FILE, index_col=0).loc[names]).flatten()


def get_image_files(datadir):
    fs = [os.path.join(dp, f) for dp, dn, fn in os.walk(datadir) for f in fn]
    return [x for x in fs if x.endswith('.tiff')]


def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]


def load_images(files):
    images = np.array([np.array(Image.open(f)).transpose(2, 1, 0)
                       for f in files])
    return images


def get_submission_filename():
    sha = get_commit_sha()
    return "{}_{}_{}.csv".format(SUBMISSION, sha,
                                 datetime.now().replace(microsecond=0))


def get_commit_sha():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], 
                         stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.strip().decode('utf-8')

