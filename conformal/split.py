import numpy as np
import random


def standard_split(x, y, p=None, seed=None):
    """
        Perform a standard p/(1-p) split between calibration and validation sets stratified on the classes.
        arguments:
            x [numpy.array]: extracted predictions
            y [numpy.array]: corresponding labels
            p: calibration data proportion
        returns:
            x_calib [numpy.array]: predictions of the calibration set
            y_calib [numpy.array]: calibration set labels
            x_val [numpy.array]: predictions of the validation set
            y_val [numpy.array]: calibration set labels

    """
    calib_idx, val_idx = [], []
    for val in np.unique(y):
        idx = np.where(y == val)[0]
        split_value = np.max([1, int(len(idx)*p)])
        if seed is not None:  # Reproducibility
            np.random.seed(seed)
        np.random.shuffle(idx, )
        calib_idx.extend(idx[:split_value])
        val_idx.extend(idx[split_value:])
    assert set(calib_idx).intersection(set(val_idx)) == set(), 'Overlapping indices.'

    x_calib, y_calib = x[calib_idx], y[calib_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    return x_calib, y_calib, x_val, y_val


def balance_split(x, y, k=16, p=None, seed=None):

    # Labels as integeets
    y = np.int8(y)

    # Total number of samples
    N = len(np.unique(y)) * k

    calib_idx, val_idx = [], []
    for val in list(np.unique(y)):
        idx = np.where(y == val)[0]
        split_value = np.max([1, round(N*p[val])])  # + random.sample([0, 1], k=1)[0]
        if seed is not None:  # Reproducibility
            np.random.seed(seed)
        np.random.shuffle(idx, )
        calib_idx.extend(idx[:split_value])

    x_calib, y_calib = x[calib_idx], y[calib_idx]

    return x_calib, y_calib
