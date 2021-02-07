import numpy as np


def _flatten(values):
    if isinstance(values, np.ndarray):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)


def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))


pos_filename = np.load('pic_data/neg_filename.npy', allow_pickle=True)
test_filename = flatten(pos_filename)
np.save('pic_data/neg_filename.npy', test_filename)
