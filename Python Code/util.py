def _flatten(values):
    if isinstance(values, np.ndarray):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)

def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))

test_filename = flatten(test_filename)
np.save('pic_data/test_filename.npy',test_filename)
