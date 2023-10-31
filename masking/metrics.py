"""A module for evaluating masks computed from models with human annotated
masks.
"""

import numpy as np

from spectraprints.core import arraytools

def _tp(mask, actual):
    """Returns the number of True positives between two 1-D arrays of binary
    classifications.

    Args:
        mask:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean of actual "ground-truth" classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], dtpe=bool)
        >>> _tp(arr, actual)
        3

    Returns:
        An integer count of true positive classifications.
    """

    x = set(np.flatnonzero(mask))
    y = set(np.flatnonzero(actual))
    return len(x.intersection(y))

def _tn(mask, actual):
    """Returns the number of True Negatives between two 1-D arrays of binary
    classifications.

    Args:
        mask:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean of actual "ground-truth" classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], dtpe=bool)
        >>> _tn(arr, actual)
        1

    Returns:
        An integer count of true negative classifications.
    """

    x = set(np.flatnonzero(~mask))
    y = set(np.flatnonzero(~actual))
    return len(x.intersection(y))

def _fp(mask, actual):
    """Returns the number of False Positives between two 1-D arrays of binary
    classifications.

    Args:
        mask:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean of actual "ground-truth" classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], dtpe=bool)
        >>> _fp(arr, actual)
        1

    Returns:
        An integer count of False positive classifications.
    """

    x = mask.astype(int)
    y = actual.astype(int)
    return len(np.where((x-y) > 0)[0])

def _fn(mask, actual):
    """Returns the number of False Negatives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _fn(arr, actual)
        1

    Returns:
        An integer count of false negative classifications.
    """

    x = mask.astype(int)
    y = actual.astype(int)
    return len(np.where((y- x) > 0)[0])

def accuracy(tp, tn, fp, fn):
    """Returns the accuracy between two 1-D boolean classification arrays.

    The accuracy is the number of correct classifications divided by all 
    classifications (TP + TN) / (TP + TN + FP + FN).

    Returns:
        A float accuracy value.
    """

    return (tp + tn) / (tp + tn + fp + fn)

def sensitivity(tp, fn):
    """Returns the sensitivity between two 1-D boolean classification arrays.

    The sensitivity is "how many did I catch out of all the ones I want to find"

    Returns:
        A float sensitivity value.
    """

    return tp / (tp + fn)

def specificity(tn, fp):
    """Returns the specificity between two 1-D boolean classification arrays.

    The sensitivity is "how many did I ignore out of all the ones I want to
    ignore"

    Returns:
        A float sensitivity value.
    """

    return tn / (tn + fp)

def precision(tp, fp):
    """Returns the precision between two 1-D boolean classification arrays.

    The precision is "how many did I correctly catch out of all the ones caught."

    Returns:
        A float sensitivity value.
    """

    return tp / (tp + fp)

def percent_within(mask, actual):
    """Counts the number of True values in mask that are within True runs in
    actual.

    Args:
        mask:
            A 1-D boolean of estimated mask values from thresholding.
        actual:
            A 1-D boolean obtained from a human annotations file.
            
    Returns:
        The count of mask that are between consecutive True values in actual.
    """

    events = arraytools.discretize_bool(~actual).flatten()
    detected = np.flatnonzero(~mask)

    # locate where detected would be inserted into events
    insertions = np.searchsorted(events, detected, 'right')
    odds = [x for x in insertions if x % 2 == 1]

    return len(odds) / len(detected)
