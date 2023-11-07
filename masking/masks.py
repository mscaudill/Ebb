"""This module contains functions for building 1-D boolean masks. These masks
may be built from human annotated text files, spindle state text files or
computationally computed using a local z-score threshold.

Functions:
    threshold:
        Returns a 1-D boolean by z-score thresholding produced values from
        a producer.
    artifact:
        Returns a 1-D boolean by reading human created annotations stored in
        a Pinnacle annotations file.
    state:
        Returns a 1-D boolean by reading Spindle detected sleep and wake states
        from a Spindle formatted file.

"""

import csv
import functools
from itertools import zip_longest
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from openseize.core.producer import producer, Producer
from openseize.file_io import annotations
from ebb.core import arraytools


def threshold(pro: Producer,
              nstds: List[float],
              winsize: int,
              radius: Optional[int] = None,
) -> List[npt.NDArray[np.bool_]]:
    """Thresholds the z-score of produced values using each standard deviation
    in nstds.

    Args:
        pro:
            A producer of ndarrays to be thresholded.
        nstds:
            A list of standard deviations to threshold normalized producer
            values.
        winsize:
            The number of samples used to estimate the mean and standard
            deviation to normalize the produced values by.
        radius:
            The max distance below which all intermediate samples between two
            samples are interpolated to be False.

    Examples:
        >>> from openseize import producer
        >>> import numpy as np
        >>> # make a random array with 50 spikes in each row
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(loc=0, scale=1.0, size=(4,1000))
        >>> locs = rng.choice(np.arange(1000), size=(4,50), replace=False)
        >>> for row, loc_ls in enumerate(locs):
        ...     x[row, loc_ls] = 10
        >>> # make a producer from spiked data and build masks
        >>> pro = producer(x, chunksize=100, axis=-1)
        >>> masks = threshold(pro, nstds=[2], winsize=100)
        >>> mask = masks[0]
        >>> # are the mask indices that are False equal to the locs provided?
        >>> set(np.where(~mask)[0]) == set(locs.flatten())
        True

    Returns:
        A list of 1-D boolean arrays one per standard dev. in nstds.
    """

    pro.chunksize = winsize
    axis = pro.axis

    # initialize some mask
    masks = [np.ones(pro.shape[axis], dtype=bool) for std in nstds]
    for idx, arr in enumerate(pro):

        mu = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)

        if not np.any(std):
            std = np.ones_like(std)

        arr -= mu
        arr /= std

        for sigma, mask in zip(nstds, masks):
            _, cols = np.where(np.abs(arr) > sigma)
            cols = np.unique(cols + idx * pro.chunksize)
            mask[cols] = False

    # Fill between 2 False events with False if sample distance < radius
    if radius:
        for mask in masks:
            for epoch in arraytools.aggregate1d(~mask, radius):
                mask[slice(*epoch)] = False

    return masks


def artifact(path: Union[str, Path],
             size: int,
             labels: List[str],
             fs: float,
             between: Tuple[Union[str, None], Union[str, None]]=(None, None),
             **kwargs,
) -> npt.NDArray[np.bool_]:
    """Returns a boolean mask from a Pinnacle annotations file.

    Args:
        path:
            Pinnacle file path containing annotations.
        size:
            The length of the mask to return.
        labels:
            The artifact labels that are to be marked False in the returned
            boolean.
        fs:
            The sampling rate in Hertz of the data acquisition.
        between:
            The start and stop annotation labels between which labeled
            annotations will be included in the mask.
        kwargs:
            Keyword arguments are passed to openseize's Pinnacle initializer.

    Examples:
        >>> from openseize import demos
        >>> import numpy as np
        >>> path = demos.paths.locate('annotations_001.txt')
        >>> fs = 5000
        >>> size = 3775 * 5000
        >>> amask = artifact(path, size=size, labels=['rest'], fs=fs)
        >>> # validate 1st set of Falses in mask occurs at 1st annote
        >>> with annotations.Pinnacle(path, start=6) as reader:
        ...     annotes = reader.read('rest')
        >>> start, duration = annotes[0].time, annotes[0].duration
        >>> np.any(amask[int(start*5000): int((start+duration) * 5000)])
        False

    Returns:
        A 1-D boolean mask in which labeled indices are marked False.
    """

    start = kwargs.pop('start', 6)
    with annotations.Pinnacle(path, start=start, **kwargs) as reader:
        annotes = reader.read()

    # get first and last annote to return that are in between
    a, b = between
    first = next(ann for ann in annotes if ann.label == a) if a else annotes[0]
    last = next(ann for ann in annotes if ann.label == b) if a else annotes[-1]

    #filter the annotes for requested labels and filter between
    annotes = [ann for ann in annotes if ann.label in labels]
    annotes = [ann for ann in annotes if first.time <= ann.time <= last.time]
    # adjust the annote times relative to first annote
    for annote in annotes:
        annote.time -= first.time

    return annotations.as_mask(annotes, size, fs, include=False)


def state(path, labels, fs, winsize, include=True):
    """Returns a boolean mask from a spindle sleep score text file.

    Args:
        path:
            A path to a spindle file.
        labels:
            A list of labels to include or exclude depending on include
            argument.
        fs:
            The sampling rate of the data this mask will be applied.
        winsize:
            The window length in seconds that spindle used to estimate states.
        include:
            A boolean indicating if labels should be included (i.e. True) and
            all else False or labels should be excluded (i.e. False) from the
            returned mask.

    Examples:
    >>> states = 'w w w w r n n r n w'.split()
    >>> import tempfile
    >>> base_path = tempfile.mkdtemp(prefix='test_')
    >>> fp = Path(base_path).joinpath('test_state.csv')
    >>> with open(fp, 'w') as outfile:
    ...     writer = csv.writer(outfile)
    ...     writer.writerows(list(enumerate(states)))
    >>> mask = state(fp, labels=['w'], fs=5, winsize=2)
    >>> probe = np.zeros(100, dtype=bool)
    >>> probe[0:40] = True
    >>> probe[-10:] = True
    >>> np.allclose(mask, probe)
    True

    Returns:
        A 1-D boolean array for masking data at a given sample rate.
    """

    with open(path, 'r') as infile:
        reader = csv.reader(infile)
        states = [row[1] for row in reader]

    mask = np.array([state in labels for state in states], dtype=bool)
    if not include:
        mask = ~mask

    mask = np.atleast_2d(mask)
    result = np.repeat(mask, fs * winsize, axis=0)
    return result.flatten(order='F')

# DEPRECATION NOTICE
# The next release of openseize will handle producing between values along axis
# so the following will be removed when openseize v1.3.0 is released.
def _between_gen(reader, start, stop, chunksize, axis):
    """A generating function returns a generator of ndarrays of samples between
    start and stop.

    Args:
        reader:
            An openseize reader instance.
        start:
            The index at which production of values from reader begins.
        stop:
            The index at which production of values from reader ends.
        chunksize:
            The number of samples to return along axis of each yield ndarray.
        axis:
            The axis along which samples will be produced.
    
    Returns:
        A generator of ndarrays of chunksize shape along axis between start and
        stop.

    Notes:
        This is a protected module-level function that is not intended for
        external calling. Its placement at module level versus nesting is to
        support concurrent processing.
    """

    starts = np.arange(start, stop, int(chunksize))
    for a, b in zip_longest(starts, starts[1:], fillvalue=stop):
        yield reader.read(a,b)

def between_pro(reader, start, stop, chunksize, axis=-1):
    """Returns a producer from a reader instance that produces values between
    start and stop.

    Args:
        reader:
            An openseize reader instance.
        start:
            The index at which production of values from reader begins.
        stop:
            The index at which production of values from reader ends.
        axis:
            The axis along which production should occur. Defaults to last axis.

    Examples:
        >>> from openseize.demos import paths
        >>> from openseize.file_io import edf
        >>> path = paths.locate('recording_001.edf')
        >>> reader = edf.Reader(path)
        >>> reader.channels = [0,1,2]
        >>> # read samples 10 to 1155 in chunks of 100
        >>> pro = between_pro(reader, 10, 1155, chunksize=100)
        >>> np.allclose(pro.to_array(), reader.read(10, 1155))
        True

    Returns:
        A producer instance producing samples between start and stop along axis.
    """

    # build a partial freezing all arg of _between_gen
    gen_func = functools.partial(_between_gen, reader, start, stop, chunksize, 
                                 axis)
    # compute the shape of the new producer
    shape = list(reader.shape)
    shape[axis] = stop - start

    return producer(gen_func, chunksize, axis, shape=shape)

