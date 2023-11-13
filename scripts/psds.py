"""A script for computing and storing EEG Metrics and/or Biomarkers.

"""

import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy.typing as npt
from ebb.core import concurrency, metastores
from ebb.masking import masks
from ebb.scripts.file_combine import pair
from openseize.file_io import edf
from openseize import producer
from openseize.spectra import estimators


def estimate(
    path: Union[str, Path],
    state_path: Union[str, Path],
    channels: List = [0,1,2],
    fs: float = 200,
    chunksize: int = 30e5,
    labels: Dict = {'wake': ['w'], 'sleep': ['n', 'r']},
    winsize: float = 180,
    nstds: float = 5,
    radius: float = 0.5,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Tuple[int, npt.NDArray, npt.NDArray]]:
    """Estimates the Power spectral density of EEG data stored to an EDF at path
    for each state in a states path CSV file.

    Args:
        path:
            Path to an EDF file whose PSD will be estimated.
        state_path:
            Path to a Spindle generated CSV file of states. Please note the
            allowable states are 'w'-wake, 'r'-rem and 'n'-non-rem only. This
            means the artifact slider in the Spindle web service must be set to
            0. If state_path is None, the PSDs ar
        channels:
            The channels for which PSDs will be estimated.
        fs:
            The sampling rate of the data in the EDF.
        labels:
            A dict of label names and spindle codes one per state for which PSDs
            will be computed.
        winsize:
            The size of the window for threshold artifact detection using the
            threshold mask. This value has units of seconds. Please see
            ebb.masking.threshold
        nstds:
            The number of standard deviations a deflection must cross to be
            considered an artifact. Please see ebb.masking.threshold
        radius:
            The number of secs below which two detected artifacts will be merged
            into a single artifact. Please see ebb.masking.threshold.
        verbose:
            Boolean indicating if information about which PSD is currently being
            computed should print to stdout.
        **kwargs:
            All kwargs are passed to openseize's psd estimator function.

    Returns:
        A list of Tuples one per state in labels and each containing (count,
        freqs, psd) where count is the number of estimatives (i.e. averaged
        windows); freqs are the frequencies at which the PSD is estimated; psd
        are the psd values array with shape chs x freqs.
    """

    with edf.Reader(path) as reader:

        # build a producer
        reader.channels = channels
        pro = producer(reader, chunksize=chunksize, axis=-1)

        # build state masks
        named_masks = {}
        for label, code in labels.items():
            named_masks[label] = masks.state(state_path, code, fs, winsize=4)

        # build artifact threshold mask and metamask to hold all masks
        wsize, rad = int(winsize * fs), int(radius * fs)
        threshold = masks.threshold(pro, [nstds], wsize, rad)[0]
        metamask = metastores.MetaMask(threshold=threshold, **named_masks)

        print(metamask)

        # compute psd estimates for each mask combination
        result = {}
        for name, mask in metamask.combinations():
            if 'threshold' in name:
                if verbose:
                    print(f'Computing PSD for state: {name}', end='\r')
                maskedpro = producer(pro, chunksize=chunksize, axis=-1, mask=mask)
                result[name] = estimators.psd(maskedpro, fs=fs, **kwargs)
                # clear stdout line likely POSIX specific
                print(end='\x1b[2K')

    return result

def batch(eeg_dir, state_dir, save_dir, pattern=r'[^_]+', **kwargs):
    """ """

    # remove any path arguments from kwargs
    [kwargs.pop(x, None) for x in ('path', 'state_path')]

    # pair the eeg paths with the state paths
    eeg_paths = list(Path(eeg_dir).glob('*PROCESSED.edf'))
    state_paths = list(Path(state_dir).glob('*.csv'))
    paired = pair(eeg_paths + state_paths, pattern=pattern)

    workers = concurrency.set_cores(ncores, len(paths))
    if verbose:
        msg = (f'Executing Batched PSD on {len(eeg_paths)} files using'
               f'{workers} cores.')
        print(msg)

    # construct a partial to pass to Multiprocess pool
    estimator = functoolss.partial(estimate, **kwargs)
    t0 = time.perf_counter()
    with Pool(workers) as pool:
       results = pool.starmap(estimator, paired)

    elapsed = time.perf_counter() - t0
    print(elapsed)
    return results



if __name__ == '__main__':

    """
    fp = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
          'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30_PREPROCESSED.edf')

    state_path = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/'
                  'CW0DA1_P096_KO_15_53_3dayEEG_2020' + \
                  '-04-13_08_58_30_PREPROCESSED_SPINDLE_labels.csv')

    results = estimate(fp, state_path, verbose=True)
    """

    eeg_dir = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
    state_dir = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/states/'
    paired = batch(eeg_dir, state_dir, None)

