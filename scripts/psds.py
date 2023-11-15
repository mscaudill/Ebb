"""A script for computing and storing EEG Metrics and/or Biomarkers.

"""

import functools
import itertools
import time
from dataclasses import dataclass
from operator import attrgetter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from ebb.core import concurrency, metastores
from ebb.masking import masks
from ebb.scripts.file_combine import pair
from openseize.file_io import edf
from openseize import producer
from openseize.spectra import estimators

@dataclass
class PSDResult:
    """A class for keeping track of a PSD and all metadata."""

    path: Union[str, Path]
    state: Tuple[str,...]
    estimatives: int
    freqs: npt.NDArray
    psd: npt.NDArray

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

        # compute psd estimates for each mask combination
        result = []
        for combo_state, mask in metamask.combinations():
            if 'threshold' in combo_state:
                if verbose:
                    print(f'Computing PSD for state: {combo_state}', end='\r')
                maskedpro = producer(pro, chunksize, axis=-1, mask=mask)
                result_tup = estimators.psd(maskedpro, fs=fs, **kwargs)
                # clear stdout line likely POSIX specific
                print(end='\x1b[2K')
                result.append(PSDResult(path.stem, combo_state, *result_tup))

    return result

def batch(eeg_dir, state_dir, save_dir, pattern=r'[^_]+', ncores=12,
        verbose=True, **kwargs):
    """ """

    # remove any path arguments from kwargs
    [kwargs.pop(x, None) for x in ('path', 'state_path')]

    # pair the eeg paths with the state paths
    eeg_paths = list(Path(eeg_dir).glob('*PROCESSED.edf'))
    state_paths = list(Path(state_dir).glob('*.csv'))
    paired = pair(eeg_paths + state_paths, pattern=pattern)

    workers = concurrency.set_cores(ncores, len(paired))
    if verbose:
        msg = (f'Executing Batched PSD on {len(eeg_paths)} files using'
               f' {workers} cores.')
        print(msg)

    # construct a partial to pass to Multiprocess pool
    estimator = functools.partial(estimate, verbose=False, **kwargs)
    t0 = time.perf_counter()
    with Pool(workers) as pool:
       results = pool.starmap(estimator, paired)
    elapsed = time.perf_counter() - t0

    # flatten the list of list
    results = list(itertools.chain(*results))
    # organize PSD results by state
    unique_states = set([result.state for result in results])
    state_results = []
    for state in unique_states:
        state_results.append([r for r in results if r.state == state])

    psds = []
    for subls in state_results:
        psds.append(np.stack([r.psd for r in subls]))
    psds = np.stack(psds, axis=0)

    chs = range(psds.shape[2])
    freqs = state_results[0][0].freqs
    metaarray = metastores.MetaArray(
                    psds,
                    states = unique_states,
                    paths = eeg_paths,
                    channels = chs,
                    frequencies=freqs)

    save_path = Path(save_dir).joinpath('metaarray.pkl')
    metaarray.save(save_path)

    return metaarray





if __name__ == '__main__':

    """
    fp = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
          'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30_PREPROCESSED.edf')

    state_path = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/'
                  'CW0DA1_P096_KO_15_53_3dayEEG_2020' + \
                  '-04-13_08_58_30_PREPROCESSED_SPINDLE_labels.csv')

    results = estimate(fp, state_path, verbose=True)
    """

    """
    eeg_dir = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
    state_dir = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/states/'
    """

    eeg_dir = '/media/matt/Zeus/claudia/test/standard/'
    state_dir = '/media/matt/Zeus/claudia/test/spindle/'
    results = batch(eeg_dir, state_dir, None)

