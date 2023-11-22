"""A script for computing EEG PSDs stored as MetaArrays with shape
(spindle_state x file x channels x frequencies).
"""

import functools
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from ebb.core import concurrency, metastores
from ebb.masking import masks
from ebb.scripts.file_combine import pair
from openseize import producer
from openseize.file_io import edf
from openseize.spectra import estimators


@dataclass
class PSDResult:
    """A class for keeping track of a PSD and all metadata."""

    path: Union[str, Path]
    state: Tuple[str, ...]
    estimatives: int
    freqs: npt.NDArray
    psd: npt.NDArray


def estimate(
    path: Union[str, Path],
    state_path: Union[str, Path],
    channels: List = [0, 1, 2],
    fs: float = 200,
    chunksize: int = int(30e5),
    labels: Dict = {'wake': ['w'], 'nrem': ['n'], 'rem': ['r'], 'sleep': ['n', 'r']},
    winsize: float = 180,
    nstds: float = 5,
    radius: float = 0.5,
    verbose: bool = True,
    **kwargs,
) -> List[PSDResult]:
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
        A list of PSDResult instances one per file and state in paths and
        labels.
    """

    path, state_path = Path(path), Path(state_path)
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

def batch(
    eeg_dir: Union[str, Path],
    state_dir: Union[str, Path],
    pattern=r'[^_]+',
    ncores: Optional[int] = None,
    **kwargs,
) -> List[PSDResult]:
    """Concurrently estimates PSDResults from a directories of EEG files and
    Spindle state files.

    Args:
        eeg_dir:
            Directory to EEG files to estimate PSDs of.
        state_dir:
            Directory of spindle files one per EEG file in eeg_dir indicating
            sleep states.
        pattern:
            A regex string for pairing EEG files with Spindle state files.
        ncores:
            The number of processing cores to utilize to concurrently process
            EEG files.
        kwargs:
            Any valid kwarg for the estimate function.

    Returns:
        A list of lists of PSDResults, one PSD result for each file and state.
    """

    # remove any path arguments from kwargs
    [kwargs.pop(x, None) for x in ('path', 'state_path')]

    # pair the eeg paths with the state paths
    eeg_paths = list(Path(eeg_dir).glob('*PROCESSED.edf'))
    state_paths = list(Path(state_dir).glob('*.csv'))
    paired = pair(eeg_paths + state_paths, pattern=pattern)

    # set number of cpu workers and print to stdout
    workers = concurrency.set_cores(ncores, len(paired))
    msg = f'Executing Batch on {len(paired)} files using' f' {workers} cores.'
    print(msg)

    # construct a partial to pass to Multiprocess pool
    estimator = functools.partial(estimate, verbose=False, **kwargs)
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        results = pool.starmap(estimator, paired)
    elapsed = time.perf_counter() - t0
    print(f'Batch processed {len(paired)} files in {elapsed} secs')

    # flatten the list of list of PSDResult data instances
    return [obj for sublist in results for obj in sublist]

def as_metaarray(
    results: List[PSDResult], savedir: Optional[Union[str, Path]] = None
) -> metastores.MetaArray:
    """Creates and saves a MetaArray of PSDs from batch estimate.

    Args:
        results:
            A list of PSDResult instances returned by batch.
        savedir:
            A dir where the MetAarray called PSDs.pkl will be saved to. If None,
            the metaarray will not be saved.

    Returns:
        A MetaArray instance.
    """

    # organize PSD results by state
    states = set([result.state for result in results])
    grouped = []
    for state in states:
        grouped.append([obj for obj in results if obj.state == state])

    # extract and stack psds for each state in grouped
    datums = [np.stack([obj.psd for obj in group]) for group in grouped]
    data = np.stack(datums)
    # extract paths from first group -- each group preserves this order
    paths = [obj.path for obj in grouped[0]]

    # get the estimatives
    estimatives = [obj.estimatives for group in grouped for obj in group]

    # build and save a metaarray
    metaarray = metastores.MetaArray(
        data,
        metadata={'estimatives': estimatives},
        states=states,
        paths=paths,
        channels=range(data.shape[2]),
        frequencies=results[0].freqs,
    )

    if savedir:
        savepath = Path(savedir).joinpath('PSDs.pkl')
        metaarray.save(savepath)

    return metaarray


if __name__ == '__main__':
    
    fp = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
          'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30_PREPROCESSED.edf')

    state_path = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/states/'
                  'CW0DA1_P096_KO_15_53_3dayEEG_2020' + \
                  '-04-13_08_58_30_PREPROCESSED_SPINDLE_labels.csv')

    results = estimate(fp, state_path, verbose=True)
    

    """
    eeg_dir = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
    state_dir = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/states/'
    """

    """
    eeg_dir = '/media/matt/Zeus/claudia/test/standard/'
    state_dir = '/media/matt/Zeus/claudia/test/spindle/'
    results = batch(eeg_dir, state_dir)
    marray = as_metaarray(results)
    """
