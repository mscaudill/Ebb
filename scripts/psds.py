import itertools
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

from openseize import producer
from openseize.file_io import edf, path_utils
from openseize.filtering import iir
from openseize.resampling import resampling
from openseize.spectra import estimators

from spectraprints.masking import masks
from spectraprints.core import concurrency
from spectraprints.core.metastores import MetaArray, MetaMask

# GLOBALS
# Production and preprocessing Args
CHS = [0, 1, 2]
FS = 5000       
M = 20
DFS = FS // M
CSIZE = 30E5
AXIS = -1
# TODO Cap stop value in preprocess at 48 or 72 hrs depend on data recording
START, STOP = None, None

# Thresholding Args
NSTDS = [5]
WINSIZE = 1.5E4 # @ FS = 250 THIS IS 60 SECS OF DATA
RADIUS = 125 # IN SAMPLES

# State Args
STATE_LABELS = [['w'], ['r', 'n']]
STATE_WINSIZE = 4 # Spindle window size


def preprocess(epath, channels, fs, M, start, stop, chunksize, axis):
    """Preprocesses an EEG file at path by constructing, notch filtering and
    downsampling a producer.

    Args:
        epath:
            Path to an edf file.
        channels:
            The channels to include in the processed producer.
        fs:
            The sampling rate at which the data at path was collected.
        M:
            The downsample factor to reduce the number of samples produced by
            the returned processed producer.
        start:
            The start index at which production begins. If None start is 0.
        stop:
            The stop index at which production stops. If None stop is data
            length along axis.
        chunksize:
            The number of processed samples along axis to yield from the 
            producer at a time.
        axis:
            The axis of reading and production.

    Returns:
        A producer of preprocessed values.
    """

    reader = edf.Reader(epath)
    reader.channels = channels

    start = 0 if not start else start
    stop = reader.shape[axis] if not stop else stop
    if stop - start < reader.shape[axis]:
        pro = masks.between_pro(reader, start, stop, chunksize, axis)
    else:
        pro = producer(reader, chunksize, axis)

    # Notch filter the producer
    notch = iir.Notch(fstop=60, width=6, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)

    # downsample the producer
    result = resampling.downsample(result, M, fs, chunksize, axis)
    return result


def make_metamask(epath, spath, verbose=False):
    """Returns a MetaMask instance containing thresholded masks, annotations
    mask, and Spindle sleep score masks.

    The specific parameters used to make the thresholded and spindle mask can be
    found in the GLOBALS section at the top of this script.
    """

    t0 = time.perf_counter()
   
    # preprocess a producer for between START and STOP
    pro = preprocess(epath, channels=CHS, fs=FS, M=M, start=START, stop=STOP,
                     chunksize=CSIZE, axis=AXIS)

    # build threshold masks
    thresholded = masks.threshold(pro, nstds=NSTDS, winsize=WINSIZE, 
                                  radius=RADIUS)
    thresholded_names = [f'threshold={std}' for std in NSTDS]
    thresholded = dict(zip(thresholded_names, thresholded))
    
    # build state masks
    stated = [masks.state(spath, ls, DFS, STATE_WINSIZE) for ls in STATE_LABELS]
    state_names = ['awake', 'sleep']
    stated = dict(zip(state_names, stated))

    if verbose:
        name = Path(epath).stem
        print(f' Completed building MetaMask for {name}'
               ' in {time.perf_counter() - t0} s')

    return MetaMask(**thresholded, **stated)


def process_file(epath, spath, verbose=False):
    """ """

    # there is some real ineffeciency here because we preprocess to get
    # threshold masks and then preprocess again to get psd estimates. Look to
    # see if this is avoidable
    #
    # We won't be able to really see how long this function takes until we have
    # the full spindle for 72 hours

    t0 = time.perf_counter()

    metamask = make_metamask(epath, spath)
    

    mask_names = metamask.names
    threshold_names = [name for name in mask_names if 'threshold' in name]
    state_names = [name for name in mask_names if name in ['awake', 'sleep']]
    
    result = {}
    for state, threshold in itertools.product(state_names, threshold_names):
        
        # build the combined mask
        combo_name, mask = metamask(state, threshold)

        # preprocess the producer and mask it
        pro = preprocess(epath, CHS, FS, M, START, STOP, CSIZE, AXIS)
        maskpro = producer(pro, CSIZE, AXIS, mask=mask)

        # Estimate the PSD
        cnt, freqs, psd = estimators.psd(maskpro, fs=DFS)
        result[combo_name] = (cnt, freqs, psd)

    print(f'PSDs for {epath.stem} computed in {time.perf_counter()-t0} secs.')

    # FIXME what is a unique name where animals recorded from and then treated?
    return Path(epath).stem, result


if __name__ == '__main__':

    basepath = Path('/media/matt/Zeus/jasmine/stxbp1')

    efile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.edf'
    sfile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11_sleep_states.csv'
    epath = basepath.joinpath(efile)
    spath = basepath.joinpath(sfile)

    result = process_file(epath, spath)


