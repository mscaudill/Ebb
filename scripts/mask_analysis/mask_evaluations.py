import pickle
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from openseize import producer
from openseize.file_io import edf, annotations, path_utils
from openseize.filtering import iir
from openseize.resampling import resampling

from spectraprints.masking import masks, metrics
from spectraprints.core import concurrency
from spectraprints.core.metastores import MetaMask


def preprocess(epath, channels, fs, M, start, stop, chunksize=30e5, axis=-1):
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

def make_metamask(epath, apath, spath, verbose=False):
    """Returns a MetaMask instance containing thresholded masks, annotations
    mask, and Spindle sleep score masks.

    Fixed Args:
        - produced data
            channels:
                The channels to analyze = [0, 1, 2]
            start:
                The start index of production. This should match the start label
                of the annotations and align with the first detected state in
                the spindle file. This index should be obtained by reading the
                start annotation and converting from time to samples.
            stop:
                The stop index of production. This shuld match the stop label of
                the annotations file and align with the last detected state in
                the spindle file. This index should be obtained by reading the
                stop annotation and converting from time to samples.
            fs: 
                The sampling rate of the data = 5000
            M:
                The downsampling ratio = 20
            chunksize:
                The size of produced data from preprocessor = 30e5
            axis:
                The production axis = -1
        - threshold masks
            nstds:
                The standard deviations to use as the threshold. One mask per
                threshold will be stored to MetaMask instance = [3, 4, 5, 6]
            winsize:
                The window size in samples to evaluate the mean & standard
                deviation for thresholding = 1.
            radius:
                The max distance between two detected events below which all
                intervening samples will be marked as artifact.
        - annotated masks
            size:
                The size of the of masks - this must equal the producers shape
                along the production axis.
            labels:
                The labels that denote artifacts = ['Artifact', 'Artifact ',
                'water_drinking', 'water_drinking ']
            fs:
                The downsampled sampling rate, must match fs // M = 250
            between:
                The string names of the start and stop annotation. These should
                correspond to the start and stop indices provided to producer.
        - state masks:
            labels:
                A list of list of string labels to include in each mask, one 
                sublist per state = [['w'], ['r', 'n']] gives the awake and
                sleep states.
            fs:
                The downsampled sampling rate, must match fs // M = 250
            winsize:
                The interval in secs over which spindle evaluated state
                = 4 secs.
    """

    t0 = time.perf_counter()
   
    # production Args
    CHS = [0, 1, 2]
    FS = 5000
    M = 25
    CSIZE = 30E5 # LOW ENOUGH TO ALLOW MULTIPROCESSING TO COPY
    AXIS = -1

    # thresholding Args
    NSTDS = [3, 4, 5, 6]
    #WINSIZE = 1.5E4 # @ FS = 250 THIS IS 60 SECS OF DATA
    #WINSIZE = 1.5E5 # @ FS = 250 THIS IS 600 SECS OF DATA
    WINSIZE = 3.6E4 # @FS = 200 THIS IS 180 SECS
    #WINSIZE = 1.2E5 #@FS = 200 this is 600 secs of data 
    RADIUS = 100 # THIS 1/2 sec at FS=200

    # annotation Args
    # TODO - need more flexibility in artifact mask like regex
    LABELS = ['Artifact', 'Artifact ','water_drinking','water_drinking '
              '']
    DFS = FS // M
    BETWEEN = ['Heet Start', 'Heet Stop']

    # State Args
    STATE_LABELS = [['w'], ['r', 'n']]
    STATE_WINSIZE = 4

    with annotations.Pinnacle(apath, start=6) as reader:
        annotes = reader.read()

    # preprocess a producer for between 'Start' & 'Stop' annotes
    a, b = [int(a.time * FS) for a in annotes if a.label in BETWEEN]
    pro = preprocess(epath, channels=CHS, fs=FS, M=M, start=a, stop=b,
                     chunksize=CSIZE, axis=AXIS)

    # build threshold masks
    thresholded = masks.threshold(pro, nstds=NSTDS, winsize=WINSIZE, 
                                  radius=RADIUS)
    thresholded_names = [f'threshold={std}' for std in NSTDS]
    thresholded = dict(zip(thresholded_names, thresholded))
    
    # build annotation mask
    annotated = masks.artifact(apath, pro.shape[pro.axis], LABELS, DFS, BETWEEN)
    
    # build state masks
    stated = [masks.state(spath, ls, DFS, STATE_WINSIZE) for ls in STATE_LABELS]
    state_names = ['awake', 'sleep']
    stated = dict(zip(state_names, stated))

    if verbose:
        name = Path(epath).stem
        print(f' Completed building MetaMask for {name}'
              f' in {time.perf_counter() - t0} s')

    return MetaMask(annotated=annotated, **thresholded, **stated)


def evaluate_metamask(epath, apath, spath, verbose=True):
    """Evaluates threshold masks built from one EEG, annotation & spindle file.

    This function evaluates the masks defined by the fixed arguments in
    make_metamask. It computes the accuracies, sensitivities, specificities,
    precisions and percent_withins for each std in NSTDS.

    Returns:
        A nested dict keyed on metrics then keyed on mask name of metric results
        (e.g. {'accuracy': {'awake + threshold=3: 0.9776, ...}})
    """

    result = {'accuracy': {}, 'sensitivity': {}, 'specificity': {}, 
              'precision': {}, 'perc_within': {}}

    metamask = make_metamask(epath, apath, spath)
    
    # actual & detected are name, mask tuple and list of name, mask tuples
    _, ann_mask = metamask('awake', 'annotated')
    detected = [metamask('awake', attr) for attr, _ in metamask.__dict__.items()
            if 'threshold' in attr]

    for mask_name, mask in detected:

        # compare metrics of Falses
        tp = metrics._tp(~mask, ~ann_mask)
        tn = metrics._tn(~mask, ~ann_mask)
        fp = metrics._fp(~mask, ~ann_mask)
        fn = metrics._fn(~mask, ~ann_mask) 

        result['accuracy'][mask_name] = metrics.accuracy(tp, tn, fp, fn)
        result['sensitivity'][mask_name] = metrics.sensitivity(tp, fn)
        result['specificity'][mask_name] = metrics.specificity(tn, fp)
        result['precision'][mask_name] = metrics.precision(tp, fp)
        result['perc_within'][mask_name] = metrics.percent_within(mask, 
                                                                  ann_mask)

    if verbose:
        print(f'Completed mask evaluation for file: {epath.stem}')

    return result


def evaluate(dirpath, savepath=None, ncores=None):
    """Computes the accuracy, sensitivity, specificity, precision and
    percent_within for each eeg, annotation, and spindle file pairing in
    dirpath and stores the performances to a dict.

    The performance dict is keyed on each metric and each value is a list of
    metric values one per file pairing (animal) in dirpath.

    Args:
        dirpath:
            A directory containing edf, pinnacle annotations and spindle files
            to pair by path stem.
        savepath:
            A path location to save the performance dict to.
        ncores:
            The number of processing cores to utilize. If None, all the
            available cores will be used.

    Returns:
        A dict of performances containing the performance of each file pairing
        in dirpath.

    Stores:
        A pickle of the performance dict at savepath.
    """


    t0 = time.perf_counter()

    # get file names by glob patterns with matching suffixes
    epaths = list(Path(dirpath).glob('*.edf'))
    apaths = list(Path(dirpath).glob('*.txt'))
    spaths = list(Path(dirpath).glob('*.csv'))

    # collate the files by animal names
    # use regex matching to match filenames using animal name
    # TODO openseize's re_match only matches for two sets of filenames
    # would be nice to expand to any number of sets. Here is a workaround
    a = path_utils.re_match(epaths, apaths, r'\w+_')
    b = path_utils.re_match(epaths, spaths, r'\w+_')
    paths = []
    for (epath, apath), (epath, spath) in zip(a, b):
        paths.append((epath, apath, spath))

    workers = concurrency.set_cores(ncores, len(paths))
    with Pool(workers) as pool:
        processed = pool.starmap(evaluate_metamask, paths)

    performances = {'accuracy': defaultdict(list),
                    'sensitivity': defaultdict(list),
                    'specificity': defaultdict(list), 
                    'precision': defaultdict(list),
                    'perc_within': defaultdict(list)}

    for result in processed:
        for metric_name, perf_dict in result.items():
            for mask_name, value in perf_dict.items():
                performances[metric_name][mask_name].append(value)

    performances['names'] = [epath.stem for epath in epaths]
    print(f'Processed {len(paths)} files in {time.perf_counter() - t0} s')

    if not savepath:
        savepath = Path(dirpath).joinpath('mask_performances.pkl')

    with open(savepath, 'wb') as outfile:
        pickle.dump(performances, outfile)

    return performances

            



if __name__ == '__main__':

    basepath = Path('/media/matt/DataD/Xue/EbbData/6_week_post/mask_test/')

    """
    efile = 'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30.edf'
    afile = 'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30.txt'
    sfile = ('CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30'
             '_PREPROCESSED_SPINDLE_labels.csv')

    epath, apath, spath = [basepath.joinpath(x) for x in [efile, afile, sfile]]
    #result = evaluate_metamask(epath, apath, spath)
    """

    performances = evaluate(dirpath=basepath)

    """
    import numpy as np
    basepath = Path('/media/matt/Zeus/jasmine/ube3a')

    efile = 'DL00B4_P043_nUbe3a_19_56_3dayEEG_2019-04-07_14_32_34.edf'
    afile = 'DL00B4_P043_nUbe3a_19_56_3dayEEG_2019-04-07_14_32_34.txt'
    sfile = ('DL00B4_P043_nUbe3a_19_56_3dayEEG_'
             '2019-04-07_14_32_34_ALLEN_CW_day2_sleep_states.csv')

    epath, apath, spath = [basepath.joinpath(x) for x in [efile, afile, sfile]]

    metamask = make_metamask(epath, apath, spath, verbose=True)

    save_path = '/media/matt/Zeus/sandy/results/DL00B4_P043_masks_merge125.pkl'
    with open(save_path, 'wb') as outfile:
        pickle.dump(metamask, outfile)
    """
 
    """
    dirpath = Path('/media/matt/Zeus/jasmine/stxbp1')
    performances = evaluate(dirpath,
        savepath=('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/mask_parameter_testing/'
                  'stxbp1_mask_performances_avg10mins_radius0pt5secs.pkl'))
    """
