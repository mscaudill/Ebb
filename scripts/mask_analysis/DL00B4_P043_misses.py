from pathlib import Path
import numpy as np
from openseize.file_io.annotations import Pinnacle
from openseize import producer
from openseize.spectra.estimators import psd

from spectraprints.scripts.mask_evaluations import make_metamask, preprocess

# GLOBALS
BASE = Path('/media/matt/Zeus/jasmine/ube3a')
EFILE = 'DL00B4_P043_nUbe3a_19_56_3dayEEG_2019-04-07_14_32_34.edf'
AFILE = 'DL00B4_P043_nUbe3a_19_56_3dayEEG_2019-04-07_14_32_34.txt'
SFILE = ('DL00B4_P043_nUbe3a_19_56_3dayEEG_'
         '2019-04-07_14_32_34_ALLEN_CW_day2_sleep_states.csv')
EPATH, APATH, SPATH = [BASE.joinpath(x) for x in [EFILE, AFILE, SFILE]]
THRESHOLD = 'threshold=5'
FS = 5000
LABELS = ['Artifact', 'Artifact ','water_drinking','water_drinking ','']
START_LABEL = 'Start'
STOP_LABEL = 'Stop'


def false_negatives(metamask=None, threshold=THRESHOLD):
    """ """

    with Pinnacle(APATH, start=6) as areader:
        annotes = areader.read(labels=LABELS)
    
    with Pinnacle(APATH, start=6) as areader:
        start_annote = areader.read(labels=[START_LABEL])
    start_sample = start_annote[0].time * FS

    if metamask is None:
        metamask = make_metamask(EPATH, APATH, SPATH, verbose=True)

    # want to compare positions of artifacts so flip from False to True
    annotated = ~metamask('awake', 'annotated')[1]
    detected = ~metamask('awake', THRESHOLD)[1]

    annotated = annotated.astype(int)
    detected = detected.astype(int)

    # false negatives relative to annotation start times down-sample rate
    fns = np.where((annotated - detected) > 0)[0] * 20
    fns = fns.astype(float)
    # adjust false negatives relative to recording start
    fns += start_sample

    return fns
   

def annotation_stats(dirpath=BASE):
    """ """

    results = {}
    apaths = list(Path(dirpath).glob('*.txt'))
    for apath in apaths:
        with Pinnacle(apath, start=6) as areader:
            annotes = areader.read(labels=LABELS)

        name = str(apath.stem).split('_')[0]
        results[name] = [ann.duration for ann in annotes]
    return results


def psd_diffs(metamask):
    """ """

    with Pinnacle(APATH, start=6) as areader:
        annotes = areader.read(labels=[START_LABEL, STOP_LABEL])
    start = annotes[0].time * FS
    stop = annotes[-1].time * FS

    pro = preprocess(EPATH, [0,1,2], FS, 20, int(start), int(stop))

    # get mask where threshold is True but Annotated is False (i.e. False
    # negatives) gets the ones ignored by the annotation
    t5 = metamask('awake', 'threshold=5')[1]
    ann = metamask('awake', 'annotated')[1]

    mask = ~ann.astype(int) - ~t5.astype(int) > 0
    pro = producer(pro, chunksize=int(30e5), axis=-1, mask=~mask)
    cnt, freqs, estimate = psd(pro, fs=250)
    return cnt, freqs, estimate





if __name__ == '__main__':

    import pickle
    import matplotlib.pyplot as plt
    
    
    with open('/media/matt/Zeus/sandy/results/DL00B4_P043_masks.pkl',
              'rb') as infile:
        metamask = pickle.load(infile)

    """
    missed = false_negatives(metamask=None)
    """

    stats = annotation_stats()

    """
    fig, axarr = plt.subplots(4, 3, figsize=(13,11))
    for index, (key, durations) in enumerate(stats.items()):

        row, col = np.unravel_index(index, (4,3))
        axarr[row, col].hist(durations)
        title = ( f' max duration = {np.max(durations)}'
                  f' \n Counts > 100 secs '
                  f'  = {len(np.where(np.array(durations)>100)[0])}')
        axarr[row, col].set_title(key)
        axarr[row, col].text(.2, .7, title, transform=axarr[row, col].transAxes)
        axarr[row, col].spines[['right', 'top']].set_visible(False)


        
    axarr[0,0].set_xlabel('Annotation Duration (s)')
    axarr[0,0].set_ylabel('Counts')
    # remove last axis
    axarr[-1,-1].axis('off')
    axarr[-1, -2].axis('off')

    plt.suptitle(f'UBE3A Annotation Durations')

    plt.subplots_adjust(wspace=.4, hspace=.6)
    plt.show()
    """

    cnt, freqs, estimate = psd_diffs(metamask)
    fig, axarr = plt.subplots(1,3)
    for signal, ax in zip(estimate, axarr):
        ax.plot(freqs, signal)
    axarr[0].set_title('Genotype: ube3a, Animal DL00B4 \n'
                     'Included in Threshold but removed by Annotated')

    plt.show()





