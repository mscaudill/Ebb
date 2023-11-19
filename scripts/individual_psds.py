#!/usr/bin/env python
# coding: utf-8

# In[23]:


import functools
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt
from ebb.core import concurrency, metastores
from ebb.masking import masks
from ebb.scripts.file_combine import pair
from openseize import producer
from openseize.file_io import edf
from openseize.spectra import estimators
from openseize.spectra import metrics, plotting


# In[2]:


@dataclass
class PSDResult:
    """A class for keeping track of a PSD and all metadata."""

    path: Union[str, Path]
    state: Tuple[str, ...]
    estimatives: int
    freqs: npt.NDArray
    psd: npt.NDArray


# In[3]:


def estimate(
    path: Union[str, Path],
    state_path: Union[str, Path],
    channels: List = [0, 1, 2],
    fs: float = 200,
    chunksize: int = int(30e5),
    labels: Dict = {'wake': ['w'], 'sleep': ['n', 'r']},
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


# In[37]:


if __name__ == '__main__':
    
    fp = ('/media/claudia/Data_A/claudia/STXBP1_High_Dose_Exps_3/standard/'
          'CW3203_P111_KO_19_56_3dayEEG_2019-04-15_13_25_13_PREPROCESSED.edf')

    state_path = ('/media/claudia/Data_A/claudia/STXBP1_High_Dose_Exps_3/spindle_outputs/'
                  'CW3203_P111_KO_19_56_3dayEEG_2019-04-15_13_25_13_PREPROCESSED_SPINDLE_labels.csv')

    results = estimate(fp, state_path, verbose=True)
    
    #create arrays for each channel, for each state
    wake_chan0 = results[0].psd[0]
    wake_chan1 = results[0].psd[1]
    wake_chan2 = results[0].psd[2]

    sleep_chan0 = results[1].psd[0]
    sleep_chan1 = results[1].psd[1]
    sleep_chan2 = results[1].psd[2]
    
    
    #compute CI for each channel, for each state
    chan_0_wake_U_L_CI_list = metrics.confidence_interval(wake_chan0,results[0].estimatives)
    chan_1_wake_U_L_CI_list = metrics.confidence_interval(wake_chan1,results[0].estimatives)
    chan_2_wake_U_L_CI_list = metrics.confidence_interval(wake_chan2,results[0].estimatives)
    chan_0_sleep_U_L_CI_list = metrics.confidence_interval(sleep_chan0,results[1].estimatives)
    chan_1_sleep_U_L_CI_list = metrics.confidence_interval(sleep_chan1,results[1].estimatives)
    chan_2_sleep_U_L_CI_list = metrics.confidence_interval(sleep_chan2,results[1].estimatives)

    #extract upper bounds
    chan_0_wake_U_CI = np.array([tuple[0] for tuple in chan_0_wake_U_L_CI_list])
    chan_1_wake_U_CI = np.array([tuple[0] for tuple in chan_1_wake_U_L_CI_list])
    chan_2_wake_U_CI = np.array([tuple[0] for tuple in chan_2_wake_U_L_CI_list])
    chan_0_sleep_U_CI = np.array([tuple[0] for tuple in chan_0_sleep_U_L_CI_list])
    chan_1_sleep_U_CI = np.array([tuple[0] for tuple in chan_1_sleep_U_L_CI_list])
    chan_2_sleep_U_CI = np.array([tuple[0] for tuple in chan_2_sleep_U_L_CI_list])

    #extract lower bounds 
    chan_0_wake_L_CI = np.array([tuple[1] for tuple in chan_0_wake_U_L_CI_list])
    chan_1_wake_L_CI = np.array([tuple[1] for tuple in chan_1_wake_U_L_CI_list])
    chan_2_wake_L_CI = np.array([tuple[1] for tuple in chan_2_wake_U_L_CI_list])
    chan_0_sleep_L_CI = np.array([tuple[1] for tuple in chan_0_sleep_U_L_CI_list])
    chan_1_sleep_L_CI = np.array([tuple[1] for tuple in chan_1_sleep_U_L_CI_list])
    chan_2_sleep_L_CI = np.array([tuple[1] for tuple in chan_2_sleep_U_L_CI_list])
    
    #plot arrays over frequencies
    fig, axs = plt.subplots(3,2, figsize = (14,6))

    axs[0,0].plot(range(0,201),wake_chan0)
    axs[0,0].set_title('Chan 0 Wake')
    axs[0,0].set_xlim(0,100)
    plotting.banded(range(0,201),chan_0_wake_U_CI, chan_0_wake_L_CI, axs[0,0])
    #axs[0,0].set_ylim(0,500)

    axs[0,1].plot(range(0,201),sleep_chan0)
    axs[0,1].set_title('Chan 0 Sleep')
    axs[0,1].set_xlim(0,100)
    plotting.banded(range(0,201),chan_0_sleep_U_CI, chan_0_sleep_L_CI, axs[0,1])
    #axs[0,1].set_ylim(0,500)

    axs[1,0].plot(range(0,201),wake_chan1)
    axs[1,0].set_title('Chan 1 Wake')
    axs[1,0].set_xlim(0,100)
    plotting.banded(range(0,201),chan_1_wake_U_CI, chan_1_wake_L_CI, axs[1,0])
    #axs[1,0].set_ylim(0,500)

    axs[1,1].plot(range(0,201),sleep_chan1)
    axs[1,1].set_title('Chan 1 Sleep')
    axs[1,1].set_xlim(0,100)
    plotting.banded(range(0,201),chan_1_sleep_U_CI, chan_1_sleep_L_CI, axs[1,1])
    #axs[1,1].set_ylim(0,500)

    axs[2,0].plot(range(0,201),wake_chan2)
    axs[2,0].set_title('Chan 2 Wake')
    axs[2,0].set_xlim(0,100)
    plotting.banded(range(0,201),chan_2_wake_U_CI, chan_2_wake_L_CI, axs[2,0])
    #axs[2,0].set_ylim(0,500)

    axs[2,1].plot(range(0,201),sleep_chan2)
    axs[2,1].set_title('Chan 2 Sleep')
    axs[2,1].set_xlim(0,100)
    plotting.banded(range(0,201),chan_2_sleep_U_CI, chan_2_sleep_L_CI, axs[2,1])
    #axs[2,1].set_ylim(0,500)

    animal = 'CW3203'
    fig.suptitle('PSD for ' + animal)
    fig.supxlabel('Frequencies')
    fig.supylabel(r'Power [mV$^{2}$]')

    fig.tight_layout()
    plt.savefig(animal + '_individual.pdf')

