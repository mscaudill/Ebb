#!/usr/bin/env python
# coding: utf-8


import pickle
import matplotlib.pyplot as plt
from pathlib import PosixPath
from ebb.core.metastores import MetaArray
from openseize.spectra import metrics, plotting
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


#load in our data as a MetaArray object
meta_array = MetaArray.load('/media/claudia/Data_A/claudia/STXBP1_High_Dose_Exps_3/PSDs.pkl')

#place estimatives into convenient lists for CI calculations

#estimatives_dict = meta_array.metadata 

#reshape metadata into array of 

#rem_estimatives_list = estimatives_dict['estimatives'][::2]#all sleep 
#wake_estimatives_list = estimatives_dict['estimatives'][1::2]#all wake
#len(estimatives_dict['estimatives'])
#print(estimatives_dict['estimatives'])


for idx, animal in enumerate(meta_array.coords['paths']):
    
    my_meta_array = meta_array.select(paths=[animal])
    wake_my_meta_array = my_meta_array.select(states = [('threshold', 'wake')])
    rem_my_meta_array = my_meta_array.select(states = [('threshold', 'rem')])
    nrem_my_meta_array = my_meta_array.select(states = [('threshold', 'nrem')])
    #sleep_my_meta_array = my_meta_array.select(states = [('threshold', 'sleep')])
    chan_0_wake_my_meta_array = wake_my_meta_array.select(channels = [0])
    chan_1_wake_my_meta_array = wake_my_meta_array.select(channels = [1])
    chan_2_wake_my_meta_array = wake_my_meta_array.select(channels = [2])
    chan_0_rem_my_meta_array = rem_my_meta_array.select(channels = [0])
    chan_1_rem_my_meta_array = rem_my_meta_array.select(channels = [1])
    chan_2_rem_my_meta_array = rem_my_meta_array.select(channels = [2])
    chan_0_nrem_my_meta_array = nrem_my_meta_array.select(channels = [0])
    chan_1_nrem_my_meta_array = nrem_my_meta_array.select(channels = [1])
    chan_2_nrem_my_meta_array = nrem_my_meta_array.select(channels = [2])    
    #calculate confidence intervals
       # confidence_interval(psd: npt.NDArray[np.float64], n_estimates: int, alpha: float = 0.05) --> returns list of tuples

    #chan_0_wake_U_L_CI_list = metrics.confidence_interval(chan_0_wake_my_meta_array.data[0][0][0],wake_estimatives_list[idx])
    #chan_1_wake_U_L_CI_list = metrics.confidence_interval(chan_1_wake_my_meta_array.data[0][0][0],wake_estimatives_list[idx])
    #chan_2_wake_U_L_CI_list = metrics.confidence_interval(chan_2_wake_my_meta_array.data[0][0][0],wake_estimatives_list[idx])
   # chan_0_rem_U_L_CI_list = metrics.confidence_interval(chan_0_rem_my_meta_array.data[0][0][0],sleep_estimatives_list[idx])
    #chan_1_rem_U_L_CI_list = metrics.confidence_interval(chan_1_rem_my_meta_array.data[0][0][0],sleep_estimatives_list[idx])
    #chan_2_rem_U_L_CI_list = metrics.confidence_interval(chan_2_rem_my_meta_array.data[0][0][0],sleep_estimatives_list[idx])
    #if idx == 0:
     #   print(np.array(chan_0_wake_U_L_CI_list))
    
    #extract upper bounds
   # chan_0_wake_U_CI = np.array([tuple[0] for tuple in chan_0_wake_U_L_CI_list])
   # chan_1_wake_U_CI = np.array([tuple[0] for tuple in chan_1_wake_U_L_CI_list])
   # chan_2_wake_U_CI = np.array([tuple[0] for tuple in chan_2_wake_U_L_CI_list])
   # chan_0_sleep_U_CI = np.array([tuple[0] for tuple in chan_0_sleep_U_L_CI_list])
   # chan_1_sleep_U_CI = np.array([tuple[0] for tuple in chan_1_sleep_U_L_CI_list])
   # chan_2_sleep_U_CI = np.array([tuple[0] for tuple in chan_2_sleep_U_L_CI_list])
    #if idx == 0:
     #   print(chan_0_wake_U_CI_list)
    
    #extract lower bounds 
   # chan_0_wake_L_CI = np.array([tuple[1] for tuple in chan_0_wake_U_L_CI_list])
   # chan_1_wake_L_CI = np.array([tuple[1] for tuple in chan_1_wake_U_L_CI_list])
   # chan_2_wake_L_CI = np.array([tuple[1] for tuple in chan_2_wake_U_L_CI_list])
   # chan_0_sleep_L_CI = np.array([tuple[1] for tuple in chan_0_sleep_U_L_CI_list])
   # chan_1_sleep_L_CI = np.array([tuple[1] for tuple in chan_1_sleep_U_L_CI_list])
   # chan_2_sleep_L_CI = np.array([tuple[1] for tuple in chan_2_sleep_U_L_CI_list])
    
    #plot PSDs with confidence bands 
        # banded(x: npt.NDArray[np.float64],   
                #upper: npt.NDArray[np.float64],   
                #lower: npt.NDArray[np.float64], 
                #ax: plt.Axes,   **kwargs)
    
                
    fig, axs = plt.subplots(3,3)
    #fig, axs = plt.subplots(3,2, figsize = (14,6))
    f_start = 40
    f_stop = 101
    
    axs[0,0].plot(range(f_start,f_stop),chan_0_wake_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[0,0].set_title('Chan 0 Wake')
    axs[0,0].set_xlim(f_start,f_stop)
    axs[0,0].autoscale(enable=None, axis="y", tight=True)
    axs[0,0].set_xticks(np.arange(f_start,f_stop,10))
    #plotting.banded(range(0,251),chan_0_wake_U_CI, chan_0_wake_L_CI, axs[0,0])
    #banded(chan_0_wake_U_L_CI_list)
    #axs[0,0].set_ylim(0,100)

    axs[0,1].plot(range(f_start,f_stop),chan_0_rem_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[0,1].set_title('Chan 0 REM')
    axs[0,1].set_xlim(f_start,f_stop)
    axs[0,1].autoscale(enable=None, axis="y", tight=True)
    axs[0,1].set_xticks(np.arange(f_start,f_stop,10))
    #plotting.banded(range(0,251),chan_0_sleep_U_CI, chan_0_sleep_L_CI, axs[0,1])
    #axs[0,1].set_ylim(0,100)
    
    axs[0,2].plot(range(f_start,f_stop),chan_0_nrem_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[0,2].set_title('Chan 0 NREM')
    axs[0,2].set_xlim(f_start,f_stop)
    axs[0,2].set_xticks(np.arange(f_start,f_stop,10))
    axs[0,2].autoscale(enable=None, axis="y", tight=True)
    #axs[0,2].set_ylim(0,100)
    
    axs[1,0].plot(range(f_start,f_stop),chan_1_wake_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[1,0].set_title('Chan 1 Wake')
    axs[1,0].set_xlim(f_start,f_stop)
    axs[1,0].set_xticks(np.arange(f_start,f_stop,10))
    axs[1,0].autoscale(enable=None, axis="y", tight=True)
    #plotting.banded(range(0,251),chan_1_wake_U_CI, chan_1_wake_L_CI, axs[1,0])
    #axs[1,0].set_ylim(0,100)

    axs[1,1].plot(range(f_start,f_stop),chan_1_rem_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[1,1].set_title('Chan 1 REM')
    axs[1,1].set_xlim(f_start,f_stop)
    axs[1,1].set_xticks(np.arange(f_start,f_stop,10))
    axs[1,1].autoscale(enable=None, axis="y", tight=True)
    #plotting.banded(range(0,251),chan_1_sleep_U_CI, chan_1_sleep_L_CI, axs[1,1])
    #axs[1,1].set_ylim(0,100)
    
    axs[1,2].plot(range(f_start,f_stop),chan_1_nrem_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[1,2].set_title('Chan 1 NREM')
    axs[1,2].set_xlim(f_start,f_stop)
    axs[1,2].set_xticks(np.arange(f_start,f_stop,10))
    axs[1,2].autoscale(enable=None, axis="y", tight=True)
    #axs[1,2].set_ylim(0,100)
    
    axs[2,0].plot(range(f_start,f_stop),chan_2_wake_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[2,0].set_title('Chan 2 Wake')
    axs[2,0].set_xlim(f_start,f_stop)
    axs[2,0].set_xticks(np.arange(f_start,f_stop,10))
    axs[2,0].autoscale(enable=None, axis="y", tight=True)
    #plotting.banded(range(0,251),chan_2_wake_U_CI, chan_2_wake_L_CI, axs[2,0])
    #axs[2,0].set_ylim(0,100)

    axs[2,1].plot(range(f_start,f_stop),chan_2_rem_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[2,1].set_title('Chan 2 REM')
    axs[2,1].set_xlim(f_start,f_stop)
    axs[2,1].set_xticks(np.arange(f_start,f_stop,10))
    axs[2,1].autoscale(enable=None, axis="y", tight=True)
    #plotting.banded(range(0,251),chan_2_sleep_U_CI, chan_2_sleep_L_CI, axs[2,1])
    #axs[2,1].set_ylim(0,100)
    
    axs[2,2].plot(range(f_start,f_stop),chan_2_nrem_my_meta_array.data[0][0][0][f_start:f_stop])
    axs[2,2].set_title('Chan 2 NREM')
    axs[2,2].set_xlim(f_start,f_stop)
    axs[2,2].set_xticks(np.arange(f_start,f_stop,10))
    axs[2,2].autoscale(enable=None, axis="y", tight=True)
    #axs[2,2].set_ylim(0,100)

    fig.suptitle('PSD for ' + str(animal)[0:6])
    fig.supxlabel('Frequencies')
    fig.supylabel(r'Power [mV$^{2}$]')

    fig.tight_layout()
    plt.savefig(str(animal)[0:6] + '_highfreq.pdf')







