import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectraprints.scripts.plotting import boxplot_dict

# Set the file location and load
FILENAME = 'ube3a_mask_performances_avg3mins_radius0pt5secs.pkl'
BASEPATH = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/mask_parameter_testing/'

fp = Path(BASEPATH).joinpath(FILENAME)
with open(fp, 'rb') as infile:
    performances = pickle.load(infile)

# pop animals from performances dict
animals = performances.pop('names')

# create large figure and axarr
fig, axarr = plt.subplots(2, 3, figsize=(13, 9))
for idx, (metric_name, dic) in enumerate(performances.items()):

    # compute row and column for this index
    row, col = np.unravel_index(idx, shape=(2,3))
    ax = boxplot_dict(dic, show_data=True, jitter=0.03, ax=axarr[row, col],
                      showfliers=False, medianprops=dict(color='k'))

    # manage ax props
    ax.set_ylabel(metric_name, fontweight='bold')
    ax.get_xaxis().set_ticks([])
    ax.spines[['right', 'top']].set_visible(False)

# create a legend
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
axarr[-1,-2].legend(performances['accuracy'].keys(), labelcolor=colors,
        handlelength=0, handletextpad=0)

# remove last axis
axarr[-1,-1].axis('off')

# manage plt properties
plt.suptitle(f'UBE3A Thresholding Window of 3 mins with Merge Radius = 0.5 secs'
             f'\n n = {len(animals)} animals')
plt.subplots_adjust(wspace=.4, hspace=.2)
plt.show()




