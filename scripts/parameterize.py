"""

"""
from pathlib import Path
from typing import List, Optional, Tuple, Union
from functools import partial
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from ebb.core.metastores import MetaArray
from fooof import FOOOF
import itertools
import matplotlib.pyplot as plt
from matplotlib import widgets

@dataclass
class Fitted:
    """ """

    state: Union[str, Tuple[str,...]]
    path: Union[str, Path]
    channel: int
    freqs: npt.NDArray
    data: npt.NDArray
    # FOOOF constructor parameters
    peak_width_limits: Tuple[float, float]
    max_n_peaks: float
    min_peak_height: float
    peak_threshold: float
    aperiodic_mode: str
    # FOOOF fit parameters
    freq_range: Union[None, Tuple[float, float]]


    """
    fitted = Fitted(
                state,
                path,
                channel,
                freqs,
                data,
                model.peak_width_limits,
                model.max_n_peaks,
                model.min_peak_height,
                model.peak_threshold,
                model.aperiodic_mode,
                model.freq_range,
                )
    """

class Parameterizer:
    """ """

    def __init__(
            self,
            psds,
            freq_range,
            figsize=(11, 6),
            **kwargs
            ):
        """ """

        # need to create a pkl if it does not exist to write data to
        self.psds = psds
        self.freqs = np.array(psds.coords['frequencies'])
        self.freq_range = freq_range
        self.models = {
                'fixed': FOOOF(aperiodic_mode='fixed', **kwargs),
                'knee': FOOOF(aperiodic_mode='knee', **kwargs)
                }

        # FIXME need to compare combos ar=lready in save dir
        paths, states, channels = [self.psds.coords[x] for x in 
                ['paths','states','channels']]
        self.items = list(itertools.product(states,paths, channels))
        self.idx = 0


        self.init_figure(figsize)
        self.add_forward()
        self.add_reverse()
        self.add_buttons()
        self.update_info()
        
        self.update()
        plt.ion()
        plt.show()

    def init_figure(self, figsize):
        """ """

        self.fig, self.axarr = plt.subplots(1, 2, figsize=figsize)
        self.fig.subplots_adjust(left=0.08, bottom=0.1, right=.82, top=0.90)

    def add_forward(self):
        """Add a fully configured time advance button to this viewer."""

        # build container axis, add widget and set callback
        self.forward_ax = plt.axes([0.93, .1, .04, 0.08])
        self.forward_button = widgets.Button(self.forward_ax, '>')
        self.forward_button.label.set_fontsize(32)
        self.forward_button.on_clicked(self.forward)

    def forward(self, event):
        """ """
        
        # Forward to create and save a dataclass FitResult
        # it will need to get the current value of the radio to do this
        self.idx = min(self.idx + 1, len(self.items)-1)
        print(f'Saving {self.items[self.idx]}')
        self.update()

    def add_reverse(self):
        """Add a fully configured time reverse button to this viewer."""

        # build container axis, add widget and set callback
        self.reverse_ax = plt.axes([0.87, .1, .04, 0.08])
        self.reverse_button = widgets.Button(self.reverse_ax, '<')
        self.reverse_button.label.set_fontsize(32)
        self.reverse_button.on_clicked(self.reverse)

    def reverse(self, event):
        """ """

        print('reversed pressed')
        self.idx = max(self.idx - 1, 0)
        self.update()

    def add_buttons(self):
        """Adds a panel to select 'fixed' or 'knee' parameter for the aperiodic
        component of a FOOOF model."""

        self.radio_ax =plt.axes([.87, .2, .1, .1])
        self.radio_buttons = widgets.RadioButtons(self.radio_ax, 
                labels=['fixed', 'knee'], activecolor='black')
        self.radio_buttons.set_label_props({'fontsize':[16, 16]})
        self.radio_buttons.on_clicked(self.select)

    def select(self, label):
        """ """

        print(self.items[self.idx], label)

    def update_info(self):
        """ """
        
        if hasattr(self, 'textvar'):
            self.textvar.set_text('')
        state, path, ch = self.items[self.idx]
        info = f'state:{state}\nfile: {path[:10]}...\nch: {ch}'
        self.textvar = self.fig.text(0.83, 0.35, info, size=10)

    def update(self):
        """ """

        [ax.clear() for ax in self.axarr]

        state, path, ch = self.items[self.idx]
        data = self.psds.select(
                states=[state],
                paths=[path],
                channels=[ch]).data.squeeze()
        for model in self.models.values():
            # probably need to interpolate here at line noise
            model.fit(self.freqs, data, freq_range=self.freq_range)

        for idx, (name, model) in enumerate(self.models.items()):
            model.plot(plot_peaks='dot', plt_log=True, ax=self.axarr[idx])
            title = f'{name.upper()} Fit'
            self.axarr[idx].set_title(title)

        self.axarr[-1].legend().set_visible(False)
        self.axarr[-1].set_ylabel('')
        self.update_info()

        plt.draw()







if __name__ == '__main__':

    path = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/PSDs.pkl'
    marr = MetaArray.load(path)

    state = ('threshold', 'wake')
    path = 'CW0DD2_P106_KO_94_32_3dayEEG_2020-04-29_09_56_19_PREPROCESSED'
    channel=0
    freq_range=(4, 100)

    """
    fitted = parameterize(marr, state, path, channel, freq_range,
            peak_width_limits=(1,15))
    """

    param = Parameterizer(marr, (4,100), None, peak_width_limits=(1,15))
