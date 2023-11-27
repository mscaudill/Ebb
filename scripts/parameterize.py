"""

"""
import copy
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
    """A dataclass storing the attributes needed to reconstruct a fitted FOOOF
    instance.

    Attributes:
        state:
            The threshold and sleep state of this Fitted instance.
        path:
            The path to the EEG data of this Fitted instance.
        channel:
            The channel of the EEG data of this Fitted instance.
        freqs:
            A 1-D numpy array of frequencies at which the PSDs are estimated.
        data:
            A 1-D numpy array of power spectral values one per frequency in
            freqs.
        peak_width_limits:
            The min and max width of a peak, equivalent to the gaussians std.
            for a FOOOF initializer.
        max_n_peaks:
            The maximum number of peaks to detect for a FOOOF initializer.
        peak_threshold:
            The number of standard deviations used to detect peaks passed to
            a FOOOF initializer.
        aperiodic_mode:
            A string in {'fixed', 'knee'} to indicate if there is a knee in the
            log-log plot of power vs. frequency. This parameter is passed to
            a FOOOF's initializer.
        freq_range:
            The lower and upper limit of frequencies a FOOOF model should fit
            the PSD between. If None, the entire frequency range is fitted.
    """

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


class Parameterizer:
    """An interactive matplotlib plot that fits a FOOOF model with and without
    a 'knee' parameter in a FOOOF model for all data in a dir.

    Attributes:
        psds:
            A metaarray of stored power spectral densities to fit with FOOOF
            models.
        freq_range:
            The frequencies over which the FOOOF model will be fit.
        figsize:
            The initial size of the interactive figure. Default is (11,6).
        target:
            A path to a pickle file where Fitted results will be written to. If
            None, no saving occurs.
        **kwargs:
            Any valid kwarg for initializing a FOOOF model. These kwargs will be
            fixed and applied across all fitted models.
    """

    def __init__(
            self,
            psds: MetaArray,
            freq_range: Tuple,
            target: Optional[Union[str, Path]] = None,
            figsize: Tuple[float, float] = (11, 6),
            **kwargs
            ) -> None:
        """Initialize this Parameterizer."""

        # need to create a pkl if it does not exist to write data to
        self.psds = copy.deepcopy(psds)
        self.freq_range = freq_range
        self.freqs = np.array(psds.coords.pop('frequencies'))
        self.items = list(itertools.product(*psds.coords.values()))
        self.target, self.idx = self.init_data(target)
        self.kwargs = kwargs

        # initialize FOOOF models for both aperiodic modes
        modes = ['fixed', 'knee']
        self.models = {m: FOOOF(aperiodic_mode=m, **kwargs) for m in modes}

        # initialize this Parameterizer's widgets
        self.init_figure(figsize)
        self.add_forward()
        self.add_reverse()
        self.add_buttons()
        self.add_save()
        self.update_info()
        # update the plot & show
        self.update()
        plt.ion()
        plt.show()

    def init_data(self, target):
        """Returns a Path instance to a pickle of Fitted results creating one
        if needed and sets the index to the next unsaved item.

        Args:
            target:
                A path location where new fits will be saved to. If None, target
                is None type.

        Returns:
            A tuple (Path location, item index).
        """

        if not target:
            idx = 0
            target = None
        else:
            target = Path(target)
            if target.exists():
                # for presaved data find the first unsaved item
                with open(target, 'rb') as infile:
                    presaved = pickle.load(infile)
                    idx = self.items.index(presaved[-1]) + 1
            else:
                idx = 0

        return target, idx

    def init_figure(self, figsize: Tuple[float, float]) -> None:
        """Stores a formatted matplolib figure and axis array to this
        Parameterizers instance.

        Args:
            figsize:
                A tuple of width, height for the initial figure size.
        """

        self.fig, self.axarr = plt.subplots(1, 2, figsize=figsize)
        self.fig.subplots_adjust(left=0.08, bottom=0.1, right=.82, top=0.90)

    def add_forward(self):
        """Configures and stores a button to advance the currently displayed
        trace from the PSDs metaarray."""

        # build container axis, add widget and set callback
        self.forward_ax = plt.axes([0.93, .2, .04, 0.08])
        self.forward_button = widgets.Button(self.forward_ax, '>')
        self.forward_button.label.set_fontsize(32)
        self.forward_button.on_clicked(self.forward)

    def forward(self, event):
        """On forward button press advance this Parameterizers plot & metadata
        information. """

        self.idx = min(self.idx + 1, len(self.items)-1)
        self.update()

    def add_reverse(self):
        """Configures & stores a button to revert the currently displayed trace
        from the PSDs metaarray."""

        # build container axis, add widget and set callback
        self.reverse_ax = plt.axes([0.87, .2, .04, 0.08])
        self.reverse_button = widgets.Button(self.reverse_ax, '<')
        self.reverse_button.label.set_fontsize(32)
        self.reverse_button.on_clicked(self.reverse)

    def reverse(self, event):
        """On reverse button press decrement this Parameterizer's index to
        previous item."""

        self.idx = max(self.idx - 1, 0)
        self.update()

    def add_save(self):
        """Adds a save button for saving the current Fitted result to a CSV
        file."""

        self.save_ax = plt.axes([0.87, .08, .1, .08])
        self.save_button = widgets.Button(self.save_ax, 'SAVE')
        self.save_button.label.set_fontsize(16)
        self.save_button.on_clicked(self.save)

    def save(self, event):
        """ """

        aperiodic_mode = self.radio_buttons.value_selected
        state, path, ch = self.items[self.idx]
        data = self.psds.select(
                states=[state],
                paths=[path],
                channels=[ch]).data.squeeze()

        model = self.models[aperiodic_mode]
        fitted = Fitted(
                state,
                path,
                ch,
                model.freqs,
                model.power_spectrum,
                model.peak_width_limits,
                model.max_n_peaks,
                model.min_peak_height,
                model.peak_threshold,
                model.aperiodic_mode,
                model.freq_range)
        print(fitted)

        # open the file if it exist, create it if it doesn't and write to it!

    def add_buttons(self):
        """Adds a panel to select 'fixed' or 'knee' parameter for the aperiodic
        component of a FOOOF model."""

        self.radio_ax =plt.axes([.87, .3, .1, .1], frameon=False)
        self.radio_buttons = widgets.RadioButtons(self.radio_ax,
                labels=['fixed', 'knee'], activecolor='black')
        self.radio_buttons.set_label_props({'fontsize':[16, 16]})

    def update_info(self):
        """ """
        
        if hasattr(self, 'textvar'):
            self.textvar.set_text('')
        state, path, ch = self.items[self.idx]
        info = f'state:{state}\nfile: {path[:10]}...\nch: {ch}'
        self.textvar = self.fig.text(0.83, 0.45, info, size=10)

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

    """
    state = ('threshold', 'wake')
    path = 'CW0DD2_P106_KO_94_32_3dayEEG_2020-04-29_09_56_19_PREPROCESSED'
    channel=0
    freq_range=(4, 100)
    """

    """
    fitted = parameterize(marr, state, path, channel, freq_range,
            peak_width_limits=(1,15))
    """

    target = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/fits/foof_fits.pkl'
    param = Parameterizer(marr, (4,100), target=target, peak_width_limits=(2,10))
