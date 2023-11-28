"""

"""
import copy
import csv
from pathlib import Path
import pickle
from typing import List, Optional, Tuple, Union
from functools import partial
from dataclasses import dataclass, asdict
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
        self.kwargs = kwargs

        # initialize FOOOF models for both aperiodic modes
        modes = ['fixed', 'knee']
        self.models = {m: FOOOF(aperiodic_mode=m, **kwargs) for m in modes}

        # populate instance with presaved fits
        self.target = None
        self.to_save = {}
        self.idx = 0
        if target:
            self.target = Path(target)
            self.init_data()

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

    def init_data(self):
        """Populates this Parameterizer with presaved fits and sets the index as
        the first item following the last saved item."""

        if not self.target.exists():
            return
        x = self.read()
        self.to_save = {idx: Fitted(**fitdict) for idx, fitdict in x.items()}
        self.idx = list(self.to_save.keys())[-1] + 1
        print(f'Parameterizer populated with presaved fits from\n{self.target}')

    def read(self):
        """Reads presaved fooof fits from this Parameterizer's target."""

        with open(target, 'rb') as infile:
            result = pickle.load(infile)
        return result

    def write(self):
        """Writes fooof fits to this Parameterizer's target."""

        if not self.target.parent.exists():
            Path.mkdir(self.target.parent, parents=False)
        dicts = {idx: asdict(fitted) for idx, fitted in self.to_save.items()}
        with open(target, 'wb') as outfile:
            pickle.dump(dicts, outfile)

    def init_figure(self, figsize: Tuple[float, float]) -> None:
        """Stores a formatted matplolib figure and axis array to this
        Parameterizers instance.

        Args:
            figsize:
                A tuple of width, height for the initial figure size.
        """

        self.fig, self.axarr = plt.subplots(1, 2, figsize=figsize)
        self.fig.subplots_adjust(left=0.08, bottom=0.1, right=.82, top=0.90)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def on_close(self, event):
        """On close of this Parameterizer save Fitted instances to target."""

        if self.target:
            print(f'Writing data to {self.target}')
            self.write()
            print('Write Complete!')

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

    def fetch_trace(self):
        """Returns the data trace of this Parameterizer's current item index."""

        names = 'states paths channels'.split()
        selectors = {k: [v] for k, v in zip(names, self.items[self.idx])}
        return self.psds.select(**selectors).data.squeeze()

    def save(self, event):
        """Adds the currently displayed fit of this parameterizer to the save
        list."""

        # get parameters to build fitted 
        state, path, ch = self.items[self.idx]
        aperiodic_mode = self.radio_buttons.value_selected
        data = self.fetch_trace()
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
                    model.freq_range,
                    )
        # update the save list
        self.to_save.update({self.idx: fitted})
        print((f'Adding to save list:\nState: {state}\nPath: {path:10}'
            f'\nChannel: {ch}\nAperiodic mode: {aperiodic_mode}\n'))

    def add_buttons(self):
        """Adds a panel to select 'fixed' or 'knee' parameter for the aperiodic
        component of a FOOOF model."""

        self.radio_ax =plt.axes([.87, .3, .1, .1], frameon=False)
        self.radio_buttons = widgets.RadioButtons(self.radio_ax,
                labels=['fixed', 'knee'], activecolor='black')
        self.radio_buttons.set_label_props({'fontsize':[16, 16]})

    def update_info(self):
        """Updates this Parameterizer's displayed item information."""

        if hasattr(self, 'textvar'):
            self.textvar.set_text('')
        state, path, ch = self.items[self.idx]
        info = f'state:{state}\nfile: {path[:10]}...\nch: {ch}'
        self.textvar = self.fig.text(0.83, 0.45, info, size=10)

    def update(self):
        """Updates the plotted FOOOF model."""

        [ax.clear() for ax in self.axarr]

        # update aperiodic mode if this items data is already fit
        if self.idx in self.to_save:
            presaved = self.to_save[self.idx]
            idx = ['fixed', 'knee'].index(presaved.aperiodic_mode)
            self.radio_buttons.set_active(idx)

        # get the data trace and make a FOOOF model fit
        state, path, ch = self.items[self.idx]
        data = self.fetch_trace()
        for model in self.models.values():
            model.fit(self.freqs, data, freq_range=self.freq_range)

        # plot the FOOOF model's fit
        for idx, (name, model) in enumerate(self.models.items()):
            model.plot(plot_peaks='dot', plt_log=True, ax=self.axarr[idx])
            title = f'{name.upper()} Fit R^2 = {model.r_squared_:.4f}'
            self.axarr[idx].set_title(title)

        # configure plots and update item info
        self.axarr[-1].legend().set_visible(False)
        self.axarr[-1].set_ylabel('')
        self.update_info()

        plt.draw()


if __name__ == '__main__':

    path = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/PSDs.pkl'
    marr = MetaArray.load(path)

    target = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/fits/foof_fits.pkl'
    param = Parameterizer(marr, (4,100), target=target, peak_width_limits=(2,10))
