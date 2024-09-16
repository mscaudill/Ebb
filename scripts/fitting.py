"""A matplotlib based interface for interactively performing FOOOF fits on
a collection of PSDResults.

https://fooof-tools.github.io/fooof/auto_tutorials/
"""

import copy
import dataclasses
import pickle
import textwrap
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from fooof import FOOOF
from matplotlib import widgets

#from ebb.scripts.psds import PSDResult
from ripple.psds import PSDResult


class Parameterizer:
    """An interactivate matplotlib plot that allows for customization of FOOOF
    models and their fits.

    Attributes:
        path:
            String or Path instance to a pickled list of PSDResult dataclasses.
        results:
            A list of PSDResult dataclass instances.
        default_params:
            A dictionary of the default FOOOF model parameters passed to the
            initializer.
        freq_range:
            A 2-tuple of frequency ranges over which the FOOOF model is fit.
        save:
            A boolean indicating if the fits should be saved to the PSDResults
            fit attribute on close of this Parameterizer's figure.
        result:
            The current PSDResult instance whose PSDs are displayed.
        psd_index:
            The index of the currently active axis displaying a single PSD of
            the current result.
        xscale:
            A string in {'log', 'linear'} for scaling frequencies.
        yscale:
            A string in {'log', 'linear'} for scaling PSDs
    """

    def __init__(
        self,
        path: Union[str, Path],
        target: Union[str, Path],
        peak_width_limits: Tuple[float, float] = (1, 5),
        max_n_peaks: Optional[int] = np.inf,
        min_peak_height: float = 0.1,
        peak_threshold: float = 2,
        aperiodic_mode: str = 'fixed',
        freq_range: Tuple[float, float] = (2, 100),
        save: bool = False,
        figsize: Tuple[float, float] = (10, 4),
        xscale: str = 'log',
        yscale: str = 'log',
        **kwargs,
    ) -> None:
        """Initialize this Parameterizer."""

        self.path = path
        with open(path, 'rb') as infile:
            self.results = [PSDResult(**attrs) for attrs in pickle.load(infile)]
        self.target = Path(target)

        self.default_params = {
                'peak_width_limits': peak_width_limits,
                'max_n_peaks': max_n_peaks,
                'min_peak_height': min_peak_height,
                'peak_threshold': peak_threshold,
                'aperiodic_mode': aperiodic_mode,
                }

        self.initialize_fits()
        self.freq_range = freq_range
        self.save = save

        # get the index of the last fitted psd
        idxs = [idx for idx, res in enumerate(self.results) if res.is_fit]
        self.result_index = idxs[-1] if idxs else 0
        self.psd_index = 0

        # set the power and frequency scales to log-scale
        self.xscale = xscale
        self.yscale = yscale

        # intialize this Parameterizer's widgets
        self.init_figure(figsize)
        self.add_forward()
        self.add_reverse()
        self.add_counter()
        self.add_entry()
        self.add_cycle_up()
        self.add_cycle_down()
        self.add_components()
        self.add_table()
        self.add_discard()

        # call update to display and place pickeable text for axis scaling
        self.update()
        self.add_xscale()
        self.add_yscale()

        plt.ion()
        plt.show()

    def initialize_fits(self):
        """Set the FOOOF parameters for all results on Parameterizer
        initialization to prior set parameters or defaults."""

        for result in self.results:
            for idx, fit in enumerate(result.fits):
                dic = copy.deepcopy(self.default_params)
                dic.update(fit)
                result.fits[idx] = dic

    @property
    def result(self):
        """Returns the result currently being viewed."""

        return self.results[self.result_index]

    @property
    def params(self):
        """Returns the dict of FOOOF  initializer parameters for the current
        result and current psd index."""

        return self.result.fits[self.psd_index]

    def init_figure(self, figsize: Tuple[float, float]) -> None:
        """Creates a matplolib figure and axis array instance.

        Args:
            figsize:
                A tuple of width, height for the initial figure size.
        """

        n = len(self.results[0].channels)
        self.fig, self.axarr = plt.subplots(1, n, figsize=figsize)
        [ax.spines[['top', 'right']].set_visible(False) for ax in self.axarr]
        self.fig.subplots_adjust(left=0.08, bottom=0.2, right=.8, top=0.90)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('pick_event', self.update_scale)

    def on_click(self, event):
        """On axis selection set the current PSD to the active axis."""

        if event.inaxes not in self.axarr:
            return

        self.psd_index = list(self.axarr).index(event.inaxes)
        self.update()
        self.recycle()

    def on_close(self, event):
        """On figure close, save PSDResult instances to target."""

        if self.save:
            self.write()
            print(f'PSD Results at {self.target} updated with fits')

    def write(self):
        """Writes the PSDResult instances back to this Parameterizer's path."""

        dics = [dataclasses.asdict(r) for r in self.results]
        with open(self.target, 'wb') as outfile:
            pickle.dump(dics, outfile)

    def add_yscale(self):
        """Add a pickable text for log or linear scaling the PSD."""

        label = 'log(PSD)' if self.yscale == 'log' else 'PSD'
        self.yscale_text = self.axarr[0].text(
                0.01,
                0.5,
                label,
                rotation=90,
                fontsize=12,
                color='k',
                backgroundcolor=[0.8, 0.8, 0.8],
                transform=self.fig.transFigure,
                picker=True
                )

    def add_xscale(self):
        """Add a pickable text for log or linear scaling of frequencies."""

        s = 'Frequency'
        label = f'log({s}) Hz' if self.xscale == 'log' else f'{s} Hz'
        self.xscale_text = self.axarr[0].text(
                0.12,
                0.06,
                label,
                fontsize=12,
                color='k',
                backgroundcolor=[0.8, 0.8, 0.8],
                transform=self.fig.transFigure,
                picker=True
                )

    def update_scale(self, event):
        """On a pick of xscale or yscale text, update the scale attr & plot."""

        if not isinstance(event.artist, plt.Text):
            pass

        text = event.artist
        label = text.get_text().lower()

        if 'psd' in label:
            if 'log' in label:
                new_label = 'PSD'
                self.yscale = 'linear'
            else:
                new_label = 'log(PSD)'
                self.yscale = 'log'
            text.set_text(new_label)

        if 'frequency' in label:
            if 'log' in label:
                new_label = 'Frequency Hz'
                self.xscale = 'linear'
            else:
                new_label = 'log(Frequency) Hz'
                self.xscale = 'log'
            text.set_text(new_label)

        self.update()

    def add_forward(self):
        """Configures and stores a button to advance the currently displayed
        PSD and FOOOF fit."""

        # build container axis, add widget and set callback
        self.forward_ax = plt.axes([0.9, .1, .04, 0.08])
        self.forward_button = widgets.Button(self.forward_ax, '>')
        self.forward_button.label.set_fontsize(32)
        self.forward_button.on_clicked(self.forward)

    def forward(self, event):
        """On forward button press advance this Parameterizers plot & metadata
        information. """

        # if current result is the last result forward declares it to be fit
        if self.result_index == len(self.results) - 1:
            self.result.is_fit = True
            self.update()

        # declare the previous result as fit and increment to next result
        self.result.is_fit = True
        self.result_index = min(self.result_index+1, len(self.results)-1)
        self.psd_index = 0
        self.update()
        self.recycle()

    def add_reverse(self):
        """Configures & stores a button to revert the currently displayed PSD
        and FOOOF fit."""

        # build container axis, add widget and set callback
        self.reverse_ax = plt.axes([0.85, .1, .04, 0.08])
        self.reverse_button = widgets.Button(self.reverse_ax, '<')
        self.reverse_button.label.set_fontsize(32)
        self.reverse_button.on_clicked(self.reverse)

    def add_counter(self):
        """Configures a text widget for reporting the result index of all
        results currently being viewed."""

        label = f'{self.result_index + 1}/{len(self.results)}'
        self.counter_text = plt.text(.87, .05, label, size=10,
                                     transform=self.fig.transFigure)

    def update_counter(self):
        """Updates the counter to report the result ind4ex of all results
        currently being viewed."""

        label = f'{self.result_index}/{len(self.results)-1}'
        self.counter_text.set_text(label)


    def reverse(self, event):
        """On reverse button press decrement this Parameterizer's index to
        previous item."""

        self.result_index = max(self.result_index - 1, 0)
        self.psd_index = 0
        self.update()
        self.recycle()

    def add_entry(self):
        """Adds a textbox axes and text box widget for changing a FOOOF Model
        for the currently selected PSD."""

        self.entry_ax = plt.axes([0.81, 0.78, 0.13, 0.08])
        names = list(self.params.keys())
        self.entry = widgets.TextBox(self.entry_ax, label=names[0])
        self.entry.set_val(self.params[names[0]])
        self.entry.label.set_position([1, 1.3])

        self.entry.on_submit(self.param_submit)

    def add_cycle_up(self):
        """Adds a button to reverse the displayed fit parameter."""

        self.cycle_up_ax = plt.axes([0.94, 0.82, 0.02, 0.04])
        self.cycle_up_button = widgets.Button(self.cycle_up_ax, '\u2191')
        self.cycle_up_button.on_clicked(self.cycle_up)

    def add_cycle_down(self):
        """Adds a button to forward the displayed fit parameter. """

        self.cycle_down_ax = plt.axes([0.94, 0.78, 0.02, 0.04])
        self.cycle_down_button = widgets.Button(self.cycle_down_ax, '\u2193')
        self.cycle_down_button.on_clicked(self.cycle_down)

    def cycle_up(self, event):
        """On reverse of parameter get the last FOOOF parameter and value."""

        names = list(self.params)
        values = list(self.params.values())
        param_idx = names.index(self.entry.label.get_text())
        new_idx = max(param_idx - 1, 0)
        self.entry.label.set_text(names[new_idx])
        self.entry.set_val(values[new_idx])

    def cycle_down(self, event):
        """On forward of parameter get the next FOOOF parameter and value."""

        names = list(self.params)
        values = list(self.params.values())
        param_idx = names.index(self.entry.label.get_text())
        new_idx = min(param_idx + 1, len(names) - 1)
        self.entry.label.set_text(names[new_idx])
        self.entry.set_val(values[new_idx])

    def param_submit(self, event):
        """Parameter submission occurs when enter key pressed or cycling. When
        this occurs we get the name and value in the entry box and update the
        paramter dict and the plot"""

        name = self.entry.label.get_text()
        value = event
        if name == 'peak_width_limits':
            value = tuple([float(s) for s in value[1:-1].split(',')])
        elif name == 'max_n_peaks':
            if value == 'inf':
                value = np.inf
            else:
                value = int(value)
        elif name == 'min_peak_height':
            value = float(value)
        elif name == 'peak_threshold':
            value = float(value)

        # param submit will be triggered when the fit is {} in which case we
        # should do nothing
        if self.result.fits[self.psd_index]:
            self.result.fits[self.psd_index].update({name: value})
        self.update()

    def recycle(self):
        """Gets the current set of parameters and updates the entry to the 0th
        item."""

        p = self.params if self.params else self.default_params
        self.entry.label.set_text(list(p.keys())[0])
        self.entry.set_val(list(p.values())[0])

    def add_components(self):
        """Adds 3 radio buttons for full, aperiodic and periodic FOOOF fitting
        components."""

        self.comp_ax = plt.axes([.79, 0.61, 0.2, 0.15])
        self.comp_ax.axis('off')
        components=['full', 'aperiodic', 'peak']
        self.radios = widgets.RadioButtons(self.comp_ax, labels=components)
        [r.set_fontsize(12) for r in self.radios.labels]
        # add a callback for component selection to update plot
        self.radios.on_clicked(self.select_component)

    def select_component(self, event):
        """On selection of a new component update the displayed PSDs."""

        self.update()

    def add_table(self):
        """Adds a table to report the peak parameters of each model fit."""

        self.table_ax = plt.axes([0.81, .3, 0.15, .3])
        self.table = widgets.TextBox(self.table_ax, label='')
        self.table_ax.axis('off')
        self.table.set_active(False)
        # hide the cursor by setting to bg color
        self.table.cursor.set_color('w')

    def update_table(self, data):
        """Updates the peak parameters reported in the table widget.

        Args:
            data:
                A len(peaks) x 3 array of centers powers and widths of FOOOF
                model detected peak fit values.
        """

        stringed = ['Center Power Width']
        for row in data:
            stringed.append(''.join(['{: ^8}'.format(el) for el in row]))
        msg = '\n'.join(stringed)
        self.table.set_val(msg)
        # handle fontsizes for large num of peaks
        if data.shape[0] <= 8:
            fsize = 10
        elif data.shape[0] <= 10:
            fsize = 8
        else:
            # this is the smallest readable
            fsize = 6
        self.table.text_disp.set_fontsize(fsize)

    def add_discard(self):
        """Adds a button to ignore the current PSD and its fit. """

        self.discard_axes = plt.axes([0.81, .2, 0.15, 0.07])
        self.discard_button = widgets.Button(self.discard_axes, 'Discard')
        self.discard_button.on_clicked(self.discard)

    def discard(self, event):
        """On discard press set the fit for the current PSD to be an empty
        dict."""

        if self.result.fits[self.psd_index]:
            self.results[self.result_index].fits[self.psd_index] = {}
        else:
            self.results[self.result_index].fits[self.psd_index] = self.default_params

        self.update()

    def clear(self):
        """On each plot update this method clears all drawn lines."""

        for ax in self.axarr:
            for line in list(ax.lines):
                line.remove()

    def update(self):
        """Update the currently drawn plot. This method is triggered by the
        following events:
            - A change in the x or y scale text
            - A change in the fitting parameters
            - A change in the component to plot
            - A change in the PSD to set active
            - A change in the Result to plot
        """

        # clear previous drawn lines and construct new models
        self.clear()

        # get the component and plot to this Parameterizer's axarr
        component = self.radios.value_selected
        models = []
        for idx, fit in enumerate(self.result.fits):

            pdict = fit if fit else self.default_params
            model = FOOOF(**pdict)
            model.fit(self.result.freqs, self.result.psd[idx], self.freq_range)
            models.append(model)
            f = np.log10(model.freqs) if self.xscale == 'log' else model.freqs
            model_fit = model.get_model(component, self.yscale)
            model_data = model.get_data(component, self.yscale)
            self.axarr[idx].plot(
                    f,
                    model_fit,
                    color='b',
                    label=component + ' fit',
                    alpha=0.2,
                    linewidth=2,
                    linestyle='--')
            self.axarr[idx].plot(
                    f,
                    model_data,
                    color='k',
                    label='data',
                    alpha=0.2)
            if not fit:
                self.axarr[idx].set_facecolor([1,0,0,.1])
            else:
                self.axarr[idx].set_facecolor([1,1,1])

            # reset the axes limits for each plot
            self.axarr[idx].set_xlim([np.min(f), np.max(f)])
            self.axarr[idx].set_ylim([np.min(model_data), np.max(model_data)])

        self.fig.suptitle(self.result.path.name + f' [{self.result.behavior}]')
        # set the current psd alpha to 1 (no transparency) & legend
        lines = self.axarr[self.psd_index].get_lines()
        [l.set_alpha(1) for l in lines]
        legend = self.axarr[0].legend()

        tabular = np.round(models[self.psd_index].get_params('peak_params'), 2)
        self.update_table(tabular)

        #update the counter
        self.update_counter()

        plt.draw()


if __name__ == '__main__':

    p = '/media/matt/Magnus/Qi/psds.pkl'
    target = '/media/matt/Magnus/Qi/psds_highfreq.pkl'
    param = Parameterizer(p, target, freq_range=(30, 100), save=False,
                          peak_width_limits=(1,5))
