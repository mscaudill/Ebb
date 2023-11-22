"""

"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from ebb.core.metastores import MetaArray
from fooof import FOOOF


def parameterize(psds, state, path, channel, freq_range, **kwargs):
    """ """
    
    models = {}
    models['fixed'] = FOOOF(aperiodic_mode='fixed', **kwargs)
    models['knee'] = FOOOF(aperiodic_mode='knee', **kwargs)

    marr = psds.select(states=[state], paths=[path], channels=[channel])
    data = marr.data.squeeze()
    freqs = np.array(marr.coords['frequencies'])

    for model in models.values():
        model.fit(freqs=freqs, power_spectrum=data, freq_range=freq_range)

    fig, axarr = plt.subplots(1, 2, figsize=(12, 6))
    for idx, (name, model) in enumerate(models.items()):
        model.plot(plot_peaks='dot', plt_log=True, ax=axarr[idx])
        title = f'FOOF with Aperiodic = {name}'
        axarr[idx].set_title(title)

    plt.show(block=False)
    msg = "Enter 'f' for fixed, 'k' for knee, or 'i' for ignore: "
    select = partial(input)
    selection = select(msg)
    while selection.lower() not in ['f', 'k', 'i', 'fixed', 'knee', 'ignore']:
        selection = select(msg)
    plt.close()

    # get the chosen model
    # save the selected model in JSON FMT using model's save method
    # return the chosen model


if __name__ == '__main__':

    path = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/PSDs.pkl'
    marr = MetaArray.load(path)

    state = ('threshold', 'wake')
    path = 'CW0DD2_P106_KO_94_32_3dayEEG_2020-04-29_09_56_19_PREPROCESSED'
    channel=0
    freq_range=(4, 100)

    parameterize(marr, state, path, channel, freq_range,
            peak_width_limits=(1,15))

