"""A script for computing and storing EEG Metrics and/or Biomarkers.

"""

from ebb.core import metastores
from ebb.masking import masks
from openseize.file_io import edf
from openseize import producer
from openseize.spectra import estimators


if __name__ == '__main__':

    fp = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/standard/'
          'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30_PREPROCESSED.edf')

    state_path = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/spindle/'
                  'CW0DA1_P096_KO_15_53_3dayEEG_2020' + \
                  '-04-13_08_58_30_PREPROCESSED_SPINDLE_labels.csv')


    reader = edf.Reader(fp)
    reader.channels = [0, 1, 2]
    pro = producer(reader, chunksize=30e5, axis=-1)

    threshold = masks.threshold(pro, nstds=[5], winsize=1.2e4, radius=100)[0]
    wake = masks.state(state_path, labels=['w'], fs=200, winsize=4)
    sleep = masks.state(state_path, labels=['r', 'n'], fs=200, winsize=4)
    metamask = metastores.MetaMask(threshold=threshold, wake=wake, sleep=sleep)

    name, mask = metamask('threshold', 'wake')
    maskpro = producer(pro, chunksize=30e5, axis=-1, mask=mask)

    cnt, freqs, psd = estimators.psd(maskpro, fs=200)
