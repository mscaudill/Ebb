"""This module combines EDF files that prematurely stopped and were restarted by
experimenters joining the shorter files into a single edf.
"""

from typing import Union, Sequence

import re
from pathlib import Path
from openseize.file_io import edf

def validate_length(reader, minimum: float, fs: float) -> None:
    """Validates that reader has at least minimum number of hours samples.

    Args:
        reader:
            An openseize reader instance.
        minimum:
            The minimum nuber of hours for a reader.
        fs:
            The sampling rate of the data in the reader.

    Raises:
        A ValueError is raised if reader has less samples than mimimum requires.

    Returns:
        None
    """

    if reader.shape[-1] / (3600 * fs) < minimum:
        msg = f'Reader at path {reader.path} has less {minimum} hours.'
        raise ValueError(msg)

def locate(dirpath: Union[str, Path], fs: float, expected:float) -> Sequence:
    """Returns a list of EEG paths whose number of samples in hours is less than
    expected.

    Args:
        dirpath:
            Path to a directory containing EEG files.
        fs:
            The sampling rate of the data in the file.
        expected:
            The minimum number of hours expected in the file.

    Returns:
        A Sequence of Path instances whose number of samples in hours is below
        expected.
    """

    result = []
    paths = list(Path(dirpath).glob('*.edf'))
    for fp in paths:
        with edf.Reader(fp) as reader:
            if reader.shape[-1] / (3600 * fs) < expected:
                result.append(reader.path)
    return result

def pair(paths: Sequence, pattern=r'[^_]+'):
    """Returns tuples of paths each of which contains pattern..

    Args:
        paths:
            A sequence of path instances.
        pattern:
            A valid regex expression to match path stems by.

    Returns:
        A sequence of tuples containing paths with pattern in them.

    References:
        https://developers.google.com/edu/python/regular-expressions
    """

    result  = []
    while paths:
        # remove first path and get str matching pattern
        path = paths.pop(0)
        token = re.search(pattern, path.stem).group()
        matches = [other for other in paths if token in other.stem]

        if not matches:
            msg = f'No match find for toke {token}'
            raise ValueError(msg)
        [paths.remove(m) for m in matches]
        result.append((path, *matches))

    return result






if __name__ == '__main__':

    """
    path = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/'
       'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30.edf')

    reader = edf.Reader(path)
    validate_length(reader, 72, fs=5000)
    """

    paths = list(
            Path('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/short_files/').iterdir())

    result = pair(paths)
