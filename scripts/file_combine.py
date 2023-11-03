"""This module combines EDF files that prematurely stopped and were restarted by
experimenters joining the shorter files into a single edf.
"""

import copy
import re
import reprlib
from operator import attrgetter
from pathlib import Path
from typing import Pattern, Sequence, Union

import numpy as np
from openseize.file_io import edf


def locate(dirpath: Union[str, Path], fs: float, expected: float) -> Sequence:
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


def pair(paths: Sequence, pattern=r'[^_]+') -> Sequence:
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

    result = []
    cpaths = copy.copy(paths)
    while cpaths:
        # remove first path and get str token with matching pattern
        path = cpaths.pop(0)
        token = re.search(pattern, path.stem).group()
        matches = [other for other in cpaths if token in other.stem]

        if not matches:
            msg = f'No match find for token {token}'
            raise ValueError(msg)

        [cpaths.remove(m) for m in matches]
        result.append((path, *matches))

    return result


def move_files(new_dir: Path, paths: Sequence[Path], **kwargs) -> None:
    """Moves each file with path at paths to a new directory.

    Args:
        new_dir:
            A Path instance to a directory where each path will be moved to.
        paths:
            Path instances of files to move to new_dir.
        kwargs:
            Keyword only arguments are passed to pathlibs mkdir function. These
            control whether errors are raised if the parents are not specified
            correctly or if the dir already exist. See pathlib.Path.mkdir

    Returns:
        None

    Raises:
        FileExistError or FileNotFoundErrors may be raised depending on kwargs.
    """

    new_dir.mkdir(**kwargs)
    for path in paths:
        path.rename(new_dir.joinpath(path.name))


def combine_edf(
    dirpath: Path,
    fs: float,
    amount: float = 24,
    nominal: float = 72,
    pattern: Pattern[str] = r'[^_]+',
    verbose: bool = True,
) -> None:
    """Combines EDF files whose number of samples is less than nominal hours.

    Args:
        dirpath:
            Path instance of directory containing edf files some of which may be
            shorter than nominal.
        fs:
            The sampling rate of the data in the files.
        amount:
            The amount of hours to take from the start of each file for
            combining. Defaults to 24 hours.
        nominal:
            The cutoff in hours that determines if a file is 'short'. Default is
            72 hours.
        pattern:
            A regex pattern string used to match files. Please see the re module
            for details.
        verbose:
            Boolean indicating if progress should be printed to stdout.

    Returns:
        None

    Notes:
        This function requires that each file be loaded into memory because
        Openseize does not yet allow for writing from producers. This means the
        memory consumption may be high.
    """

    dirpath = Path(dirpath)
    # locate and pair the files with less than nominal hrs of samples
    short_files = locate(dirpath, fs, nominal)
    path_tuples = pair(short_files, pattern)

    aRepr = reprlib.Repr()
    samples = amount * fs * 3600
    for tup in path_tuples:
        # make a reader instance for each file to join
        readers = [edf.Reader(fp) for fp in tup]
        readers = sorted(readers, key=attrgetter('header.start_date'))

        if verbose:
            msg = aRepr.repr([reader.path.stem for reader in readers])
            print('Combining: \n' + msg)

        # make a read to a pre-allocated arr for each reader in readers
        arr = np.zeros((*readers[0].shape[:-1], samples * len(readers)))
        for idx, reader in enumerate(readers):
            start, stop = idx * samples, (idx + 1) * samples
            arr[..., start:stop] = reader.read(0, samples)
            reader.close()

        # create a header and increase the num_records to match combined size
        header = copy.deepcopy(readers[0].header)
        header['num_records'] = arr.shape[-1] // header.samples_per_record[0]

        # make a new path for the combined file
        fp = reader.path.with_stem(readers[0].path.stem + '_COMBINED')
        with edf.Writer(fp) as writer:
            writer.write(header, arr, reader.channels, verbose=verbose)

    #  move each short file to a short files subdirectory
    move_files(dirpath.joinpath('short_files'), short_files)


if __name__ == '__main__':
    """
    path = ('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/'
       'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30.edf')

    reader = edf.Reader(path)
    validate_length(reader, 72, fs=5000)
    """

    """
    paths = list(
        Path('/media/matt/Zeus/STXBP1_High_Dose_Exps_3/short_files/').iterdir()
    )

    result = pair(paths)
    """

    dirpath = Path('/media/matt/DataD/Xue/EbbData/6_week_post')
    combine_edf(dirpath, fs=5000)
