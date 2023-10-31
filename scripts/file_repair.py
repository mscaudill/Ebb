""""This module corrects files that are segregated over multiple files due to
recording problems such as power-outages. It joins the files together into
a single EDF."""

import copy
import re
from pathlib import Path

import numpy as np

from openseize.file_io import edf


def validate_length(reader, minimum, fs):
    """Validates that a reader instance contains at least minimum hours of
    data."""

    if reader.shape[-1] / (3600 * fs) < minimum:
        msg = f'Reader at path {reader.path} has less than {minimum} hours.'
        raise ValueError(msg)

def locate(dirpath, fs, expected=72):
    """Locates EDF files in dirpath whose duration in hours is less than
    expected.

    Args:
        dirpath:
            A directory containing EDF files.
        fs:
            The sampling rate each EDF was recorded at assumed to be equivalent
            across all files in dirpath.
        expected:
            The number of hours expected in each file. Files below this are
            considered 'short'.

    Returns:
        A list of files whose number of recordings hours is below expected.
    """

    result = []
    paths = list(Path(dirpath).glob('*.edf'))
    for fp in paths:
        reader = edf.Reader(fp)
        if reader.shape[-1] / (3600 * fs) < expected:
            result.append(reader.path)
        reader.close()
    return result

def pair_paths(paths, pattern=r'[^_]+'):
    """Returns tuples of paths from  list of paths both of which contain the
    same pattern.

    Args:
        paths:
            A sequence of path instances.
        pattern:
            A valid regex pattern to use for matching.
    """

    result = []
    while paths:
        # remove first path & get mouse_id from path
        path = paths.pop(0)
        mouse_id = re.search(pattern, path.stem).group()
        # find other path with matching mouse_id & remove it
        matching = [other for other in paths if mouse_id in other.stem][0]

        if not matching:
            msg = f'No match find for mouse_id {mouse_id}'
            raise ValueError(msg)

        paths.remove(matching)
        # add path and matching path to results
        result.append((path, matching))

    return result

def combine(path, other, time, fs, minimum=24, save_dir=None):
    """Read n_hours of data from the start of each EDF file in path and other
    and writes the combined data to a new EDF file in save_dir.

    Args:
        path:
            A path to an edf file
        other:
            A path to another edf file
        time:
            The number of hours to take from the start of each EDF file to write
            to the new combined EDF file.
        fs:
            The sampling rate of the EDF data located at path and other.
        fs:
            The sampling rate of the files, assuming all files are sampled at the
            same rate.
        minimum:
            The minimum number of hours for a single file to be combined with
            another.
        save_dir:
            An optional directory to write the combined EDF file to. If None,
            the combined file will be written to the same dir as path.

    Returns:
        None
    """

    # create readers & sort by header start dates
    readers = [edf.Reader(fp) for fp in (path, other)]
    readers = sorted(readers, key=lambda r: r.header['start_date'])
    # validate the lengths of the readers
    [validate_length(reader) for reader in readers]
    arrs = [reader.read(time * fs * 3600) for reader in readers]
    data = np.concatenate(arrs, axis=-1)

    header = copy.deepcopy(readers[0].header)
    header['num_records'] = data.shape[-1] / header.samples_per_record[0]

    name = path.stem + '_COMBINED'
    parent = path.parent
    if save_dir:
        parent = save_dir
    write_path = Path(parent).joinpath(new_name).with_suffix(path.suffix)

    with edf.Writer(write_path) as writer:
        writer.write(header, data, channels=reader.channels)


if __name__ == '__main__':

    dirpath = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/short_files/'

    shorts = locate(dirpath, fs=5000)
    paired = pair_paths(shorts)

    path, other = paired[0]
    readers = combine(path, other, time=24, fs=5000, save_dir=None)
