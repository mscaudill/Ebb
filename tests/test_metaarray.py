"""A module for testing a metaarray using PSD batch processed data.

For these tests to run you will need to save a pickle file of batch processed
PSDResult instances and a metaarray of formatted data.

Typical usage example:
    # -rA flag shows extra summary see pytest -h
    !pytest -rA test_metaarray::<TEST_NAME>
"""

import pickle
import pytest
import numpy as np

from pathlib import Path

from ebb.core.metastores import MetaArray
from ebb.scripts.psds import PSDResult

@pytest.fixture(scope='module')
def psd_results():
    """Returns a list of PSDResult instances computed by the batch function in
    scripts.psd"""

    path = '/media/matt/DataD/Xue/EbbData/6_week_post/standard/psd_results.pkl'
    with open(path, 'rb') as infile:
        results = pickle.load(infile)
    
    return [PSDResult(**result) for result in results]

@pytest.fixture(scope='module')
def metaarray():
    """Returns a metaarray instance computed by the as_array function in
    scripts.psd"""

    path = '/media/matt/DataD/Xue/EbbData/6_week_post/standard/psd_metaarray.pkl'
    return MetaArray.load(path)

def test_metaarry_ordering(psd_results, metaarray):
    """Test if each PSDResult instance in psd_results is correctly positioned in
    metaarray's psd data attribute."""

    states = metaarray.coords['states']
    names = metaarray.coords['names']
    
    for result in psd_results:
        state_idx = states.index(result.state)
        name_idx = names.index(result.path[:6])

        assert np.allclose(result.psd, metaarray.data[state_idx, name_idx, :, :])

def test_metaarray_estimatives(psd_results, metaarray):
    """Test if the number of estimatives for each state and path in metaarray
    matches the corresponding PSDResult instance."""

    states = metaarray.coords['states']
    # group the psd_results by state
    grouped = {state: [] for state in states}
    for result in psd_results:
        grouped[result.state].append(result)

    for state in states:
        result_estimatives = [r.estimatives for r in grouped[state]]
        marr_estimatives = metaarray.metadata['estimatives'][state]
        assert np.allclose(result_estimatives, marr_estimatives)

def test_select_method(psd_results, metaarray):
    """Test if random selection by a single state and single name from
    a metaarray gives the correct psd_results."""

    states = metaarray.coords['states']
    names = metaarray.coords['names']

    # group the psd_results by state
    grouped = {state: [] for state in states}
    for result in psd_results:
        grouped[result.state].append(result)

    rng = np.random.default_rng()
    random_states = rng.choice(states, size=30)
    random_names = rng.choice(names, size=30)
    for state, name in zip(random_states, random_names):
        state = tuple(state) # rng converts to array so remake as tup
        submeta = metaarray.select(states=[state], names=[name])
        probe = grouped[state][names.index(name)]
        assert np.allclose(submeta.data.squeeze(), probe.psd)







