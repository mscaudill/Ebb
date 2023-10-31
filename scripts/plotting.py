from typing import Dict, List, Optional

from matplotlib import pyplot as plt
import numpy as np

def boxplot_dict(data: Dict, show_data: bool = False, jitter: float = 0,
                 ax=None, **kwargs):
    """Boxplots the data in dictionary using keys as labels.

    Args:
        data:
            Dataset containing named arrays to boxplot.
        show_data:
            Boolean indicating if individual data points should be overlayed on
            the boxplots.
        jitter:
            The maximum x-displacement per sequence in data if show_data
            otherwise this parameter is ignored.
        kwargs:
            Any valid keyword argument for matplotlib's boxplot.
    
    Returns:
        A matplotlib axes instance.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    groups = data.keys()
    ax.boxplot(data.values(), labels=groups, **kwargs)
    if show_data:
        for idx, values in enumerate(data.values(), 1):
            locs = np.random.uniform(-jitter, jitter, len(values)) + idx
            ax.scatter(locs, values)
    return ax
