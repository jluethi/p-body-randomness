import numpy as np
from scipy.stats import rv_discrete


def sample_pbodies(sampling_map, n, area_fn=None):
    '''
    Args:
        cytoplasmic_mask: Either a binary numpy mask representing the sample
            area or a intensity map representing sampling probabilities
        n: The number of pbodies to sample
        area_fn: A function to sample the p body area from or None

    Returns a list of n sampled positions
    '''

    samples = []
    for x in _sample_x_from_map(sampling_map, n):
        y = _sample_y_for_x(sampling_map, x)

        try:
            area = area_fn()
        except TypeError:
            area = 0

        samples.append((x, y, area))

    return samples


def _sample_y_for_x(sampling_map, x):
    prob_y = rv_discrete(name='prob_y', values=(
        np.arange(sampling_map.shape[0]),
        sampling_map[:, x] / sampling_map[:, x].sum())
    )

    return prob_y.rvs(size=1)[0]


def _sample_x_from_map(sampling_map, n):
    # if len(np.unique(sampling_map) == 2):
    #     while

    # compute marginal probability
    column_sums = sampling_map.sum(axis=0)

    marginal_x = rv_discrete(name='marginal_x', values=(
        np.arange(len(column_sums)),
        column_sums / column_sums.sum())
    )

    x = marginal_x.rvs(size=n)

    return x
