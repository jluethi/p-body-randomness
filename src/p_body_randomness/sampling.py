import numpy as np
from scipy.stats import rv_discrete
from metrics import nearest_neighbor_distance


def sample_pbodies(sampling_map, n, area_fn=None, min_distance = 6):
    '''
    Args:
        cytoplasmic_mask: Either a binary numpy mask representing the sample
            area or a intensity map representing sampling probabilities
        n: The number of pbodies to sample
        area_fn: A function to sample the p body area from or None
        min_distance: Int. Minimal distance that a newly sampled P-body has to
        all existing P-bodies. If it's sampled closer to an existing P-body,
        the sampling is redone (becase P-bodies can't biologically overlap,
        they would have fused and be treated as 1 P-body)

    Returns a list of n sampled positions
    '''

    samples = []
    for x in _sample_x_from_map(sampling_map, n):
        y = _sample_y_for_x(sampling_map, x)

        try:
            area = area_fn()
        except TypeError:
            area = 0

        # Check distance to existing P-pbodies
        if len(samples) > 1:
            try:
                # Calculate the nearest neighbor distance to all other existing P-pbodies
                min_distance_measured = min(nearest_neighbor_distance(samples))
                if min_distance_measured > min_distance:
                    samples.append((x, y, area))
                # when only 2 P-bodies are present and it samples the same location,
                # the nearest neighbor function throws a ValueError
            except ValueError:
                pass

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
