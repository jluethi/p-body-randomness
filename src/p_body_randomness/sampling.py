import numpy as np
import random


def sample_pbodies(cytoplasmic_mask, n, area_fn=None):
    '''
    Args:
        cytoplasmic_mask: A binary numpy mask representing the sample area
        n: The number of pbodies to sample
        area: A function to sample the p body area from or None
    '''

    samples = []
    while len(samples) < n:
        y, x = random.randint(0, cytoplasmic_mask.shape[0] - 1), \
            random.randint(0, cytoplasmic_mask.shape[1] - 1)
        if cytoplasmic_mask[y, x] == 0:
            continue
        else:
            try:
                area = area_fn()
            except TypeError:
                area = 0

            samples.append((x, y, area))

    return samples
