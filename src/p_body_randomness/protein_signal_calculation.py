'''
Calculate what the average protein signals are.
'''

import cv2
import numpy as np

from p_body_randomness.smooth_image import smooth_protein_image

def calculate_mean_intensities(protein_signal, pbody_mask, cell_mask, nucleus_image):
    '''
    Function: Calculates average signal intensity in the cytoplasm and in areas surrounding p-bodies.
    Input:    Image files containing protein signal, p-body locations, cell segmentation, DAPI signal.
    Output:   Tuple, first float is average intensity in the cytoplasm, second float is average intensity around p-bodies.
    '''
    # Creating mask for the nucleus
    nucleus_smoothed = cv2.GaussianBlur(nucleus_image, (5,5), 0)
    nucleus_mask = nucleus_smoothed > 10

    # Removing small p-bodies from the image.
    small_p_body_kernel = np.ones((3, 3), np.uint8)
    no_small_p_bodies = cv2.morphologyEx(pbody_mask, cv2.MORPH_OPEN, small_p_body_kernel)

    # Dilating remaining p-bodies.
    dilation_kernel = np.ones((10, 10), np.uint8)
    dilated_pbodies = cv2.dilate(no_small_p_bodies, dilation_kernel)

    # Removing resulting areas that are not in the cytoplasm.
    dilated_pbodies[(cell_mask == 0) | (nucleus_mask != 0)] = 0

    # Determining which pixels are in p-body surroundings.
    p_body_surroundings = dilated_pbodies - pbody_mask > 0

    # Processing the image to make the protein signal smoother.
    proteins_smooth = smooth_protein_image(protein_signal, pbody_mask, cell_mask, nucleus_image)

    # Calculating average signal in the p-body surroundings.
    surroundings_signal = proteins_smooth[p_body_surroundings]
    surr_mean = np.mean(surroundings_signal)

    # Calculating average signal in the cytoplasm.
    cytoplasm_signal = proteins_smooth[(cell_mask != 0) & (nucleus_mask == 0)]
    cyto_mean = np.mean(cytoplasm_signal)

    return (cyto_mean, surr_mean)
