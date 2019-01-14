'''
Smoothens the images which contain the protein signal.
'''

import numpy as np
import cv2

def smooth_protein_image(protein_signal, pbody_mask, cell_mask, nucleus_image, nucleus_percentage = 0, nucleus_threshold = 10):
    '''
    Function: Substracts the p-body signals from the image and smoothens it out.
    Input:    Protein signal (numpy array), p-body segmentation (numpy array), cell mask (numpy array)
    Output:   The smoothened image.
    '''
    # We first substract some signal in areas where p-bodies are located.
    no_p_bodies = 1 * protein_signal
    no_p_bodies[pbody_mask != 0] = no_p_bodies[pbody_mask != 0] * 0.75

    # We create a mask for the nucleus
    nucleus_smoothed = cv2.GaussianBlur(nucleus_image,(5,5),0)
    nucleus_mask = nucleus_smoothed > nucleus_threshold

    # We extract only the cytoplasmic signal
    cytoplasm = 1 * no_p_bodies
    cytoplasm[(cell_mask == 0) | (nucleus_mask != 0)] = 0

    # Calculating total signal from the cytoplasm and nucleus area.
    total_cytoplasm = np.sum(cytoplasm)
    nucleus_area = np.sum(nucleus_mask != 0)

    # Making sure nucleus intensity is a certain percentage from the overall signal.
    cytoplasm[nucleus_mask != 0] = (nucleus_percentage * total_cytoplasm)/(nucleus_area * (1 - nucleus_percentage))

    # Now we go through five rounds of smoothing the resulting image.
    smooth_array = 1 * cytoplasm
    for i in range(5):
        smooth_array = cv2.GaussianBlur(smooth_array, (9, 9), 0)

    # Finally, restricting signal to area within the cell mask.
    final_image = 1 * smooth_array
    final_image[cell_mask == 0] = 0

    return final_image

def add_nuclear_probability(cell_mask, nucleus_image, nucleus_percentage = 0.05, nucleus_threshold = 10, shrink_nucleus = 3):
    '''
    Function: Adds a low probability (/intensity) to the nucleus
    Input:    cell mask (numpy array),
    nucleus_percentage (percentage of signal that should be in the nucleus)
    Output:   The image with nuclear probability.
    '''
    # We create a mask for the nucleus
    nucleus_smoothed = cv2.GaussianBlur(nucleus_image,(5,5),0)
    nucleus_mask_unshrunk = nucleus_smoothed > nucleus_threshold
    kernel = np.ones((3,3),np.uint8)
    nucleus_mask = cv2.erode(np.array(nucleus_mask_unshrunk, dtype=np.uint8),kernel,iterations = shrink_nucleus)

    # Calculating total signal from the cytoplasm and nucleus area.
    total_cytoplasm = np.sum(cell_mask)
    nucleus_area = np.sum(nucleus_mask != 0)

    # Making sure nucleus intensity is a certain percentage from the overall signal.
    cell_mask[nucleus_mask != 0] = (nucleus_percentage * total_cytoplasm)/(nucleus_area * (1 - nucleus_percentage))

    return cell_mask
