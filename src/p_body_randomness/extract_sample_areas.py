#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Creates the mask of where the Monte Carlo simulation should sample the P-bodies
"""
import numpy as np
import cv2

def extract_sample_area(cell_mask, dapi_image, dapi_threshold = 10):
    '''
    Returns a binary mask of where sampling is possible (inside the cell border,
    NOT inside the nucleus => in the cytoplasm of the cells)
    Dapi Threshold is used to define the nucleus based on the dapi intensity image
    '''
    cell_mask_binary = cell_mask > 0
    # Make a mask of the nucleus based on smoothed dapi_img
    dapi_smoothed = cv2.GaussianBlur(dapi_img,(5,5),0)
    nucleus_segmented = dapi_smoothed > dapi_threshold

    # Combine the masks using XOR: creates a mask of the cytoplasm
    cytoplasm_mask = np.logical_xor(cell_mask_binary, nucleus_segmented) * 255

    return cytoplasm_mask


# cell_img_path = '/Users/Joel/p-body-randomness/data/input_data/20180606-SLP_Multiplexing_p1_C03_x000_y000_z000_t000_segmentation_Label12.png'
# dapi_img_path = '/Users/Joel/p-body-randomness/data/input_data/20180606-SLP_Multiplexing_p1_C03_x000_y000_z000_t000_2_DAPI_Label12.png'
#
# cell_img = cv2.imread(cell_img_path, 0)
# dapi_img = cv2.imread(dapi_img_path, 0)
#
# cytoplasm_mask = extract_sample_area(cell_img, dapi_img)
# cv2.imwrite('test_img.png', cytoplasm_mask)
