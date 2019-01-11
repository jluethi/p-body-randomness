#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions to calculate centroids & statistics of P-bodies

"""
from math import pi, sqrt

import numpy as np

import cv2


def extract_centroids(pbodies_image: np.ndarray, area_threshold=5):
    centroid_list = []
    # convert the grayscale image to binary image
    ret, pbodies_binary = cv2.threshold(pbodies_image, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(pbodies_binary, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    # Loop through all P bodies of the cell
    for c in contours:

        # Calculate area of all P-bodies
        curr_area = cv2.contourArea(c)

        # Filter out the P-bodies with an area smaller than
        if curr_area > area_threshold:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid_list.append((cX, cY, curr_area))

    return centroid_list


def generate_centroids_mask(centroid_list: list, mask_shape=(640, 640)):
    '''
    Returns:
        A mask with centroids
    '''

    mask = np.zeros(shape=mask_shape, dtype=np.uint8)
    for centroid in centroid_list:
        cv2.circle(mask, (centroid[0], centroid[1]), int(
            sqrt(centroid[2] / pi)), (255, 0, 0), -1)

    return mask


def extract_centroids_in_sample_area(pbodies_image: np.ndarray,
                                     sample_mask: np.ndarray,
                                     area_threshold=5):
    # Extract the centroids of P bodies inside and outside the sample mask
    centroid_list_in_area = []
    centroid_list_outside_area = []
    sample_mask_binary = sample_mask > 0
    # convert the grayscale image to binary image
    ret, pbodies_binary = cv2.threshold(pbodies_image, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(pbodies_binary, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    # Loop through all P bodies of the cell
    for c in contours:

        # Calculate area of all P-bodies
        curr_area = cv2.contourArea(c)

        # Filter out the P-bodies with an area smaller than
        if curr_area > area_threshold:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # If centroids are in sample mask, add them to centroid_list_in_area
            if sample_mask_binary[cY, cX]:
                centroid_list_in_area.append((cX, cY, curr_area))
            else:
                centroid_list_outside_area.append((cX, cY, curr_area))

    return [centroid_list_in_area, centroid_list_outside_area]


if __name__ == '__main__':
    pbodies_img = cv2.imread(
        '/Users/Joel/p-body-randomness/data/input_data/20180606-SLP_Multiplexing_p1_C03_x000_y000_z000_t000_13_Pbody_Segm_Label12.png',
        0)
    print(extract_centroids(pbodies_img, 5))
    pbodies_img = cv2.imread(
        '/Users/Joel/p-body-randomness/data/input_data/20180606-SLP_Multiplexing_p1_C03_x000_y000_z000_t000_13_Pbody_Segm_Label14.png',
        0)
    img_dapi = cv2.imread(
        '/Users/Joel/p-body-randomness/data/input_data/20180606-SLP_Multiplexing_p1_C03_x000_y000_z000_t000_2_DAPI_Label14.png',
        0)
    cell_mask = cv2.imread(
        '/Users/Joel/p-body-randomness/data/input_data/20180606-SLP_Multiplexing_p1_C03_x000_y000_z000_t000_segmentation_Label14.png',
        0)
    from extract_sample_areas import extract_sample_area
    sample_area = extract_sample_area(cell_mask, img_dapi, 10)
    print(extract_centroids_in_sample_area(pbodies_img, sample_area, 5))
