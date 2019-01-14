import random
from pathlib import Path
from math import sqrt, pi
import os
import re
import pandas as pd
import sys

import numpy as np
import cv2

from p_body_randomness.extract_sample_areas import extract_sample_area
from p_body_randomness.sampling import sample_pbodies
from p_body_randomness.centroids import extract_centroids
from p_body_randomness.metrics import nearest_neighbor_distance

# Script that calculates all the nearest neighbor distances of P-bodies in all
# cells. Used to define minimal observed nearest neighbor distance

# Parameters
pbody_area_threshold = 5
dapi_threshold = 10
min_nb_pbodies = 3

wells = ['C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
         'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16',
         'E04', 'E05', 'E06', 'E07', 'E08', 'F04', 'F05', 'F06', 'F07', 'F08']
dim_x = 6
dim_y = 6

image_types = {'pbodies': '13_Pbody_Segm', 'protein': '13_Succs', 'cellmask': 'segmentation', 'dapi': '2_DAPI'}
subfolders = {'pbodies': 'singleCellImages_PbodySegmentation', 'protein': 'singleCellImages', 'cellmask': 'singleCellSegmentations', 'dapi': 'singleCellImages'}
base_path = '/data/active/jluethi/20180503-SubcellularLocalizationMultiplexing'
output_path = '/data/homes/jluethi/20190109-pbodies-random-distribution-test/run5_all_individual_nn_distances'
TEMPLATE_FILENAME = '20180606-SLP_Multiplexing_p1_C03_x00{x}_y00{y}_z000_t000_{image_type}_Label{label}.png'
total_file_list = os.listdir(os.path.join(base_path, subfolders['cellmask']))

def evaluate_site(well, site_x, site_y, label):
    print('Evaluating site ' + well + ': ' + str(site_x) + str(site_y) + ', Label ', label)
    # load images
    pbodies_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['pbodies'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['pbodies'], label = label))), 0)
    dapi_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['dapi'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['dapi'], label = label))), 0)
    cellmask_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['cellmask'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['cellmask'], label = label))), 0)
    cytoplasmic_mask = extract_sample_area(cellmask_image, dapi_image, dapi_threshold)
    centroids = extract_centroids(pbodies_image, pbody_area_threshold)
    number_of_pbodies = len(centroids)

    # Nearest neighbors can only be calculate if there are at least 3 P-bodies
    if number_of_pbodies >= min_nb_pbodies:
        # Calculate the real nearest neighbor distances
        real_distances = nearest_neighbor_distance(centroids)

        return real_distances


# Read in the input from command line about which site to work on
index = int(sys.argv[1])
well = wells[index]
columns = ['Well', 'SiteX', 'SiteY', 'Label', 'NearestNeighborDistance']
measurements = pd.DataFrame()
for site_x in range(dim_x):
    for site_y in range(dim_y):
        # Get all the labels for a given site
        regex = re.compile(TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['cellmask'], label = '.*'))
        site_file_list = list(filter(regex.match, total_file_list))
        label_list = []
        for filename in site_file_list:
            label_list.append(re.search('.*Label(\d*).png', filename).group(1))

        for label in label_list:
            results = evaluate_site(well, site_x, site_y, label)
            # Only add results for cells that contain at least min number of P-bodies
            if results:
                nbEntries = len(results)

                # Create a metadata pandas table & combine it with the results table
                metadata_input = pd.Series([well, site_x, site_y, label], index = columns[:-1])
                curr_data = pd.DataFrame()
                curr_data = curr_data.append([metadata_input]*nbEntries, ignore_index=True)
                curr_data[columns[-1]] = results

                measurements = measurements.append(curr_data, ignore_index=True)

output_filename = 'AllNearestNeighborDistances_' + well + '.csv'
measurements.to_csv(os.path.join(output_path, output_filename), na_rep = 'NaN', index=False)
