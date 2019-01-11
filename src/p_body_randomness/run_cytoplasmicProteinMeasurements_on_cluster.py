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
from p_body_randomness.centroids import extract_centroids, extract_centroids_in_sample_area, generate_centroids_mask
from p_body_randomness.metrics import nearest_neighbor_distance
from p_body_randomness.smooth_image import smooth_protein_image
from p_body_randomness.protein_signal_calculation import calculate_mean_intensities

# Divide the dataset into the different wells, this script runs one well
# (depending on the input value it gets). This allows it to run on the
# Pelkmanslab slurm cluster

# Parameters
pbody_area_threshold = 5
dapi_threshold = 10
min_nb_pbodies = 3
percentage_pbodies_in_nucleus = 0.07388 # Percentage of P-bodies that should be sampled over the nucleus

wells = ['C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
         'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16',
         'E04', 'E05', 'E06', 'E07', 'E08', 'F04', 'F05', 'F06', 'F07', 'F08']
dim_x = 6
dim_y = 6

image_types = {'pbodies': '13_Pbody_Segm', 'protein': '13_Succs', 'cellmask': 'segmentation', 'dapi': '2_DAPI'}
subfolders = {'pbodies': 'singleCellImages_PbodySegmentation', 'protein': 'singleCellImages', 'cellmask': 'singleCellSegmentations', 'dapi': 'singleCellImages'}
base_path = '/data/active/jluethi/20180503-SubcellularLocalizationMultiplexing'
output_path = '/data/homes/jluethi/20190109-pbodies-random-distribution-test/run4_protein_measurements'
TEMPLATE_FILENAME = '20180606-SLP_Multiplexing_p1_C03_x00{x}_y00{y}_z000_t000_{image_type}_Label{label}.png'
total_file_list = os.listdir(os.path.join(base_path, subfolders['cellmask']))

class TooFewPbodiesException(Exception):
    pass

def evaluate_site(well, site_x, site_y, label):
    print('Evaluating site ' + well + ': ' + str(site_x) + str(site_y) + ', Label ', label)
    # load images
    pbodies_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['pbodies'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['pbodies'], label = label))), 0)
    dapi_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['dapi'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['dapi'], label = label))), 0)
    cellmask_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['cellmask'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['cellmask'], label = label))), 0)
    protein_image = cv2.imread(os.path.join(base_path,os.path.join(subfolders['protein'],TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['protein'], label = label))), 0)

    cytoplasmic_mask = extract_sample_area(cellmask_image, dapi_image, dapi_threshold)
    #[centroids, centroids_in_nucleus] = extract_centroids_in_sample_area(pbodies_image, cytoplasmic_mask, pbody_area_threshold)
    centroids = extract_centroids(pbodies_image, pbody_area_threshold)
    number_of_pbodies = len(centroids)

    # Nearest neighbors can only be calculate if there are at least 3 P-bodies
    if number_of_pbodies >= min_nb_pbodies:
        # Calculate area of cytoplasm of the cell
        cytoplasmic_area = np.sum(np.sum(cytoplasmic_mask > 1))

        # Get a smoothed Succs image to estimate the volume of the cell
        smooth = smooth_protein_image(protein_image, pbodies_image, cytoplasmic_mask, dapi_image, nucleus_percentage = percentage_pbodies_in_nucleus, nucleus_threshold = dapi_threshold)

        mean_of_multiple_simulation_rounds = []
        # Run multiple simulations, take the average
        for i in range(100):
            sampled_pbodies = sample_pbodies(smooth, number_of_pbodies, area_fn=lambda: 40)
            sampled_pbodies_mask = generate_centroids_mask(sampled_pbodies)
            [, mean_protein_intensity_pbodies_simulated] = calculate_mean_intensities(protein_image, sampled_pbodies_mask, cellmask_image, dapi_image)
            mean_of_multiple_simulation_rounds.append(mean_protein_intensity_pbodies_simulated)

        # Calculate the mean protein intensity around the actual P-bodies
        [mean_protein_intensity_cytoplasm, mean_protein_intensity_pbodies] = calculate_mean_intensities(protein_image, pbodies_image, cellmask_image, dapi_image)

        return [number_of_pbodies, mean_protein_intensity_cytoplasm, mean_protein_intensity_pbodies, np.mean(mean_of_multiple_simulation_rounds), cytoplasmic_area]

    else:
        raise TooFewPbodiesException("This cell only has " + str(number_of_pbodies) + " P-bodies.")


# Read in the input from command line about which site to work on
index = int(sys.argv[1])
well = wells[index]
columns = ['Well', 'SiteX', 'SiteY', 'Label','Number_of_pbodies', 'Mean_protein_intensity_cytoplasm', 'Mean_protein_intensity_around_pbodies', 'Mean_protein_intensity_around_simulated_pbodies','Area_of_Cytoplasm']
measurements = pd.DataFrame()
for site_x in range(dim_x):
    for site_y in range(dim_y):
        # print('Evaluating site ' + well + ': ' + str(site_x) + str(site_y))
        # Get all the labels for a given site
        regex = re.compile(TEMPLATE_FILENAME.format(x = site_x, y = site_y, image_type=image_types['cellmask'], label = '.*'))
        site_file_list = list(filter(regex.match, total_file_list))
        label_list = []
        for filename in site_file_list:
            label_list.append(re.search('.*Label(\d*).png', filename).group(1))
        for label in label_list:
            try:
                results = evaluate_site(well, site_x, site_y, label)
            except TooFewPbodiesException:
                continue
            # Fill results in pandas table
            combined_results = pd.Series([well, site_x, site_y, label] + results, index = columns)
            measurements = measurements.append(combined_results, ignore_index=True)

output_filename = 'Protein_intensities_' + well + '.csv'
measurements.to_csv(os.path.join(output_path, output_filename), na_rep = 'NaN', index=False)
