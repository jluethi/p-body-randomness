'''
Calculate different metrics.
'''
import numpy as np


def eucledian_distance(coordinates1: tuple, coordinates2: tuple):
    '''
    Function: calculates Eucledian distance.
    Input:    two lists of coordinates of equal length
    Output:   the calculated distance.
    '''
    distanceSum = 0
    for i in range(len(coordinates1)):
        distanceSum += (coordinates1[i] - coordinates2[i])**2
    return np.sqrt(distanceSum)


def nearest_neighbor_distance(pBodylist: list):
    '''
    Function: for each p-body in a list, calculates the distance to its nearest neighbor.
    Input:    a list of tuples, where a tuple is of format: (xCoord, yCoord, area) for each p-body.
    Output:   a list of distances to nearest neighbors.
    '''

    # Creating a distance matrix.
    pairwiseDistances = np.zeros((len(pBodylist), len(pBodylist)))

    # Going through all p-bodies one by one.
    for i in range(len(pBodylist)-1):
        for j in range(i+1, len(pBodylist)):
            # Calculating Eucledian distance.
            distance = eucledian_distance(pBodylist[i][:2], pBodylist[j][:2])
            # Filling up distance matrix.
            pairwiseDistances[i, j] = distance
            pairwiseDistances[j, i] = distance

    # List containig all minimal distances.
    nearestNeighbors = []
    # Selecting minimum distances.
    for pBody in pairwiseDistances:
        nearestNeighbors.append(min(pBody[pBody > 0]))

    # Returning list of distances to the nearest neighbors.
    return nearestNeighbors
