import numpy as np
from utils import print_verbose
from models.farthest_kmeans import Initialization, kernelKmeans

def hmrf(data, classes, constraint_matrix):
    """
        Arguments:
            data {Array n * d} -- Data
            classes {int} -- Number of clusters to form
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points

        Returns:
            Assignation of length n
    """
    return None