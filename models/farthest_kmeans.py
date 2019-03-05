"""
    This file contains an implementation of a kernel kmeans
    and a farthest first optimization
"""
import numpy as np
from utils import print_verbose
from scipy.sparse.csgraph import connected_components

def kernelKmeans(kernel, assignation, max_iteration = 100, verbose = 0):
    """
        Compute kernel kmeans
        
        Arguments:
            kernel {Array n*n} -- Kernel
            assignation {Array n} -- Initial assignation 
        
        Keyword Arguments:
            max_iteration {int} -- Maximum iteration (default: {100})
            verbose {int} -- Verbose level (default: {0})
        
        Returns:
            Assignation - Array n
    """
    assignation_cluster, intra_distance, number, base_distance = {}, {}, {}, {}
    index = np.arange(len(assignation))
    clusters = np.unique(assignation)
    iteration, change = 0, True

    while change and iteration < max_iteration:
        change, approx_error = False, 0.
        np.random.shuffle(index)

        # Update cluster centers
        for k in clusters:
            assignation_cluster[k] = (assignation == k).reshape((-1,1))
            intra_distance[k] = np.matmul(kernel, assignation_cluster[k])
            number[k] = np.sum(assignation_cluster[k])
            base_distance[k] = np.dot(assignation_cluster[k].T, intra_distance[k])/(number[k]**2)

        for i in index:
            previous = assignation[i]
            distance = {k: float(base_distance[k]) for k in clusters}
            for k in clusters:
                # Only this term implies a change if center unupdated
                distance[k] += kernel[i,i] - 2*intra_distance[k][i]/number[k]
                assignation[i] = k

            assignation[i] = min(distance, key=lambda d: float(distance[d]))
            if previous != assignation[i]:
                change = True
            approx_error += float(distance[assignation[i]])

        print_verbose("Iteration Assignation {} - {}".format(iteration, approx_error), verbose)
        iteration += 1

    return assignation

class Initialization:
    """
        This object precompute the main components implied by constraints
    """

    def __init__(self, k, constraint):
        """
            Precompute connected components
            Arguments:
                k {int} -- Number of cluster
                constraint {Array n * n} -- Constraint matrix with value in (-1, 1)
                    Positive values are must link constraints
                    Negative values are must not link constraints
        """
        assert constraint is not None, "Farthest initialization cannot be used with no constraint"
        # Computes the most important components and order by importance
        positive = np.where(constraint > 0, constraint, np.zeros_like(constraint))
        self.number, components = connected_components(positive, directed=False)
        unique, count = np.unique(components, return_counts = True)
        order = np.argsort(count)[::-1]
        self.components = np.argsort(unique[order])[components] 
        self.constraint = constraint
        assert self.number >= k, "Constraint too important for number of cluster"
        
    def farthest_initialization(self, kernel, k):
        """
            Farthest points that verify constraint

            Arguments:
                kernel {Array n * n} -- Kernel matrix (n * n)
                k {Int} --  Number cluster
                constraint {Array n * n} -- Constraint matrix
        """
        components = self.components.copy()

        # Precompute center distances
        assignation_cluster, intra_distance, intra_number = {}, {}, {}
        for c in range(k):
            assignation_cluster[c] = (components == c).reshape((-1,1))
            intra_distance[c] = np.matmul(kernel, assignation_cluster[c])
            intra_number[c] = np.sum(assignation_cluster[c])

        # Merge components respecting constraint until # = k
        for i in range(k, self.number):
            # Computes intra distance
            assignation_cluster[i] = (components == i).reshape((-1,1))
            intra_distance[i] = np.matmul(kernel, assignation_cluster[i])
            intra_number[i] = np.sum(assignation_cluster[i])

            # Computes distances to all other cluster 
            # We ignore the last part which depends on the intravariance of the cluster i
            distance = [float(np.dot(assignation_cluster[c].T, intra_distance[c])/(intra_number[c]**2) 
                - 2 * np.dot(assignation_cluster[i].T, intra_distance[c])/(intra_number[c] * intra_number[i]))
                for c in range(k)]

            # Closest verifying constraint
            order = np.argsort(distance)

            # If no constraint is positive => Too much constraint
            broken_constraint = self.constraint[:, assignation_cluster[i].flatten()]
            closest = min(order, key=lambda o: np.sum(broken_constraint[(components == o),:] < 0))
            components[assignation_cluster[i].flatten()] = closest

            # Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            intra_distance[closest] += intra_distance[i]
            intra_number[closest] += intra_number[i]

        return components
