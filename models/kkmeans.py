"""
    This file contains an implementation of a kernel kmeans
    and a farthest first optimization
"""
import numpy as np
from utils import print_verbose

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
                distance[k] -= 2*intra_distance[k][i]/number[k]

            assignation[i] = min(distance, key=lambda d: float(distance[d]))
            if previous != assignation[i]:
                change = True
            approx_error += float(distance[assignation[i]])

        print_verbose("Iteration Assignation {} - {}".format(iteration, approx_error), verbose)
        iteration += 1

    return assignation