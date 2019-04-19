import numpy as np
from models.kkmeans import kernelKmeans
from KernelConstrainedKmeans.initialization import Initialization
from KernelConstrainedKmeans.wksckmeans import weightedKernelSoftConstrainedKmeans

def crossValidation(kernels, classes, constraint_matrix, folds = 5):
    """
        Compute a cross validation for maximizing the kernel alignment
        Between linked predicted and ground truth
        
        Arguments:
            kernels {List Array n * n} -- List of the precomputed kernels
            classes {int} -- Number of clusters to form
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points

        Keyword Arguments:
            folds {int} -- Number of fold (default: {3})

        Returns:
            Assignation of length n, Assignation with enforced constraints
    """
    # Compute the components implied by constrained (without any distance)
    initializer = Initialization(classes, constraint_matrix)

    # Computes initial score of each kernels
    kernel_kta, assignment = np.zeros(len(kernels)), {}
    for i, kernel in enumerate(kernels):
        # Farthest assignation given current distance
        assignment[i] = initializer.farthest_initialization(kernel)

        # Kmeans step
        assignment[i] = kernelKmeans(kernel, assignment[i], max_iteration = 100)
        observed_constraint = 2 * np.equal.outer(assignment[i], assignment[i]) - 1.0
        kernel_kta[i] = np.dot(observed_constraint.ravel(), constraint_matrix.ravel())

    best_i = np.nanargmax(kernel_kta)
    initialization = initializer.farthest_initialization(kernels[best_i])
    ksckmeans = weightedKernelSoftConstrainedKmeans(kernels[best_i], initialization, constraint_matrix)

    return assignment[best_i], ksckmeans