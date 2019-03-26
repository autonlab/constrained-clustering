import numpy as np
from models.farthest_kmeans import Initialization, kernelKmeans
from models.kernel_bayes_opt import compute_KTA
from KernelConstrainedKmeans.wkckmeans import weightedKernelConstrainedKmeans

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
        assignment[i] = initializer.farthest_initialization(kernel, classes)

        # Kmeans step
        assignment[i] = kernelKmeans(kernel, assignment[i], max_iteration = 100)
        observed_constraint = 2 * np.equal.outer(assignment[i], assignment[i]) - 1.0
        kernel_kta[i] = compute_KTA(observed_constraint, constraint_matrix)

    best_i = np.argmax(kernel_kta)
    initialization = initializer.farthest_initialization(kernels[best_i], classes)
    return assignment[best_i], weightedKernelConstrainedKmeans(kernels[best_i], initialization, constraint_matrix)