import utils.constraint as constraint
import numpy as np

def crossValidation(kernels, clustering, constraint_matrix, folds = 5):
    """
        Compute a cross validation for maximizing the kernel alignment
        Between linked predicted and ground truth
        
        Arguments:
            kernels {List Array n * n} -- List of the precomputed kernels
            clustering {Clustering} -- Clustering algo to use
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points

        Keyword Arguments:
            folds {int} -- Number of fold (default: {3})

        Returns:
            Assignation of length n, Assignation with enforced constraints
    """
    # Computes initial score of each kernels
    kernel_kta, assignment = np.zeros(len(kernels)), {}
    for i, kernel in enumerate(kernels):
        assignment[i] = clustering.fit_transform(kernel)
        kernel_kta[i] = constraint.kta_score(constraint_matrix, assignment[i])

    best_i = np.nanargmax(kernel_kta)
    return assignment[best_i], clustering.fit_transform(kernels[best_i], constraint_matrix.toarray())