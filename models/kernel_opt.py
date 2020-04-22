"""
    This file contains an implementation of our method
    With Baysian optimization and agreement of observed constraints
"""
import numpy as np
import utils.constraint as constraint
from utils.utils import print_verbose
from utils.optimizer import CombinationKernelOptimizer

def kernel_clustering(kernels, clustering, constraint_matrix, 
        optimizer_type = "model", kernel_approx = False, verbose = 0, **args_optimizer):
    """
        Constraint clustering with kernel combination
        With maximization of the KTA score on the observed constraints computed with Kmeans

        Arguments:
            kernels {List Array n * d} -- List of the precomputed kernels 
            classes {int} -- Number of clusters to form
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points
            optimizer_type {string} -- Optimizer to use (default: 'model' (model guided optimization))
            kernel_approx {bool} -- Provided kernels are approximation (default: False)

        Keyword Arguments:
            verbose {int} -- Level of verbosity (default: {0} -- No verbose)
            args_optimizer {option} -- Option to use for optimizer intialization

        Returns:
            Assignation of length n, Assignation with enforced constraints
    """
    def compute_assignation(weights):
        # Compute new kernel
        if kernel_approx:
            kernel = np.hstack([w * k for k, w in zip(kernels, weights)])
        else:
            kernel = np.sum(w * k for k, w in zip(kernels, weights))
            
        # Computation assignation
        assignation = clustering.fit_transform(kernel)
        return assignation
    
    # Computes initial score of each kernels
    init_candidates = np.eye(len(kernels))
    
    # Create optimizer
    optimizer = CombinationKernelOptimizer.create(optimizer_type, verbose = verbose, init_candidates = init_candidates, 
                                                  objective_function = lambda weights: constraint.kta_score(constraint_matrix, compute_assignation(weights)),
                                                  dimensionality = len(kernels), **args_optimizer)
    
    best_weights = optimizer.run_optimization()
    
    # Compute final assignation for best candidate
    if (best_weights > 0).sum() > 1:
        print_verbose("Combination better than one : {}".format(best_weights), verbose)

    return compute_assignation(best_weights)