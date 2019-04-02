"""
    This file contains an implementation of our method
    With Baysian optimization and agreement of observed constraints
"""
import numpy as np
from utils import print_verbose
from GPyOpt.methods import BayesianOptimization
from models.farthest_kmeans import Initialization, kernelKmeans
from KernelConstrainedKmeans.wkckmeans import weightedKernelConstrainedKmeans

def compute_KTA(A, B):
    """
        Compute Kernel Target Alignment between matrices A and B 
    
        Arguments:
            A {array n * n} -- a Gram matrix
            B {array n * n} -- a Gram matrix
        
        Returns:
            KTA {float}
    """
    return np.dot(A.ravel(),B.ravel())

# TODO: Change this function to an object that has a fit and transform and labels attributes
def kernel_bayes_clustering(kernels, classes, constraint_matrix, kernel_components = 3, bayes_iter = 1000, verbose = 0):
    """
        Bayesian optimization on the space of combinasions of the given kernels
        With maximization of the KTA score on the observed constraints computed with Kmeans

        Arguments:
            kernels {List Array n * n} -- List of the precomputed kernels
            classes {int} -- Number of clusters to form
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points

        Keyword Arguments:
            kernel_components {int} -- Number of kernel to combine simultaneously (default: {3})
            bayes_iter {int} -- Number of iteration to compute on the space (default: {1000})
                NB: Higher this number slower is the algorithm
            verbose {int} -- Level of verbosity (default: {0} -- No verbose)

        Returns:
            Assignation of length n, Assignation with enforced constraints
    """
    if kernel_components >= len(kernels):
        print_verbose("Reduce combiantions to the total number of kernel", verbose)
        kernel_components = len(kernels) - 1

    # Compute the components implied by constrained (without any distance)
    initializer = Initialization(classes, constraint_matrix)

    # Computes initial score of each kernels
    kernel_kta = np.zeros(len(kernels))
    for i, kernel in enumerate(kernels):
        # Farthest assignation given current distance
        assignment = initializer.farthest_initialization(kernel, classes)

        # Kmeans step
        assignment = kernelKmeans(kernel, assignment, max_iteration = 100, verbose = verbose)
        observed_constraint = 2 * np.equal.outer(assignment, assignment) - 1.0
        kernel_kta[i] = compute_KTA(observed_constraint, constraint_matrix)
        print_verbose("Initial assignation kernel {} - KTA {}".format(i, kernel_kta[i]), verbose)

        # Stop for perfect clustering
        if kernel_kta[i] == np.abs(constraint_matrix).sum():
            print_verbose("Perfect Classification", verbose)
            return assignment, assignment

    # Select best kernel
    best_i = np.argmax(kernel_kta)
    beta_best = np.zeros(len(kernels))
    beta_best[best_i] = 1.0

    # We put these global variables to not recompute the assignation
    # And also to overcome the randomness of the kmeans step
    global assignations, kernelsBeta, step
    assignations, kernelsBeta, step = {}, {}, 0
    def objective_KTA_sparse(subspace):
        """
            Computes the KTA for the combinations of kernels implied by
        """
        global assignations, kernelsBeta, step
        kta_score = []
        for baysian_beta in subspace:
            # Constraints are return in same order than space
            indices = np.array([best_i] + [int(i) for i in baysian_beta[kernel_components:]])
            weights = baysian_beta[:kernel_components]

            if weights.sum() == 0:
                # Avoid case where all weight are null
                kta_score.append(- np.inf)
            else:
                weights /= weights.sum()

                # Compute new kernel
                kernel = np.sum(kernels[i] * w for i, w in zip(indices, weights))
                
                # Computation assignation
                assignment = initializer.farthest_initialization(kernel, classes)
                assignations[step] = kernelKmeans(kernel, assignment, max_iteration = 100, verbose = verbose)
                
                # Computation score on observed constraints
                observed_constraint = 2 * np.equal.outer(assignations[step], assignations[step]) - 1.0
                kta = compute_KTA(observed_constraint, constraint_matrix)
                kta_score.append(kta)

                # For memory efficiency: Add kernel only if better than before
                if np.greater_equal(kta, kta_score).all():
                    kernelsBeta[step] = kernel

                print_verbose("Step {}".format(step), verbose, level = 1)
                print_verbose("\t Kernels  : {}".format(indices), verbose, level = 1)
                print_verbose("\t Weights  : {}".format(weights), verbose, level = 1)
                print_verbose("\t KTA  : {}".format(kta_score[-1]), verbose)
            step += 1

        return - np.array(kta_score).reshape((-1, 1))
    
    # Weights on the different kernels
    # var1 = 0.5 means that the baysian optim puts a weight of 0.5 on the 4th kernel
    space =  [ {'name': 'var_{}'.format(j), 
                'type': 'continuous', 
                'domain': (0, 1.)}
            for j in range(kernel_components)]
    # Use of the different kernels 
    # var1 = 4 means that the baysian optim uses the 4 th kernel
    space += [ {'name': 'var_{}'.format(j), 
                'type': 'discrete', 
                'domain': np.array([i for i in range(len(kernels)) if i != best_i])}
            for j in range(1, kernel_components)]
    # It provides a weight for kernel_components kernels with enforcing to use the
    # one that performs the best at first (var0 contains only its weight)

    myBopt = BayesianOptimization(f = objective_KTA_sparse, 
        de_duplication = True,  # Avoids re-evaluating the objective at previous, pending or infeasible locations 
        domain = space)         # Domain to explore
    
    myBopt.run_optimization(max_iter = bayes_iter)

    kernel = kernelsBeta[np.nanargmin(myBopt.Y)]
    initialization = initializer.farthest_initialization(kernel, classes)
    return assignations[np.nanargmin(myBopt.Y)], weightedKernelConstrainedKmeans(kernel, initialization, constraint_matrix)