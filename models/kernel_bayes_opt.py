"""
    This file contains an implementation of our method
    With Baysian optimization and agreement of observed constraints
"""
import numpy as np
import constraint
from scipy.sparse import triu as striu
from utils import print_verbose
from sklearn.cluster import KMeans
from models.kkmeans import kernelKmeans
from GPyOpt.methods import BayesianOptimization
from KernelConstrainedKmeans.wksckmeans import weightedKernelSoftConstrainedKmeans
from KernelConstrainedKmeans.initialization import Initialization, InitializationScale

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
        print_verbose("Reduce combinations to the total number of kernel", verbose)
        kernel_components = len(kernels) - 1

    # Compute the components implied by constrained (without any distance)
    initializer = Initialization(classes, constraint_matrix)

    # Computes initial score of each kernels
    kernel_kta = np.zeros(len(kernels))
    for i, kernel in enumerate(kernels):
        # Farthest assignation given current distance
        assignment = initializer.farthest_initialization(kernel)

        # Kmeans step
        assignment = kernelKmeans(kernel, assignment, max_iteration = 100, verbose = verbose)
        kernel_kta[i] = constraint.kta_score(constraint_matrix, assignment)
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
                assignment = initializer.farthest_initialization(kernel)
                assignations[step] = kernelKmeans(kernel, assignment, max_iteration = 100, verbose = verbose)
                
                # Computation score on observed constraints
                kta = constraint.kta_score(constraint_matrix, assignations[step])
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
    initialization = initializer.farthest_initialization(kernel)
    ksckmeans = weightedKernelSoftConstrainedKmeans(kernel, initialization, constraint_matrix)

    return assignations[np.nanargmin(myBopt.Y)], ksckmeans

def kernel_csc_clustering(kernels, classes, constraint_matrix, learnrate1 = 0.05, learnrate2 = 0.01,
    numb_nonzero_max = 5, iterations = 1000, verbose = 0):
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
            Assignation of length n
    """
    # Compute the components implied by constrained (without any distance)
    initializer = InitializationScale(classes, constraint_matrix)

    def objective_custom(weights):
        """
            Computes score for the combinations of kernels implied by weights
        """
        if weights.sum() == 0:
            # Avoid case where all weight are null
            return 0.0
        else:
            # Compute new kernel
            kernel_approx = np.hstack([weight * kernel for kernel, weight in zip(kernels, weights) if weight > 0.0])

            # Computation assignation
            farthest_init = initializer.farthest_initialization(kernel_approx)
            if farthest_init is None:
                farthest_init = 'k-means++'
                n_init = 10
            else:
                n_init = 1
            assignation = KMeans(n_clusters = classes, init = farthest_init, n_init = n_init, algorithm = 'elkan').fit(kernel_approx).labels_

            kta = constraint.kta_score(constraint_matrix, assignation)
            return kta

    # Computes initial score of each kernels
    best_weights = None
    kernel_proba = np.zeros(len(kernels))
    for i in range(len(kernels)):
        # Create weight vector
        weights = np.zeros(len(kernels))
        weights[i] = 1.

        # Evaluate performances
        kernel_proba[i] = objective_custom(weights)

        print_verbose("Initial assignation kernel {} - KTA {}".format(i, kernel_proba[i]), verbose)

        # Stop for perfect clustering
        if kernel_proba[i] == 1.:
            print_verbose("Perfect Classification", verbose)
            best_weights = weights

    # Optimization if not best found
    if best_weights is None:
        # Proba distribution of components number
        number_component_proba = np.ones(numb_nonzero_max)
        number_component_proba[:2] = 0 

        # Initialize best kta
        best_kta = kernel_proba.max()
        weights = np.zeros(len(kernels))
        weights[kernel_proba.argmax()] = 1.
        best_weights = weights
        for iteration in range(iterations):
            # Normalize proba
            number_component_proba /= number_component_proba.sum()

            if number_component_proba.sum() < 0.00001 or kernel_proba.sum() < 0.00001:
                break

            # Draw weight
            number_components = min(np.random.choice(numb_nonzero_max, p = number_component_proba), np.count_nonzero(kernel_proba))
            indices = np.random.choice(len(kernels), number_components, p = kernel_proba / kernel_proba.sum(), replace = False)
            weights = [np.clip(np.random.normal(kernel_proba[i], 0.1), 0, 1) for i in indices]

            # Create a weight vector with zeros
            beta = np.zeros(len(kernels))
            beta[indices] = weights

            # Compute performances
            kta = objective_custom(beta)

            # Update Probability
            number_component_proba[number_components] = np.clip(number_component_proba[number_components] + learnrate2 * (kta - best_kta), 0, 1)
            kernel_proba[indices] = np.clip(kernel_proba[indices] + learnrate1 * (kta - best_kta), 0, 1)

            print_verbose("Step {}".format(iteration), verbose, level = 1)
            print_verbose("\t Weights  : {}".format(weights), verbose, level = 1)
            print_verbose("\t KTA  : {}".format(kta), verbose)

            # Update best performance
            if kta > best_kta:
                best_kta = kta
                best_weights = weights

                # Perfect solution
                if best_kta == 1.:
                    break

    # Do final assignment
    kernel_approx = np.hstack([weight * kernel for kernel, weight in zip(kernels, best_weights) if weight > 0.0])

    # Computation assignation
    n_init = 1
    farthest_init = initializer.farthest_initialization(kernel_approx)
    if farthest_init is None:
        farthest_init = 'k-means++'
        n_init = 10
    assignation = KMeans(n_clusters = classes, init = farthest_init, n_init = n_init, algorithm = 'elkan').fit(kernel_approx).labels_

    return assignation