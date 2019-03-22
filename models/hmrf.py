from utils import print_verbose
from models.farthest_kmeans import Initialization
from scipy.optimize import minimize
import numpy as np

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize

# TODO: Clean those classes and remove them if possible
class Assignment_HMRFkmeans:

    def __init__(self, error, max_iteration = 100):
        self.error = error
        self.max_iteration = max_iteration

    def compute(self, avec, X, assignation, verbose=0):
        """
            Order of point is important in Kernel Kmeans under constraint
            Compute the center only once and observe the impact of moving
            one point on the constraint

            Cluster centers are fixed.

            Arguments:
                assignation {array n } -- Assignation of each points
                kernel {array n * n} -- Kernel matrix
        """
        # Computes characteristics of each cluster
        index = np.arange(len(assignation))
        clusters = np.unique(assignation)
        num_clusters = clusters.size
        change = np.ones(len(assignation), dtype=bool)
        iteration = 0

        # first fix cluster centers and precompute projections.
        centers = np.zeros((clusters.size, X.shape[1]))
        for k in clusters:
            centers[k] = X[assignation == k].mean(0)

        Xproj = np.multiply(X, np.sqrt(avec))
        normalize(Xproj, copy=False)
        Y = np.multiply(centers, np.sqrt(avec))
        normalize(Y, copy=False)
        while np.any(change) and iteration < self.max_iteration:
            change, approx_error = np.zeros(len(assignation), dtype=bool), 0.0
            np.random.shuffle(index)

            for i in index:
                previous = assignation[i]
                distance = {k: 0.0 for k in clusters}
                for k in clusters:
                    assignation[i] = k
                    distance[k] = self.error.compute_error_assignment(Xproj, Y, assignation, clusters)

                assignation[i] = min(distance, key=lambda d: float(distance[d]))
                if np.unique(assignation).size != num_clusters:
                    #don't leave cluster empty
                    assignation[i] = previous
                change[i] = previous != assignation[i]
                approx_error += float(distance[assignation[i]])
            
            print_verbose("Iteration {} - {}".format(iteration, approx_error), verbose, end='\r')
            iteration += 1

        return assignation


class HMRFkmeans_Error:

    def __init__(self,X,constraint_matrix):
        self.Centers = None
        self.masks = None
        self.idxs_mustl_violated = None
        self.idxs_cannotl_violated = None
        self.constraint_matrix = constraint_matrix
        self.must_link_idxs = constraint_matrix > 0
        self.not_link_idxs = constraint_matrix < 0
        self.Xsquared = X**2

    def update_centers_masks(self, X, assignation):
        # TODO: Rename this variable
        ulbls = np.unique(assignation)
        self.Centers = np.zeros((ulbls.size,X.shape[1]))
        for k in ulbls:
            self.Centers[k] = X[assignation == k].mean(0)
        self.masks = [assignation == k for k in ulbls]
        #precompute indices for penalty terms
        observed_constraint = 2 * np.equal.outer(assignation, assignation) - 1  # -1 different cluster, 1 same cluster
        comparison_assignation = np.multiply(observed_constraint,self.constraint_matrix)
        #set upper triangular to zero so that we don't double count (i,j) as (j,i)
        comparison_assignation[np.triu_indices(X.shape[0],1)]=0
        self.idxs_mustl_violated = np.where(np.multiply(comparison_assignation<0,self.must_link_idxs))
        self.idxs_cannotl_violated = np.where(np.multiply(comparison_assignation<0,self.not_link_idxs))

    # TODO: Merge those functions: Why some are global and other not ?
    def compute_error(self,avec,X):
        '''
        compute error. Some constants are ignored
        '''
        Xproj = np.multiply(X, np.sqrt(avec))
        normalize(Xproj, copy=False)
        Y = np.multiply(self.Centers, np.sqrt(avec))
        normalize(Y, copy=False)
        res = 0.0
        for i, idxs in enumerate(self.masks):
            res -= np.dot(Xproj[idxs], Y[i]).sum()
        
        # constraints
        # TODO: Explain that
        if self.idxs_mustl_violated[0].size > 0:
            res += (1.1-np.einsum('ij,ij->i', Xproj[self.idxs_mustl_violated[0]], Xproj[self.idxs_mustl_violated[1]])).sum()
        if self.idxs_cannotl_violated[0].size > 0:
            res += (2.0-(1.0-np.einsum('ij,ij->i', Xproj[self.idxs_cannotl_violated[0]], Xproj[self.idxs_cannotl_violated[1]]))).sum()
        return float(res)
            

    def compute_error_assignment(self, Xproj, Y, assignation, clusters):
        '''
        compute error. Some constants are ignored
        '''
        res = 0.0
        for k in clusters:
            idxs = assignation == k
            res -= np.dot(Xproj[idxs], Y[k]).sum()

        # constraints
        observed_constraint = 2 * np.equal.outer(assignation, assignation) - 1  # -1 different cluster, 1 same cluster
        comparison_assignation = np.multiply(observed_constraint, self.constraint_matrix)

        # set upper triangular to zero so that we don't double count (i,j) as (j,i)
        comparison_assignation[np.triu_indices(Xproj.shape[0], 1)] = 0
        idxs_mustl_violated = np.where(np.multiply(comparison_assignation < 0, self.must_link_idxs))
        idxs_cannotl_violated = np.where(np.multiply(comparison_assignation < 0, self.not_link_idxs))
        
        if idxs_mustl_violated[0].size > 0:
            res += (1.1-np.einsum('ij,ij->i', Xproj[self.idxs_mustl_violated[0]], Xproj[self.idxs_mustl_violated[1]])).sum()
        if idxs_cannotl_violated[0].size > 0: 
            res += (2.0-(1.0-np.einsum('ij,ij->i', Xproj[idxs_cannotl_violated[0]], Xproj[idxs_cannotl_violated[1]]))).sum()
        return float(res)


def convergence_eps(previous, error, iteration, eps=1e-05, maxiteration=100):
    """
        Criterion of convergence

        Arguments:
            previous {Float or None} -- Error at the previous step
            error {Float or None} -- Current error

        Returns:
            Boolean -- Algo has converged
    """
    if previous is None or error is None:
        return True
    return previous - error > eps and iteration < maxiteration


def hmrf_kmeans_cosine(data, k, constraint_matrix,
                        verbose = 0, a_init = None, par_kernel_jobs = 1,
                        init_args = {}, convergence_args = {} ,
                        solveroptions={'disp': None, 'iprint': -1, 'gtol': 1e-07, 'eps': 1e-08,
                                       'maxiter': 10, 'ftol': 2.220446049250313e-09, 'maxcor': 10,
                                       'maxfun': 15000}):
    """
        Constrained  K-Means with parametrized cosine distance
        Optimization of a parametrized cosine distance
        In order to minimize the variance of cluster (kmeans)
        Under constraints (semi supervised learning)

        Arguments:
            data {np array} -- data matrix (n * d)
            k {Int} -- Number of cluster

        Keyword Arguments:
            constraint_matrix {sparse array} -- Constrained on data points
                +1 => Constraint the points to be in the same cluster
                -1 => Constraint the points to be in separate clusters
                (default: {None : No Constraint => Unsupervised clustering})
                Diagonal => Zeros for avoiding sef assignation penalty
            completion_matrix {bool} -- Complete the constrained matrix
                by consistency (default: {False})
            warm_start {bool} -- Reuse the previous E step for the maximisation (default: {True})
            verbose{int}
                -> 0 -- No output except error
                -> 1 -- Display comment
                -> 2 -- Display step error
            a_init {array} -- Initial parameters for diagonal psd matrix A

        Returns:
            array n -- Assignation of each point
            array k -- Weight for each kernel
            float -- Final error
    """
    # Initialization
    if a_init is None:
        avec = np.ones(data.shape[1])
        kernel = pairwise_kernels(data, metric='cosine', n_jobs=par_kernel_jobs)
    else:
        if np.alltrue(a_init>=0):
            avec = a_init
            kernel = pairwise_kernels(np.multiply(data, np.sqrt(avec)), metric='cosine', n_jobs=par_kernel_jobs)
        else:
            raise ValueError("resulting A not psd")
    
    iteration = 0
    bounds = [[ 1.0e-07, 1.0]] * len(avec)  # setting bounds to eps or 0 may create instability in solver.
    previous, error_current = None, None
    best_error, best_assignation = None, None

    ## Error to optimize
    error = HMRFkmeans_Error(data, constraint_matrix)

    ## Assignation
    assignation = Assignment_HMRFkmeans(error)
    assignation_current = Initialization(k, constraint_matrix).farthest_initialization(kernel, k)
    error.update_centers_masks(data, assignation_current)
    print_verbose("Initial assignation : {}".format(assignation_current), verbose)

    while convergence_eps(previous, error_current, iteration, **convergence_args):
        previous = error_current

        ## M step : Minimize the error 
        ### Use last avec
        result = minimize(error.compute_error, avec, args=(data),
                           bounds=bounds,
                           options =solveroptions)

        print_verbose(result, verbose, level=1)
        avec = result.x

        ## E step : Assign each points to its cluster
        assignation_current = assignation.compute(avec, data, assignation_current)
        error.update_centers_masks(data, assignation_current)
        error_current = error.compute_error(avec, data)

        iteration += 1
        print_verbose("Iteration: {} - Observed error: {}".format(iteration, error_current), verbose)
        print_verbose("Assignation : {}".format(assignation_current), verbose)
        print_verbose("avec : {}".format(avec), verbose)

        ## Update error and beta
        if best_error is None or error_current < best_error:
            best_error, best_assignation = error_current, assignation_current

    return best_assignation