from utils.utils import print_verbose
from KernelConstrainedKmeans.initialization import Initialization, Euclidean_Initialization

from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from sklearn.preprocessing import normalize
from utils.constraint import indices_constraint_violated
from scipy.optimize import minimize
import numpy as np

def mahalanobis_log_det_a(avec):
    #The determinant of diag(a1, ..., an) is the product a1...an.
    return np.log(avec).sum()

def rayleigh_prior(avec,s=1.0):
    """
    Rayleigh prior over the parameters of the adaptive distortion measure. 
    """
    return np.sum(np.log(avec)-avec**2/(2.0*s**2)-2.0*np.log(s))

# TODO: Clean those classes and remove them if possible
class Assignment_HMRFkmeans:

    def __init__(self, error, metric,max_iteration = 100):
        self.error = error
        self.max_iteration = max_iteration
        self.metric = metric

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

        # first fix cluster centers 
        centers = np.zeros((clusters.size, X.shape[1]))
        for k in clusters:
            centers[k] = X[assignation == k].mean(0)
        # precompute projections.
        Xproj = np.multiply(X, np.sqrt(avec))
        Y = np.multiply(centers, np.sqrt(avec))

        if self.metric == 'cosine':
            normalize(Xproj, copy=False)
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

    def __init__(self,X,metric,constraint_matrix):
        self.Centers = None
        self.masks = None
        self.idxs_mustl_violated = None
        self.idxs_cannotl_violated = None
        self.constraint_matrix = constraint_matrix
        self.dmaxmal = 1e+7
        if metric == 'cosine':
            self.compute_error = self.compute_error_cosine
            self.compute_error_assignment = self.compute_error_assignment_cosine
        else:
            self.compute_error = self.compute_error_mahalanobis
            self.compute_error_assignment = self.compute_error_assignment_mahalanobis

    def update_centers_masks(self, X, assignation):
        # TODO: Rename this variable
        ulbls = np.unique(assignation)
        self.Centers = np.zeros((ulbls.size,X.shape[1]))
        for k in ulbls:
            self.Centers[k] = X[assignation == k].mean(0)
        self.masks = [assignation == k for k in ulbls]
        #precompute indices for penalty terms
        self.idxs_mustl_violated, self.idxs_cannotl_violated = indices_constraint_violated(self.constraint_matrix, assignation)

    # TODO: Merge those functions: Why some are global and other not ?
    def compute_error_cosine(self,avec,X):
        '''
        Compute error to minimize objective in terms of the underlying metric. 
        Some constants are ignored. 
        '''
        Xproj = np.multiply(X, np.sqrt(avec))
        normalize(Xproj, copy=False)
        Y = np.multiply(self.Centers, np.sqrt(avec))
        normalize(Y, copy=False)
        res = 0.0
        for i, idxs in enumerate(self.masks):
            res += (1.1 - np.dot(Xproj[idxs], Y[i])).sum()
        
        # compute violated constraints. 
        # We use einsum to compute a fast product between two matrices row wise.
        # Constant 1.1 used instead of 1.0 due to numerical instability which can lead to the inner product between
        # normalized vectors to be greater than 1. 
        if len(self.idxs_mustl_violated) > 0:
            res += (1.1-np.einsum('ij,ij->i', Xproj[self.idxs_mustl_violated[0]], Xproj[self.idxs_mustl_violated[1]])).sum()
        if len(self.idxs_cannotl_violated) > 0:
            res += (2.0-(1.0-np.einsum('ij,ij->i', Xproj[self.idxs_cannotl_violated[0]], Xproj[self.idxs_cannotl_violated[1]]))).sum()
        # add rayleigh prior over A matrix that parameterizes the metric and return
        return float(res)-rayleigh_prior(avec)

    def compute_error_mahalanobis(self,avec,X):
        '''
        Compute error to minimize objective in terms of the underlying metric. 
        Some constants are ignored. 
        '''
        
        Xproj = np.multiply(X, np.sqrt(avec))
        Y = np.multiply(self.Centers, np.sqrt(avec))
        res = 0.0
        for i, idxs in enumerate(self.masks):
            res += np.square(Xproj[idxs] - Y[i]).sum()
        
        # compute violated constraints. 
        if len(self.idxs_mustl_violated) > 0:
            res += np.square(Xproj[self.idxs_mustl_violated[0]] - Xproj[self.idxs_mustl_violated[1]]).sum()
        if len(self.idxs_cannotl_violated) > 0:
            D = np.square(Xproj[self.idxs_cannotl_violated[0]] - Xproj[self.idxs_cannotl_violated[1]]).sum(1)
            # clip so none are below zero 
            # (i.e. they are far enough from each other so that violation is ok)
            # otherwise metric learning might increase distance even further to minimize objective
            res += np.clip(self.dmaxmal-D, 0.0, None).sum()
            
        # add rayleigh prior over A matrix that parameterizes the metric
        # Omit logarithm of determinant of A that parameterizes the Mahalanobis distance
        # it would just encourage smaller values on diagonal, nothing more.
        # Then return
        return float(res)-rayleigh_prior(avec)

    def compute_error_assignment_cosine(self, Xproj, Y, assignation, clusters):
        '''
        Compute error to minimize the objective in terms of cluster assignment. 
        Some constants are ignored. 
        '''
        res = 0.0
        for k in clusters:
            idxs = assignation == k
            res -= np.dot(Xproj[idxs], Y[k]).sum()

        idxs_mustl_violated, idxs_cannotl_violated = indices_constraint_violated(self.constraint_matrix, assignation)

        # compute violated constraints. 
        # We use einsum to compute a fast product between two matrices row wise.
        # Constant 1.1 used instead of 1.0 due to numerical instability which can lead to the inner product between
        # normalized vectors to be greater than 1. 
        if len(idxs_mustl_violated) > 0:
            res += (1.1-np.einsum('ij,ij->i', Xproj[idxs_mustl_violated[0]], Xproj[idxs_mustl_violated[1]])).sum()
        if len(idxs_cannotl_violated) > 0:
            res += (2.0-(1.0-np.einsum('ij,ij->i', Xproj[idxs_cannotl_violated[0]], Xproj[idxs_cannotl_violated[1]]))).sum()

        # return error, ignore rayleigh_prior (it remains constant when metric does not change)
        return float(res)
    
    def compute_error_assignment_mahalanobis(self, Xproj, Y, assignation, clusters):
        '''
        Compute error to minimize the objective in terms of cluster assignment. 
        Some constants are ignored. 
        '''
        res = 0.0
        for k in clusters:
            idxs = assignation == k
            res += np.square(Xproj[idxs] - Y[k]).sum()

        idxs_mustl_violated, idxs_cannotl_violated = indices_constraint_violated(self.constraint_matrix, assignation)

        # compute violated constraints. 
        if len(idxs_mustl_violated) > 0:
            res += np.square(Xproj[idxs_mustl_violated[0]] - Xproj[idxs_mustl_violated[1]]).sum()
        if len(idxs_cannotl_violated) > 0:
            D = np.square(Xproj[idxs_cannotl_violated[0]] - Xproj[idxs_cannotl_violated[1]]).sum(1)
            # clip so none are below zero 
            # (i.e. they are far enough from each other so that violation is ok)
            # otherwise metric learning might increase distance even further to minimize objective
            res += np.clip(self.dmaxmal-D, 0.0, None).sum()

        # ignore rayleigh_prior and log det A (remain constant since metric does not change)
        # return error
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


def hmrf_kmeans(data, k, constraint_matrix,
                verbose = 0, a_init = None, par_kernel_jobs = 1,
                init_args = {}, convergence_args = {} ,metric='mahalanobis',
                solveroptions={'disp': None, 'iprint': -1, 'gtol': 1e-07, 'eps': 1e-08,
                                'maxiter': 200, 'ftol': 2.220446049250313e-09, 'maxcor': 10,
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
    if metric == 'euclidean':
        metric = 'mahalanobis'
    if not metric in ['cosine','mahalanobis']:
        raise ValueError("Metric unknown. Supported metrics are cosine and mahalanobis ")

    # Initialization
    if a_init is None:
        avec = np.ones(data.shape[1])
    else:
        if np.alltrue(a_init>=0):
            avec = a_init
        else:
            raise ValueError("resulting A not psd")

    iteration = 0
    ## Error to optimize
    error = HMRFkmeans_Error(data, metric, constraint_matrix)
    
    if metric == 'cosine':
        bounds = [[ 1.0e-07, 1.0]] * len(avec)  # setting bounds to eps or 0 may create instability in solver.
    else:
        bounds = [[ 1.0e-15, np.infty]] * len(avec)  # setting bounds to eps or 0 may create instability in solver.
        # pre-estimate dmax
        if data.shape[0]>500:
            error.dmaxmal = np.max(euclidean_distances(data[np.random.choice(data.shape[0],500)]))*1.2   
        else:
            error.dmaxmal = np.max(euclidean_distances(data))*1.2
    previous, error_current = None, None
    best_error, best_assignation = None, None

    ## Assignation
    assignation = Assignment_HMRFkmeans(error,metric)

    # we use a kernel matrix to initialize to reuse some code
    if metric== 'cosine':
        kernel = pairwise_kernels(np.multiply(data, np.sqrt(avec)), metric='cosine', n_jobs=par_kernel_jobs)
        assignation_current = Initialization(k, constraint_matrix).farthest_initialization(kernel)
        del kernel
    else:
        assignation_current = Euclidean_Initialization(k, constraint_matrix).farthest_initialization(np.multiply(data, np.sqrt(avec)))

    # M step: update centers
    error.update_centers_masks(data, assignation_current)
    print_verbose("Initial assignation : {}".format(assignation_current), verbose)

    while convergence_eps(previous, error_current, iteration, **convergence_args):
        previous = error_current

        ## M step : Minimize error via metric
        ### Initialize with last avec
        result = minimize(error.compute_error, avec, args=(data),
                           bounds=bounds,
                           options =solveroptions)

        print_verbose(result, verbose, level=1)
        avec = result.x

        ## E step : Assign each point to its cluster
        assignation_current = assignation.compute(avec, data, assignation_current)
        # M step: update centers
        error.update_centers_masks(data, assignation_current)
        error_current = error.compute_error(avec, data)

        iteration += 1
        print_verbose("Iteration: {} - Observed error: {}".format(iteration, error_current), verbose)
        print_verbose("Assignation : {}".format(assignation_current), verbose)
        print_verbose("avec : {}".format(avec), verbose)

        ## Update error 
        if best_error is None or error_current < best_error:
            best_error, best_assignation = error_current, assignation_current

    return best_assignation