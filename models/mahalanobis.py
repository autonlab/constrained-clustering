import utils.constraint as constraint
import numpy as np
from utils.utils import print_verbose
from hyperopt import hp, tpe, fmin
from utils.clustering import Clustering
from GPyOpt.methods import BayesianOptimization
from sklearn.metrics.pairwise import linear_kernel

def mahalanobis_bayes_clustering(data, clustering, constraint_matrix, bayes_iter = 1000, verbose = 0, scale = True):
    """
        Bayesian optimization on the space of combinasions of the given kernels
        With maximization of the constraint satisfaction on the observed constraints computed with constrained Kmeans

        Arguments:
            data {Array n * f} -- Data
            clustering {Clustering} -- Clustering algo to use
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points

        Keyword Arguments:
            bayes_iter {int} -- Number of iteration to compute on the space (default: {1000})
                NB: Higher this number slower is the algorithm
            verbose {int} -- Level of verbosity (default: {0} -- No verbose)

        Returns:
            Assignation of length n, Assignation with enforced constraints
    """
    # Number features
    features = data.shape[1]

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    global assignations, step
    assignations, step = {}, 0
    def objective(subspace):
        """
            Compute constraint satisfaction
        """
        global assignations, step
        score = []
        for weights_features in subspace:
            if weights_features.sum() == 0:
                # Avoid case where all weight are null
                score.append(- np.inf)
            else:
                kernel_approx = np.multiply(data, weights_features)

                assignations[step] = clustering.fit_transform(kernel_approx)  
                score.append(constraint.kta_score(constraint_matrix, assignations[step]))

                print_verbose("Step {}".format(step), verbose, level = 1)
                print_verbose("\t Alpha  : {}".format(weights_features), verbose, level = 1)
                print_verbose("\t Alignment  : {}".format(score[-1]), verbose)
            step += 1

        return - np.array(score).reshape((-1, 1))
    
    # Constraint the scaling of each features to be between 0 and 1
    space =  [ {'name': 'var_{}'.format(j), 
                'type': 'continuous', 
                'domain': (0, 1.)}
            for j in range(features)]

    myBopt = BayesianOptimization(f = objective, 
        de_duplication = True,  # Avoids re-evaluating the objective at previous, pending or infeasible locations 
        domain = space)         # Domain to explore
    
    myBopt.run_optimization(max_iter = bayes_iter)

    return assignations[np.nanargmin(myBopt.Y)]

def mahalanobis_tpe_clustering(data, clustering, constraint_matrix, iterations = 1000, verbose = 0,scale = True):
    """
        Bayesian optimization on the space of combinasions of the given kernels
        With maximization of the constraint satisfaction on the observed constraints computed with constrained Kmeans

        Arguments:
            data {Array n * f} -- Data
            clustering {Clustering} -- Clustering algo to use
            constraint_matrix {Array n * n} -- Constraint matrix with value between -1 and 1 
                Positive values represent must link points
                Negative values represent should not link points

        Keyword Arguments:
            bayes_iter {int} -- Number of iteration to compute on the space (default: {1000})
                NB: Higher this number slower is the algorithm
            verbose {int} -- Level of verbosity (default: {0} -- No verbose)

        Returns:
            Assignation of length n, Assignation with enforced constraints
    """
    # Number features
    features = data.shape[1]

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    def objective_custom_randomweights_hp(args):
        """
            Computes the KTA for the combinations of kernels implied by
        """
        weights = np.array([x for x in args])
        if weights.sum() == 0:
            # Avoid case where all weight are null
            return  np.array(1)
        else:
            # Compute new kernel
            kernel_approx = np.multiply(data, weights)

            assignation = clustering.fit_transform(kernel_approx)
            kta = constraint.kta_score(constraint_matrix, assignation)

            print_verbose("\t Alpha  : {}".format(weights), verbose, level = 1)
            print_verbose("\t Alignment  : {}".format(kta), verbose)

            return - kta
    
    # Constraint the scaling of each features to be between 0 and 1
    space = [hp.uniform('x%d'%i, 0, 1) for i in range(features)]

    best = fmin(fn = objective_custom_randomweights_hp,
            space = space,
            algo = tpe.suggest,
            max_evals = iterations)

    beta_best = np.array([best['x%d'%i] for i in range(features)])

    # Compute new kernel
    kernel_approx = np.multiply(data, beta_best)
    return clustering.fit_transform(kernel_approx)
