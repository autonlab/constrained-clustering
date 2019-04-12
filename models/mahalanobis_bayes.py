import numpy as np
from utils import print_verbose

from GPyOpt.methods import BayesianOptimization
from sklearn.metrics.pairwise import linear_kernel
from models.farthest_kmeans import Initialization, kernelKmeans

def compute_alignment(A, B):
    """
        Arguments:
            A {array n * n} -- a Gram matrix
            B {array n * n} -- a Gram matrix
        
        Returns:
            score {float}
    """
    return np.dot(A.ravel(),B.ravel())

def mahalanobis_bayes_clustering(data, classes, constraint_matrix, bayes_iter = 1000, verbose = 0,scale=True):
    """
        Bayesian optimization on the space of combinasions of the given kernels
        With maximization of the constraint satisfaction on the observed constraints computed with constrained Kmeans

        Arguments:
            data {Array n * f} -- Data
            classes {int} -- Number of clusters to form
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

    # Compute the components implied by constrained (without any distance)
    initializer = Initialization(classes, constraint_matrix)
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
                weights_features /= weights_features.sum()

                # Transforms data
                transformed = np.multiply(data, np.sqrt(weights_features))

                # Compute new pairwise distances
                kernel = linear_kernel(transformed)

                # Computation assignation
                assignment = initializer.farthest_initialization(kernel, classes)
                assignations[step] = kernelKmeans(kernel, assignment, max_iteration = 100, verbose = verbose)
                
                # Computation score on observed constraints
                observed_constraint = 2 * np.equal.outer(assignations[step], assignations[step]) - 1.0
                score.append(compute_alignment(observed_constraint, constraint_matrix))

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