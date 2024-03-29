import numpy as np
from utils.utils import print_verbose
from GPyOpt.methods import BayesianOptimization
from sklearn.metrics.pairwise import cosine_similarity
import utils.constraint as constraint

def cosine_bayes_clustering(data, clustering, constraint_matrix, bayes_iter = 1000, verbose = 0):
    """
        Bayesian optimization on the space of combinasions of the given kernels
        With maximization of the KTA score on the observed constraints computed with Kmeans

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

    global assignations, step
    assignations, step = {}, 0
    def objective_KTA_sparse(subspace):
        """
            Computes the KTA for the combinations of kernels implied by
        """
        global assignations, step
        kta_score = []
        for weights_features in subspace:
            if weights_features.sum() == 0:
                # Avoid case where all weight are null
                kta_score.append(- np.inf)
            else:
                weights_features /= weights_features.sum()

                # Transforms data
                transformed = np.multiply(data, np.sqrt(weights_features))

                # Compute new cosine
                kernel = cosine_similarity(transformed)

                # Computation assignation
                assignations[step] = clustering.fit_transform(kernel)
                
                # Computation score on observed constraints
                kta_score.append(constraint.kta_score(constraint_matrix, assignations[step]))

                print_verbose("Step {}".format(step), verbose, level = 1)
                print_verbose("\t Alpha  : {}".format(weights_features), verbose, level = 1)
                print_verbose("\t KTA  : {}".format(kta_score[-1]), verbose)
            step += 1

        return - np.array(kta_score).reshape((-1, 1))
    
    # Constraint the scaling of each features to be between 0 and 1
    space =  [ {'name': 'var_{}'.format(j), 
                'type': 'continuous', 
                'domain': (0, 1.)}
            for j in range(features)]

    myBopt = BayesianOptimization(f = objective_KTA_sparse, 
        de_duplication = True,  # Avoids re-evaluating the objective at previous, pending or infeasible locations 
        domain = space)         # Domain to explore
    
    myBopt.run_optimization(max_iter = bayes_iter)

    return assignations[np.nanargmin(myBopt.Y)]