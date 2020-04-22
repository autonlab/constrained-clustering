import numpy as np
from numba import jit
from .utils import ConfidenceModel, print_verbose, GPModel
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression

@jit(nopython=True)
def get_random_candidate(number, dimensionality, non_zero):
    """
        Creates random candidates
        
        Arguments:
            number {int} -- Number of random candidates
        
        Returns:
            Array of weights for kernel combination (dimension: number * dimensionality)
    """
    epsilon = 0.001
    weights = np.zeros((number, dimensionality))
    for i in range(number):     
        number_components = np.random.randint(2, non_zero + 1)
        indices = np.random.choice(dimensionality, number_components, replace = False)
        weights_indices = [np.random.uniform(epsilon, 1) for k in indices]
        for j, w in zip(indices, weights_indices):
            weights[i, j] = w
    return weights

class CombinationKernelOptimizer:
    """
        Wrapper for optimizer 
        This object is abstract and should be implemented
    """
    
    @classmethod
    def create(cls, method, dimensionality, iteration = 1000, init_candidates = [], **args):
        """
            Optimizer factory
        """
        if method == "random":
            return CombinationKernelOptimizer(dimensionality = dimensionality, iteration = iteration, random_init = iteration - len(init_candidates), init_candidates = init_candidates, **args)
        elif method == "model":
            return ModelGuidedOptimization(dimensionality = dimensionality, iteration = iteration, init_candidates = init_candidates, **args)
        elif method == "gp":
            return ModelGuidedOptimization(dimensionality = dimensionality, iteration = iteration, init_candidates = init_candidates, model = GPModel(dimensionality), **args)
        else:
            print("Optimizer unknown")

    def __init__(self, objective_function, dimensionality, iteration = 1000,
                 init_candidates = [], random_init = 100, non_zero = 5, verbose = 0, **args):
        """
            Wrapper for optimizer initialization
            
            Arguments:
                objective_function {func} -- Function that evaluates the value
                dimensionality {int} -- Dimensionality of the candidates for optimization
                iteration {int} -- Number of iteration to use
                init_candidates {Array} -- Initial candidates to use
                non_zero {int} -- Maximum non zeros kernels in the combination
        """
        assert len(init_candidates) + random_init <= iteration, "No optimizer needed"
        
        self.objective_function = objective_function
        self.dimensionality = dimensionality
        self.iteration = iteration
        self.non_zero = non_zero
        self.verbose = verbose

        self.candidates = np.zeros((iteration, dimensionality))
        self.scores = np.zeros(iteration)
        
        # Add random points
        if random_init > 0:
            self.candidates[:random_init] = get_random_candidate(random_init, dimensionality, non_zero)
            self.scores[:random_init] = [objective_function(c) for c in self.candidates[:random_init]]
            
        # Add initial points
        for i, candidate in enumerate(init_candidates):
            self.candidates[random_init + i] = candidate
            self.scores[random_init + i] = objective_function(candidate)
        
        # Next eval should start at this level
        self.random_init = random_init + len(init_candidates)
        
    def run_optimization(self):
        """
            Runs the optimization and returns the best candidate
        """
        return self.get_best_candidate()
    
    def get_best_candidate(self):
        """
            Returns best candidate
        """
        best = np.nanargmax(self.scores)
        if best <= self.random_init:
            print_verbose("Best solution not obtained after optimization", self.verbose, level = 1)
        return self.candidates[best]

class ModelGuidedOptimization(CombinationKernelOptimizer):
    """
        Explores the space of possible candidates guided by the prediction of a model
    """
    
    def __init__(self, objective_function, dimensionality, iteration = 1000,
                 init_candidates = [], random_init = 100, non_zero = 5, verbose = 0,
                 model = None, acquisition_evals = 1000, exploration = 0.1, base_model = LinearRegression()):
        """
            Initialize model for evaluation
        
            Keyword Arguments:
                model {ConfidenceModel} -- Model which estimated confidence (default: {None})
                random_init {int} -- Number of random initialization (default: {10})
                acquisition_evals {int} -- Number of estimation (default: {10000})
                exploration {float} -- Parameter controlling exploration for UCB (Higher bound has more weight)
        """
        CombinationKernelOptimizer.__init__(self, objective_function, dimensionality, iteration, init_candidates, random_init, non_zero, verbose)
        if model is None:
            self.model = ConfidenceModel(
                            BaggingRegressor(base_model, n_estimators = 150),
                            acquisition_evals
                            )
        else:
            self.model = model
        self.acquisition_evals = acquisition_evals
        self.exploration = exploration
        
    def run_optimization(self):
        """
            Explores the space of possible values given a model predicting the
            estimated performances at this point in the space of possible values
        """
        for step in range(self.random_init, self.iteration):
            # Fit model on previous evaluation
            self.model.fit(self.candidates[:step], self.scores[:step])
            
            # Evaluate random sample
            potential_candidates = get_random_candidate(self.acquisition_evals, self.dimensionality, self.non_zero)
            predictions, confidence = self.model.predict_confidence(potential_candidates)
            
            # Compute the best candidate
            if step < self.iteration - 1:
                predictions += self.exploration * confidence
            index_candidate = np.argmax(predictions)
            self.candidates[step] = potential_candidates[index_candidate]
            self.scores[step] = self.objective_function(potential_candidates[index_candidate])
            
            print_verbose("Step {} - KTA {:.5f}".format(step, self.scores[step]), self.verbose, level = 1)
        
            if self.scores[step] == 1:
                print_verbose("Optimal solution obtained", self.verbose, level = 1)
                break
            
        return self.get_best_candidate()
